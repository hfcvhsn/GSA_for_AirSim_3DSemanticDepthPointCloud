"""Bound CUDA model startup without changing model precision or input data."""
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import sys
import time


@contextmanager
def model_load_guard(device, lock_path=None, min_free_gib=0., timeout=300.):
    if not math.isfinite(min_free_gib) or min_free_gib < 0 or not math.isfinite(timeout) or timeout < 0:
        raise ValueError('Invalid CUDA startup memory/timeout setting')
    if not device.startswith('cuda'):
        yield None
        return
    import torch
    started = time.monotonic()
    lock = None
    if lock_path is not None:
        path = Path(lock_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = path.open('a')
    def log(event, **fields):
        print('[GSA_STARTUP] '+json.dumps(dict(event=event,device=device,
            cuda_visible=os.environ.get('CUDA_VISIBLE_DEVICES'),wait_s=time.monotonic()-started,**fields)),
            file=sys.stderr,flush=True)
    def wait(reason):
        remaining = timeout-(time.monotonic()-started)
        if remaining <= 0:
            raise RuntimeError(f'GSA CUDA startup timeout ({reason}); required free memory {min_free_gib:g} GiB')
        log('waiting',reason=reason,required_free_gib=min_free_gib)
        time.sleep(min(5.,remaining))
    try:
        while True:
            if lock is not None:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    wait('another GSA model is initializing')
                    continue
            log('cuda_init')
            # Initialize before CPU checkpoint loading; keep concurrent first
            # CUDA contexts/model transfers behind the optional shared lock.
            torch.cuda.set_device(device)
            free,total = torch.cuda.mem_get_info(device)
            if free >= min_free_gib*1024**3:
                info=dict(free_bytes_before=free,total_bytes=total,
                          required_free_gib=min_free_gib,wait_s=time.monotonic()-started)
                log('model_load',free_gib=free/1024**3,total_gib=total/1024**3)
                yield info
                log('model_ready')
                return
            if lock is not None:fcntl.flock(lock,fcntl.LOCK_UN)
            wait(f'free {free/1024**3:.2f} GiB')
    finally:
        if lock is not None:lock.close()
