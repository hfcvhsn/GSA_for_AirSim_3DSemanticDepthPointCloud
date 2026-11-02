"""Startup scheduling contracts; no CUDA device or model is used."""
import fcntl
import sys
from types import SimpleNamespace as NS

import pytest

import src.cuda_resources as resources


@pytest.fixture
def cuda(monkeypatch):
    state = NS(t=0.,free=[16*1024**3],devices=[])
    def memory(device):
        free=state.free[0]
        if len(state.free)>1:state.free.pop(0)
        return free,48*1024**3
    monkeypatch.setitem(sys.modules,'torch',NS(cuda=NS(set_device=state.devices.append,mem_get_info=memory)))
    monkeypatch.setattr(resources.time,'monotonic',lambda:state.t)
    monkeypatch.setattr(resources.time,'sleep',lambda seconds:setattr(state,'t',state.t+seconds))
    return state


def test_waits_for_memory_then_loads_without_changing_device(cuda,tmp_path):
    cuda.free=[1024**3,16*1024**3]
    with resources.model_load_guard('cuda:0',tmp_path/'init.lock',8,20) as info:
        assert info['free_bytes_before']==16*1024**3
        assert info['wait_s']==5
    assert cuda.devices==['cuda:0','cuda:0']


def test_memory_timeout_does_not_start_model(cuda,tmp_path):
    cuda.free=[1024**3]
    with pytest.raises(RuntimeError,match='startup timeout'):
        with resources.model_load_guard('cuda:0',tmp_path/'init.lock',8,5):
            pytest.fail('A model must not start below the configured free-memory floor')


def test_concurrent_startup_lock_and_exception_cleanup(cuda,tmp_path):
    path=tmp_path/'init.lock'
    with path.open('a') as held:
        fcntl.flock(held,fcntl.LOCK_EX)
        with pytest.raises(RuntimeError,match='another GSA model'):
            with resources.model_load_guard('cuda:0',path,8,5):pytest.fail('Lock was bypassed')
    assert not cuda.devices
    with pytest.raises(RuntimeError,match='fixture model OOM'):
        with resources.model_load_guard('cuda:0',path,8,5):raise RuntimeError('fixture model OOM')
    with resources.model_load_guard('cuda:0',path,8,0) as info:
        assert info['free_bytes_before']>=8*1024**3


def test_cpu_path_needs_no_torch_and_rejects_invalid_settings(monkeypatch):
    monkeypatch.setitem(sys.modules,'torch',None)
    with resources.model_load_guard('cpu') as info:assert info is None
    for value in (-1,float('nan')):
        with pytest.raises(ValueError):
            with resources.model_load_guard('cpu',min_free_gib=value):pass
