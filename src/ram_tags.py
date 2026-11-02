"""Optional, reusable Recognize Anything Model tagger for front/down RGB images."""
from contextlib import redirect_stderr, redirect_stdout
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

from src.config import ROOT, load_config
from src.cuda_resources import model_load_guard


def _rgb_image(value):
    if isinstance(value, (str, Path)):
        with Image.open(value) as image:
            return image.convert('RGB')
    if isinstance(value, Image.Image):
        return value.convert('RGB')
    if isinstance(value, np.ndarray) and value.dtype == np.uint8 and value.ndim == 3 and value.shape[2] == 3:
        return Image.fromarray(value)
    raise ValueError('RGB input must be an image path, PIL image, or uint8 HxWx3 RGB array')


class RAMViewTagger:
    """Tag two complete RGB views in one batch; no depth, cloud, files, or logs.

    tag(front_rgb, down_rgb) returns front/down dictionaries containing English
    `tags` and Chinese `tags_zh` lists. The official RAM model is loaded once on
    first use. Its original per-class thresholds and vocabulary are preserved.
    """
    def __init__(self, config=None, *, device='cuda:0', init_lock=None, min_free_gib=0., memory_wait=300.):
        self.config = config if isinstance(config, dict) else load_config(config)
        self.device = device
        self.init_lock, self.min_free_gib, self.memory_wait = init_lock, min_free_gib, memory_wait
        self._model = self._transform = None

    def _load(self):
        settings = self.config.get('ram', {})
        source = (ROOT / os.path.expandvars(os.path.expanduser(settings.get('source', 'third_party/recognize-anything')))).resolve()
        checkpoint = (ROOT / os.path.expandvars(os.path.expanduser(settings.get('checkpoint', 'assets/checkpoints/ram_swin_large_14m.pth')))).resolve()
        if not (source / 'ram/models/ram.py').is_file() or not checkpoint.is_file():
            raise FileNotFoundError('RAM source/checkpoint missing; follow docs/RAM_TAGS.md and configure ram.source / ram.checkpoint')
        expected = json.loads((ROOT / 'config/sources.json').read_text())['ram']['commit']
        actual = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != expected:
            raise RuntimeError(f'RAM source commit {actual} != pinned {expected}')
        sys.path.insert(0, str(source))
        official = importlib.import_module('ram')
        if Path(official.__file__).resolve() != source / 'ram/__init__.py':
            raise RuntimeError('A different ram package is already imported; use the configured official source')
        factory = importlib.import_module('ram.models').ram
        import torch
        if self.device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; use --device cpu or run with GPU access')
        os.environ.setdefault('HF_HUB_OFFLINE', '1')
        with model_load_guard(self.device, self.init_lock, self.min_free_gib, self.memory_wait):
            model = factory(pretrained=str(checkpoint), image_size=384, vit='swin_l',
                            text_encoder_type=self.config['paths']['bert'])
            self._model = model.eval().to(self.device)
        self._transform = official.get_transform(image_size=384)

    def tag(self, front_rgb, down_rgb, *, verbose=False):
        """Return full-image RAM tags, independently of the action cone masks."""
        images = [_rgb_image(front_rgb), _rgb_image(down_rgb)]
        with open(os.devnull, 'w') as quiet:
            sink = sys.stderr if verbose else quiet
            with redirect_stdout(sink), redirect_stderr(sink):
                if self._model is None:
                    self._load()
                import torch
                batch = torch.stack([self._transform(image) for image in images]).to(self.device)
                with torch.inference_mode():
                    english, chinese = self._model.generate_tag(batch)
        return {name: {'tags': [tag.strip() for tag in en.split('|') if tag.strip()],
                       'tags_zh': [tag.strip() for tag in zh.split('|') if tag.strip()]}
                for name, en, zh in zip(('front', 'down'), english, chinese)}
