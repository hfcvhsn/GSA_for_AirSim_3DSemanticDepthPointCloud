"""RAM view association, batching, quiet operation, and optional pipeline contract."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from src.ram_tags import RAMViewTagger


def test_two_rgb_views_are_batched_without_losing_the_down_tags(tmp_path, monkeypatch, capsys):
    import torch
    front = tmp_path/'front.png'
    Image.fromarray(np.full((8, 8, 3), 30, np.uint8)).save(front)
    down = np.full((8, 8, 3), 90, np.uint8)
    loaded, batches = [], []
    class Model:
        def generate_tag(self, batch):
            batches.append(batch.cpu().numpy())
            print('official RAM progress')
            return ['building | road', 'grass'], ['建筑物 | 道路', '草地']
    def load(self):
        loaded.append(True)
        print('official model loading')
        self._model = Model()
        self._transform = lambda image: torch.from_numpy(np.array(image)).permute(2, 0, 1).float()
    monkeypatch.setattr(RAMViewTagger, '_load', load)
    tagger = RAMViewTagger(device='cpu')
    expected = {'front': {'tags': ['building', 'road'], 'tags_zh': ['建筑物', '道路']},
                'down': {'tags': ['grass'], 'tags_zh': ['草地']}}
    before = sorted(tmp_path.rglob('*'))
    assert tagger.tag(front, down) == expected
    assert tagger.tag(Image.open(front), down) == expected
    assert len(loaded) == 1 and all(batch.shape == (2, 3, 8, 8) for batch in batches)
    assert batches[0][0].mean() == 30 and batches[0][1].mean() == 90
    assert sorted(tmp_path.rglob('*')) == before
    captured = capsys.readouterr()
    assert captured.out == captured.err == ''


@pytest.mark.parametrize('bad', [None, np.zeros((8, 8), np.uint8), np.zeros((8, 8, 3), np.float32)])
def test_invalid_rgb_fails_before_loading_ram(bad, monkeypatch):
    def fail(self):
        pytest.fail('Bad input must not load a model')
    monkeypatch.setattr(RAMViewTagger, '_load', fail)
    with pytest.raises(ValueError, match='RGB input'):
        RAMViewTagger(device='cpu').tag(bad, Image.new('RGB', (8, 8)))


def test_ram_cli_json_only_and_optional_output(tmp_path, monkeypatch, capsys):
    from scripts import tag_rgb_views
    expected = {'front': {'tags': ['road'], 'tags_zh': ['道路']}, 'down': {'tags': [], 'tags_zh': []}}
    seen = []
    class Tagger:
        def __init__(self, config, device):
            assert device == 'cpu'
        def tag(self, front, down, verbose):
            seen.append((front, down))
            return expected
    monkeypatch.setattr(tag_rgb_views, 'RAMViewTagger', Tagger)
    out = tmp_path/'tags.json'
    assert tag_rgb_views.main(['--front-rgb', 'front.png', '--down-rgb', 'down.png', '--device', 'cpu', '--output', str(out)]) == 0
    captured = capsys.readouterr()
    assert captured.err == '' and json.loads(captured.out) == expected
    assert json.loads(out.read_text()) == expected
    assert seen == [(Path('front.png'), Path('down.png'))]
