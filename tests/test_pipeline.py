"""Exercise real import/fusion/query with a labeled-image fixture instead of ML.

Actual GroundingDINO+MobileSAM is separately smoke-tested on a real GPU sample.
"""
from itertools import combinations
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from src.config import ROOT, load_config
from src.openuav.capture import OPENUAV_CAMERAS
from src.openuav.dataset import load_dataset_frame
from src.pipeline import GSAActionPipeline


@pytest.fixture
def sample(tmp_path):
    source = tmp_path / 'sample'
    for camera in OPENUAV_CAMERAS:
        name = camera.lower()
        (source / name).mkdir(parents=True)
        (source / (name+'_depth')).mkdir()
        Image.fromarray(np.full((16, 16, 3), [40, 100, 180], np.uint8)).save(source / name / '000001.png')
        depth = np.tile(np.arange(16, dtype=np.uint8) % 4+20, (16, 1))
        Image.fromarray(depth).save(source / (name+'_depth') / '000001.png')
    return source


@pytest.fixture
def runner(monkeypatch):
    class LabelFixture:
        loaded = 0

        def __init__(self, config, classes, device):
            type(self).loaded += 1
            self.classes, self.settings, self.device = classes, config['segmentation'], device

        def segment(self, image_path, max_detections):
            rgb = image_path if isinstance(image_path, np.ndarray) else np.asarray(Image.open(image_path).convert('RGB'))
            shape = rgb.shape[:2]
            return (rgb, np.ones(shape, np.uint16), np.ones(shape, np.uint16),
                    np.full(shape, .9, np.float32), [])

    monkeypatch.setattr('src.pipeline.GroundedMobileSAM', LabelFixture)
    config = load_config(ROOT / 'config/resources.example.json')
    config['cloud'].update(stride=1, voxel_size=0)
    pipeline = GSAActionPipeline(config, device='cpu', classes=['tree'], gravity_aligned=False)
    return pipeline, LabelFixture


def test_raw_sample_needs_only_ten_images_and_infers_frame(sample, runner, tmp_path):
    pipeline, model = runner
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in sample.rglob('*.png')}
    output = tmp_path / 'result'
    rows = pipeline.run(sample, output=output)
    assert len(rows) == 4 and any(row['semantic_name'] == 'tree' for row in rows)
    assert model.loaded == 1
    assert json.loads((output / 'action_values.json').read_text()) == rows
    report = json.loads((output / 'pipeline.json').read_text())
    assert report['status'] == 'completed' and report['world_pose_known'] is False
    assert report['vehicle_pose_reference'] == 'body_origin_identity' and not report['full_sphere_allocated']
    assert (output/'action_features.npz').is_file()
    assert not (output/'cloud').exists()
    assert not (output / 'cloud/cloud.html').exists()
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in before.items())
    mask_meta = json.loads((output / 'masks/meta.json').read_text())
    assert mask_meta['capture_meta_sha256'] == hashlib.sha256((output / 'capture/meta.json').read_bytes()).hexdigest()
    # A prepared metric capture runs through the same API, reusing the model.
    assert pipeline.run(output / 'capture') == rows
    assert model.loaded == 1


def test_world_pose_is_not_needed_for_body_relative_geometry(sample, runner, tmp_path):
    pipeline, _ = runner
    pipeline.config['cloud']['voxel_size'] = .15
    local = tmp_path / 'local'
    first = pipeline.run(sample, output=local)
    (sample / 'log').mkdir()
    state = {'position': [200., -30., -15.], 'orientation': [.1, .2, .3, .9]}
    (sample / 'log/000001.json').write_text(json.dumps({'sensors': {'state': state}}))
    world = tmp_path / 'world'
    second = pipeline.run(sample, frame=1, output=world)
    assert [r['semantic_id'] for r in first] == [r['semantic_id'] for r in second]
    np.testing.assert_allclose([r['depth_m'] for r in first], [r['depth_m'] for r in second], atol=2e-6)
    assert json.loads((world / 'pipeline.json').read_text())['world_pose_known'] is True


def test_original_dataset_importer_still_requires_pose_by_default(sample):
    with pytest.raises(FileNotFoundError):
        load_dataset_frame(sample, '000001')


def test_bad_input_and_nonempty_output_do_not_load_model(sample, runner, tmp_path):
    pipeline, model = runner
    (sample / 'frontcamera_depth/000001.png').unlink()
    output = tmp_path / 'failed'
    with pytest.raises(FileNotFoundError):
        pipeline.run(sample, frame='000001', output=output)
    assert model.loaded == 0
    assert json.loads((output / 'pipeline.json').read_text())['status'] == 'failed'
    assert not (output / 'action_values.json').exists()
    with pytest.raises(ValueError, match='empty or absent'):
        pipeline.run(sample, output=output)
    assert model.loaded == 0


def test_cli_keeps_progress_on_stderr_and_json_on_stdout(monkeypatch, capsys):
    from scripts import run_pipeline
    expected = [{'action': 'front', 'depth_m': None, 'semantic_id': -1, 'semantic_name': 'no_return'}]

    class Runner:
        def __init__(self, *args, **kwargs):
            print('model progress')

        def run(self, *args, **kwargs):
            assert kwargs['ram_tags'] is True
            return {'actions': expected, 'ram_tags': {'front': {'tags': ['road']}, 'down': {'tags': []}}}

    monkeypatch.setattr(run_pipeline, 'GSAActionPipeline', Runner)
    assert run_pipeline.main(['--input', 'sample']) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)['actions'] == expected and 'model progress' in captured.err
    assert 'ram_tags' in json.loads(captured.out)


def test_default_front_down_input_is_quiet_and_memory_only(sample, runner):
    import shutil
    pipeline, _ = runner
    for name in ('leftcamera', 'rightcamera', 'rearcamera'):
        shutil.rmtree(sample/name); shutil.rmtree(sample/(name+'_depth'))
    assert pipeline.cameras == ('FrontCamera', 'DownCamera')
    before = sorted(str(p) for p in sample.parent.rglob('*'))
    rows = pipeline.run(sample)
    assert [row['action'] for row in rows] == ['front', 'L30', 'R30', 'DOWN']
    assert all(row['depth_m'] is not None for row in rows)
    assert sorted(str(p) for p in sample.parent.rglob('*')) == before
    assert pipeline.last_report['in_memory'] and not pipeline.last_report['full_point_cloud_generated']
    assert pipeline.last_cloud is None and not pipeline.last_report['point_cloud_generated']
    assert pipeline.last_features['sample_count'].shape == (4,)
    assert all('xyz' not in samples for samples in pipeline.last_samples.values())


def test_all_far_points_skip_model_and_scene_point_generation(sample, runner):
    pipeline, model = runner
    for path in sample.glob('*_depth/*.png'):
        Image.fromarray(np.full((16, 16), 200, np.uint8)).save(path)
    rows = pipeline.run(sample)
    assert model.loaded == 0 and all(row['depth_m'] is None for row in rows)
    assert pipeline.last_report['selected_depth_pixels'] == 0
    assert pipeline.last_cloud is None


def test_optional_ram_tags_preserve_action_rows_and_use_full_rgb(sample, runner, monkeypatch, tmp_path):
    from src.ram_tags import RAMViewTagger
    pipeline, _ = runner
    calls = []
    tags = {'front': {'tags': ['road'], 'tags_zh': ['道路']}, 'down': {'tags': ['grass'], 'tags_zh': ['草地']}}
    def tag(self, front, down, **kwargs):
        calls.append((front.shape, down.shape))
        return tags
    monkeypatch.setattr(RAMViewTagger, 'tag', tag)
    actions = pipeline.run(sample)
    assert pipeline.last_ram_tags is None and pipeline._ram is None
    result = pipeline.run(sample, ram_tags=True)
    assert result == {'actions': actions, 'ram_tags': tags}
    assert pipeline.last_ram_tags == tags and calls == [((16, 16, 3), (16, 16, 3))]
    ram = pipeline._ram
    output = tmp_path/'ram_output'
    assert pipeline.run(sample, ram_tags=True, output=output) == result
    assert pipeline._ram is ram
    assert json.loads((output/'action_values.json').read_text()) == actions
    assert json.loads((output/'ram_tags.json').read_text()) == tags


def test_ram_tagging_runs_even_when_all_depth_is_outside_action_radius(sample, runner, monkeypatch):
    from src.ram_tags import RAMViewTagger
    pipeline, model = runner
    for path in sample.glob('*_depth/*.png'):
        Image.fromarray(np.full((16, 16), 200, np.uint8)).save(path)
    tags = {'front': {'tags': ['sky'], 'tags_zh': ['天空']}, 'down': {'tags': [], 'tags_zh': []}}
    monkeypatch.setattr(RAMViewTagger, 'tag', lambda *args, **kwargs: tags)
    result = pipeline.run(sample, ram_tags=True)
    assert model.loaded == 0 and all(row['depth_m'] is None for row in result['actions'])
    assert result['ram_tags'] == tags


def test_direct_pipeline_never_backprojects_fuses_or_voxelizes(sample, runner, monkeypatch, tmp_path):
    pipeline, _ = runner
    def forbidden(*args, **kwargs):
        pytest.fail('The action wrapper must not generate or fuse a point cloud')
    monkeypatch.setattr('src.geometry.backproject', forbidden)
    monkeypatch.setattr('src.geometry.voxel_representatives', forbidden)
    monkeypatch.setattr('src.cloud.fuse_action_observations', forbidden)
    first = pipeline.run(sample)
    counts = pipeline.last_features['sample_count'].copy()
    pipeline.config['cloud']['voxel_size'] = 999.
    output = tmp_path/'direct'
    assert pipeline.run(sample, output=output) == first
    np.testing.assert_array_equal(pipeline.last_features['sample_count'], counts)
    assert not (output/'cloud').exists() and pipeline.last_cloud is None
    assert pipeline.last_report['backprojected_pixels'] == 0
    assert pipeline.last_report['semantic_vote_unit'] == 'sampled_pixel'


CAMERA_SUBSETS = [subset for count in range(1, 6) for subset in combinations(OPENUAV_CAMERAS, count)]
CARDINAL_QUERIES = [dict(name=name, yaw_deg=yaw, elevation_deg=elevation) for name, yaw, elevation in
                   [('front', 0, 0), ('left', -90, 0), ('right', 90, 0), ('rear', 180, 0), ('down', 0, -90)]]


@pytest.mark.parametrize('cameras', CAMERA_SUBSETS, ids=lambda cameras: '+'.join(cameras))
def test_all_31_view_subsets_accept_only_selected_files_and_roundtrip(sample, runner, tmp_path, cameras):
    import shutil
    pipeline, model = runner
    for camera in set(OPENUAV_CAMERAS)-set(cameras):
        shutil.rmtree(sample/camera.lower())
        shutil.rmtree(sample/(camera.lower()+'_depth'))
    pipeline = GSAActionPipeline(pipeline.config, device='cpu', classes=['tree'], cameras=cameras,
                                 directions=CARDINAL_QUERIES, gravity_aligned=False)
    output = tmp_path/'selected'
    rows = pipeline.run(sample, output=output)
    assert [row['action'] for row in rows] == [query['name'] for query in CARDINAL_QUERIES]
    assert [row['semantic_id'] > 0 for row in rows] == [camera in cameras for camera in OPENUAV_CAMERAS]
    assert pipeline.last_report['cameras'] == list(cameras)
    assert set(pipeline.last_report['inference_cameras']) == set(cameras)
    assert not pipeline.last_report['point_cloud_generated']
    assert pipeline.last_features['sample_count'].shape == (5,)
    assert set(json.loads((output/'capture/meta.json').read_text())['cameras']) == set(cameras)
    assert pipeline.run(output/'capture') == rows
    assert model.loaded == 1
    with np.load(output/'action_features.npz', allow_pickle=False) as features:
        assert features['action_names'].tolist() == [query['name'] for query in CARDINAL_QUERIES]
        np.testing.assert_allclose(features['direction_frd'], pipeline.direction_frd)


def test_cli_single_view_automatically_skips_ram_and_reads_directions(monkeypatch, capsys, tmp_path):
    from scripts import run_pipeline
    directions = [dict(name='left', yaw_deg=-90, elevation_deg=0)]
    queries = tmp_path/'directions.json'
    queries.write_text(json.dumps(directions))
    expected = [dict(action='left', depth_m=3., semantic_id=1, semantic_name='tree')]

    class Runner:
        def __init__(self, *args, **kwargs):
            assert kwargs['cameras'] == ['LeftCamera']
            assert kwargs['directions'] == directions

        def run(self, *args, **kwargs):
            assert not kwargs.get('ram_tags', False)
            return expected

    monkeypatch.setattr(run_pipeline, 'GSAActionPipeline', Runner)
    assert run_pipeline.main(['--input', 'sample', '--cameras', 'LeftCamera', '--directions', str(queries)]) == 0
    assert json.loads(capsys.readouterr().out) == expected


def test_configured_queries_and_explicit_ram_requirement(sample, runner):
    pipeline, _ = runner
    config = dict(pipeline.config, actions={'directions': CARDINAL_QUERIES[-2:]})
    rear = GSAActionPipeline(config, device='cpu', classes=['tree'], cameras=['RearCamera'])
    rows = rear.run(sample)
    assert [row['action'] for row in rows] == ['rear', 'down']
    assert rows[0]['depth_m'] is not None and rows[1]['depth_m'] is None
    with pytest.raises(ValueError, match='RAM view tagging requires'):
        rear.run(sample, ram_tags=True)
    with pytest.raises(ValueError, match='distinct standard'):
        GSAActionPipeline(config, cameras=[])


@pytest.mark.parametrize('cameras', [('LeftCamera',), ('RearCamera',), ('FrontCamera', 'DownCamera')])
def test_capture_cli_imports_selected_views_without_a_pose_log(sample, tmp_path, cameras):
    import subprocess
    import sys
    import shutil
    for camera in set(OPENUAV_CAMERAS)-set(cameras):
        shutil.rmtree(sample/camera.lower())
        shutil.rmtree(sample/(camera.lower()+'_depth'))
    output = tmp_path/'cli_capture'
    result = subprocess.run([sys.executable, 'scripts/capture_openuav.py', '--config',
        'config/resources.example.json', '--from-dataset', '--allow-quantized-depth',
        '--trajectory', str(sample), '--frame', '000001', '--cameras', *cameras,
        '--no-render', '--output', str(output)], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    metadata = json.loads((output/'meta.json').read_text())
    assert list(metadata['cameras']) == list(cameras) and not metadata['world_pose_known']
    with np.load(output/'depth_cloud/cloud.npz', allow_pickle=False) as cloud:
        assert len(cloud['xyz']) > 0
        assert set(cloud['view_id']) == set(range(len(cameras)))
