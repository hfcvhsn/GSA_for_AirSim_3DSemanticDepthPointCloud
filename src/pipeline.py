"""Front/down RGB-D -> direct per-pixel action pooling and full-image RAM tags."""
import copy
import csv
import hashlib
import json
import math
import os
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

from src.actions import (DEFAULT_DIRECTIONS, DEFAULT_WINDOW_DEG, RADIUS_M, direction_vectors,
                         format_action_rows, pool_action_samples, validate_directions)
from src.config import ROOT, load_config, write_json
from src.cuda_resources import model_load_guard
from src.geometry import action_depth_samples, body_to_action_rotation
from src.gsa import GroundedMobileSAM, palette, save_segmentation, validate_classes
from src.openuav.capture import CameraCapture, OPENUAV_CAMERAS, save_capture
from src.openuav.dataset import load_dataset_frame

DEFAULT_CAMERAS = ('FrontCamera', 'DownCamera')


class GSAActionPipeline:
    """Reuse models and directly pool depth pixels; no XYZ, cloud fusion or voxels.

    Core run() returns direction rows; ram_tags=True returns {actions, ram_tags}.
    The default queries are front/L30/R30/DOWN. Files and logs are opt-in.
    """
    def __init__(self, config=None, *, device='cuda:0', classes=None,
                 window_deg=DEFAULT_WINDOW_DEG, kde_bandwidth=.4, max_detections=50,
                 cameras=None, gravity_aligned=True, init_lock=None, min_free_gib=0., memory_wait=300.,
                 directions=None):
        self.config = copy.deepcopy(config) if isinstance(config, dict) else load_config(config)
        self.classes = validate_classes(classes if classes is not None else self.config['segmentation']['classes'])
        if not math.isfinite(window_deg) or not 0 < window_deg <= 180:
            raise ValueError('Require finite 0 < window_deg <= 180')
        if not math.isfinite(kde_bandwidth) or kde_bandwidth <= 0:
            raise ValueError('Require finite positive KDE bandwidth')
        if not isinstance(max_detections, int) or not 1 <= max_detections <= 65535:
            raise ValueError('max_detections must be an integer in [1, 65535]')
        self.cameras = tuple(DEFAULT_CAMERAS if cameras is None else cameras)
        if not self.cameras or len(set(self.cameras)) != len(self.cameras) or not set(self.cameras) <= set(OPENUAV_CAMERAS):
            raise ValueError('Choose distinct standard OpenUAV cameras')
        self.directions = validate_directions(directions if directions is not None else
                                             self.config.get('actions', {}).get('directions'))
        self.direction_frd = direction_vectors(self.directions)
        self.device, self.window_deg, self.kde_bandwidth = device, window_deg, kde_bandwidth
        self.max_detections, self.gravity_aligned = max_detections, gravity_aligned
        self.init_lock, self.min_free_gib, self.memory_wait = init_lock, min_free_gib, memory_wait
        self._model = self._ram = None
        self.last_report = self.last_features = self.last_samples = self.last_ram_tags = None
        self.last_cloud = None  # Compatibility: this interface no longer creates clouds.

    def tag_rgb_views(self, front_rgb, down_rgb, *, verbose=False):
        """RGB-only RAM interface, independent of depth and action visibility."""
        if self._ram is None:
            from src.ram_tags import RAMViewTagger
            self._ram = RAMViewTagger(self.config, device=self.device, init_lock=self.init_lock,
                min_free_gib=self.min_free_gib, memory_wait=self.memory_wait)
        self.last_ram_tags = self._ram.tag(front_rgb, down_rgb, verbose=verbose)
        return self.last_ram_tags

    def _load(self, source, frame, camera_layout):
        meta_path = source / 'meta.json'
        if meta_path.is_file():
            if frame is not None:
                raise ValueError('--frame applies to a dataset directory, not a prepared capture')
            metadata = json.loads(meta_path.read_text())
            if not set(self.cameras) <= set(metadata['cameras']):
                raise ValueError(f'Capture lacks selected cameras: {self.cameras}')
            metadata['cameras'] = {name: metadata['cameras'][name] for name in self.cameras}
            observations = [CameraCapture(name, np.asarray(Image.open(source/name/'rgb.png').convert('RGB')),
                            np.load(source/name/'depth_perspective_m.npy', mmap_mode='r', allow_pickle=False),
                            metadata['cameras'][name]) for name in self.cameras]
        else:
            if self.config['dataset']['depth_encoding'] != 'uint8_0_100m':
                raise ValueError('Dataset PNG import requires uint8_0_100m; use a prepared capture for metric NPY')
            if frame is None:
                folders = [source/(name.lower()+suffix) for name in self.cameras for suffix in ('', '_depth')]
                frames = set.intersection(*(set(p.stem for p in folder.glob('*.png')) for folder in folders))
                if len(frames) != 1:
                    raise ValueError('Supply --frame unless exactly one complete selected-view frame exists')
                frame = next(iter(frames))
            if not str(frame).isdigit():
                raise ValueError('Frame must be a numeric ID, e.g. 000145')
            observations, metadata = load_dataset_frame(source, frame, self.config['dataset']['fov_deg'],
                camera_layout, allow_missing_pose=True, cameras=self.cameras)
            metadata['cameras'] = {obs.name: obs.metadata for obs in observations}
        for obs in observations:
            if obs.rgb.shape[:2] != obs.depth_perspective_m.shape:
                raise ValueError(f'Unregistered RGB/depth shape: {obs.name}')
        if metadata.get('source') == 'airsim_rpc' and not metadata.get('pose_synchronized_by_pause'):
            raise ValueError('Live pose is not synchronized; capture with --pause')
        return observations, metadata

    def run(self, input, *, frame=None, output=None, camera_layout=None, verbose=False, ram_tags=False):
        """Return configured direction depth/semantics, optionally with RAM image tags."""
        if ram_tags and not {'FrontCamera', 'DownCamera'} <= set(self.cameras):
            raise ValueError('RAM view tagging requires FrontCamera and DownCamera')
        source = Path(input).expanduser()
        if not source.is_absolute() and not source.exists():
            source = Path(self.config['paths']['dataset_root']) / source
        source = source.resolve()
        if not source.is_dir():
            raise ValueError(f'Missing input directory: {source}')
        if output is not None:
            output = Path(output).expanduser().resolve()
            if output.exists() and (not output.is_dir() or any(output.iterdir())):
                raise ValueError(f'Output directory must be empty or absent: {output}')
            output.mkdir(parents=True, exist_ok=True)
        for key, value in (('HF_HOME', ROOT/'.cache/huggingface'), ('TORCH_HOME', ROOT/'.cache/torch'),
                           ('HF_HUB_OFFLINE', '1'), ('TOKENIZERS_PARALLELISM', 'false')):
            os.environ.setdefault(key, str(value))
        self.last_features = self.last_samples = self.last_ram_tags = None
        with open(os.devnull, 'w') as quiet:
            sink = sys.stderr if verbose else quiet
            with redirect_stdout(sink), redirect_stderr(sink):
                return self._run(source, frame, output, camera_layout, ram_tags)

    def _run(self, source, frame, output, camera_layout, ram_tags):
        started = time.perf_counter()
        report = dict(schema_version=5,
            feature_schema='four-action-pixels-v5' if self.directions == DEFAULT_DIRECTIONS else 'direction-pixels-v1',
            status='running', stage='input',
            input=str(source), device=self.device, radius_m=RADIUS_M, window_deg=self.window_deg,
            cone_half_angle_deg=self.window_deg/2, window_shape='spherical_cap',
            action_names=[item['name'] for item in self.directions], directions=list(self.directions),
            point_cloud_generated=False, full_point_cloud_generated=False, full_sphere_allocated=False,
            backprojected_pixels=0, voxel_downsampling=False, semantic_vote_unit='sampled_pixel', in_memory=output is None)
        self.last_report = report
        try:
            observations, metadata = self._load(source, frame, camera_layout)
            settings = self.config['cloud']
            rotation = body_to_action_rotation(metadata['vehicle_pose_world_ned'], self.gravity_aligned)
            input_done = time.perf_counter()
            report['stage'] = 'pixel_geometry'
            samples = {obs.name: action_depth_samples(obs.depth_perspective_m, obs.metadata,
                metadata['vehicle_pose_world_ned'], stride=settings['stride'], min_depth=settings['min_depth'],
                max_depth=settings['max_depth'], window_deg=self.window_deg, action_rotation=rotation,
                directions_frd=self.direction_frd)
                for obs in observations}
            self.last_samples = samples
            geometry_done = time.perf_counter()
            active = [name for name, value in samples.items() if value['selected_depth_pixels']]
            report['stage'] = 'segmentation'
            if active and self._model is None:
                with model_load_guard(self.device, self.init_lock, self.min_free_gib, self.memory_wait):
                    self._model = GroundedMobileSAM(self.config, self.classes, self.device)
            segmentation = {obs.name: self._model.segment(obs.rgb, self.max_detections)
                            for obs in observations if obs.name in active}
            segmented = time.perf_counter()
            report['stage'] = 'action_pooling'
            distances, labels, memberships = [], [], []
            for name in active:
                sample = samples[name]
                distances.append(sample['distance_m'])
                labels.append(segmentation[name][1][sample['v'], sample['u']])
                memberships.append(sample['membership'])
            features = pool_action_samples(
                np.concatenate(distances) if distances else np.empty(0),
                np.concatenate(labels) if labels else np.empty(0, np.uint16),
                np.concatenate(memberships) if memberships else np.empty((0, len(self.directions)), bool),
                len(self.classes)+1, self.kde_bandwidth)
            self.last_features = features
            rows = format_action_rows(features, ['unknown']+self.classes, self.directions)
            pooled = time.perf_counter()
            report.update(status='completed', stage='completed', cameras=list(self.cameras),
                frame=metadata.get('frame'), world_pose_known=metadata.get('world_pose_known'),
                vehicle_pose_reference=metadata.get('vehicle_pose_reference', 'capture_metadata'),
                depth_quality=metadata.get('depth_quality', 'unspecified'),
                action_frame='heading_FRD' if self.gravity_aligned else 'body_FRD',
                body_to_action_rotation=rotation.tolist(), inference_cameras=active,
                skipped_inference_cameras=[name for name in self.cameras if name not in active],
                action_sample_counts=features['sample_count'].tolist(),
                valid_depth_pixels=sum(s['valid_depth_pixels'] for s in samples.values()),
                selected_depth_pixels=sum(s['selected_depth_pixels'] for s in samples.values()),
                observed_actions=sum(row['depth_m'] is not None for row in rows),
                known_semantic_actions=sum(row['semantic_id'] > 0 for row in rows),
                timings_seconds=dict(input=input_done-started, pixel_geometry=geometry_done-input_done,
                    segmentation=segmented-geometry_done, action_pooling=pooled-segmented), elapsed_seconds=pooled-started)
            if ram_tags:
                report['stage'] = 'ram_tagging'
                rgb = {obs.name: obs.rgb for obs in observations}
                tags = self.tag_rgb_views(rgb['FrontCamera'], rgb['DownCamera'], verbose=True)
                report['timings_seconds']['ram_tagging'] = time.perf_counter()-pooled
                report.update(stage='completed', elapsed_seconds=time.perf_counter()-started)
            if output is not None:
                self._save(output, observations, metadata, segmentation, rows)
                if ram_tags:
                    write_json(output/'ram_tags.json', tags)
            return {'actions': rows, 'ram_tags': tags} if ram_tags else rows
        except Exception as error:
            report.update(status='failed', error=str(error), elapsed_seconds=time.perf_counter()-started)
            if output is not None:
                write_json(output/'pipeline.json', report)
            raise

    def _save(self, output, observations, metadata, segmentation, rows):
        """Opt-in diagnostics: captures, masks and direction features; no cloud."""
        capture, masks = output/'capture', output/'masks'
        save_capture(observations, capture, metadata)
        colors = palette(len(self.classes)+1)
        for obs in observations:
            values = segmentation.get(obs.name)
            if values is None:
                shape = obs.rgb.shape[:2]
                values = (obs.rgb, np.zeros(shape, np.uint16), np.zeros(shape, np.uint16), np.zeros(shape, np.float32), [])
            save_segmentation(masks/obs.name, *values, colors)
        write_json(masks/'meta.json', dict(classes=['unknown']+self.classes, colors=colors.tolist(),
            capture=str(capture), capture_meta_sha256=hashlib.sha256((capture/'meta.json').read_bytes()).hexdigest(),
            gsa_commit=json.loads((ROOT/'config/sources.json').read_text())['gsa']['commit'],
            detections={name: len(value[-1]) for name, value in segmentation.items()}))
        np.savez_compressed(output/'action_features.npz', **self.last_features,
                            action_names=np.asarray(self.last_report['action_names']),
                            direction_frd=self.direction_frd,
                            yaw_deg=np.asarray([item['yaw_deg'] for item in self.directions]),
                            elevation_deg=np.asarray([item['elevation_deg'] for item in self.directions]),
                            feature_schema=np.array(self.last_report['feature_schema']))
        write_json(output/'action_values.json', rows)
        with (output/'action_values.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        write_json(output/'pipeline.json', self.last_report)
