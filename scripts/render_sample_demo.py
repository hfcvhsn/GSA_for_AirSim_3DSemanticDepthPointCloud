#!/usr/bin/env python3
"""PRESENTATION ONLY: render front/down masks and an action-labeled RGB-D unwrap.

Read a diagnostic run produced by run_pipeline.py --output. Does not change the
runtime pipeline or create/export a point cloud. Empty angular regions stay empty
except for a small, explicitly recorded display footprint around observed pixels.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image

from src.actions import DIRECTION_FRD, RADIUS_M, kde_depth_peak
from src.config import ROOT, write_json
from src.geometry import body_to_action_rotation, calibrated_rays, check_registration, pose_arrays


CAMERAS = ('FrontCamera', 'DownCamera')
EMPTY = np.array([27, 34, 46], np.uint8)


def unwrap_pixels(capture, masks, metadata, width, height, min_depth, max_depth, bandwidth):
    """Direct scalar pixel projections to yaw/elevation; no Nx3 scene point array."""
    body = metadata['vehicle_pose_world_ned']
    p_body, r_body = pose_arrays(body)
    rotation = body_to_action_rotation(body)
    cell_chunks, range_chunks, label_chunks = [], [], []
    source_counts = {}
    for name in CAMERAS:
        camera = metadata['cameras'][name]
        check_registration(camera)
        depth = np.load(capture/name/'depth_perspective_m.npy', allow_pickle=False)
        semantic = np.load(masks/name/'semantic.npy', allow_pickle=False)
        v, u, rays = calibrated_rays(depth, camera, 1)
        values = depth[v, u].astype(float)
        selected = np.isfinite(values) & (values > 0) & (values >= min_depth) & (values < max_depth)
        v, u, r, d = v[selected], u[selected], rays[selected], values[selected]
        p_camera, r_camera = pose_arrays(camera['depth_response_pose_world_ned'])
        origin = (p_camera-p_body)@r_body@rotation
        R = r_camera.T@r_body@rotation
        # Scalar projections are only used to calculate angles and range; no cloud is retained.
        def component(axis):
            return origin[axis]+d*(r[:, 0]*R[0, axis]+r[:, 1]*R[1, axis]+r[:, 2]*R[2, axis])
        forward, right, down = component(0), component(1), component(2)
        rho = np.sqrt(forward*forward+right*right+down*down)
        valid = np.isfinite(rho) & (rho > 1e-8) & (rho <= RADIUS_M)
        yaw = (np.degrees(np.arctan2(right[valid], forward[valid]))+180)%360-180
        elevation = np.degrees(np.arctan2(-down[valid], np.hypot(forward[valid], right[valid])))
        col = np.clip(((yaw+180)/360*width).astype(int), 0, width-1)
        row = np.clip(((elevation+90)/180*height).astype(int), 0, height-1)
        cell_chunks.append(row*width+col)
        range_chunks.append(rho[valid])
        label_chunks.append(semantic[v[valid], u[valid]])
        source_counts[name] = int(valid.sum())
    cells = np.concatenate(cell_chunks)
    distances, labels = np.concatenate(range_chunks), np.concatenate(label_chunks)
    unique, first, counts = np.unique(cells, return_index=True, return_counts=True)
    order = np.argsort(cells, kind='stable')
    starts = np.r_[0, np.cumsum(counts)[:-1]].astype(int)
    depth = np.full(width*height, np.nan, np.float32)
    for cell, start, count in zip(unique, starts, counts):
        depth[cell] = kde_depth_peak(distances[order[start:start+count]], bandwidth)
    pairs, votes = np.unique(np.column_stack((cells, labels)), axis=0, return_counts=True)
    def winner(pair, vote):
        rank = np.lexsort((pair[:, 1], -vote, pair[:, 0]))
        picked = rank[np.r_[True, pair[rank, 0][1:] != pair[rank, 0][:-1]]] if len(rank) else rank
        return pair[picked, 0].astype(int), pair[picked, 1].astype(np.int32)
    pixel, category = winner(pairs, votes)
    semantic = np.full(width*height, -1, np.int32)
    semantic[pixel] = category
    # Match the established demo's known-label preference, never action labels.
    display = semantic.copy()
    known_cell, known_category = winner(pairs[pairs[:, 1] > 0], votes[pairs[:, 1] > 0])
    display[known_cell] = known_category
    shape = (height, width)
    return depth.reshape(shape), semantic.reshape(shape), display.reshape(shape), source_counts


def display_footprints(depth, semantic, radius=1.5):
    """Enlarge empty display pixels using great-circle distance, including poles.

    Work entirely on the 2D angular raster. A source row's closest longitude is
    also its nearest spherical sample. Check nearby latitude rows and compare
    spherical distances, so footprints do not shrink at the south pole.
    """
    from scipy.ndimage import distance_transform_edt
    observed = np.isfinite(depth)
    if not observed.any() or radius == 0:
        return depth.copy(), semantic.copy(), np.zeros(depth.shape, bool)
    height, width = depth.shape
    longitude = np.arange(width)
    nearest_col = np.full(depth.shape, -1, np.int32)
    for row in np.flatnonzero(observed.any(axis=1)):
        padded = np.tile(observed[row], 3)
        indices = distance_transform_edt(~padded, return_distances=False, return_indices=True)[0]
        nearest_col[row] = (indices[width:2*width]-width)%width
    latitude = np.radians(-90+(np.arange(height)+.5)*180/height)
    sin_phi, cos_phi = np.sin(latitude), np.cos(latitude)
    threshold = np.cos(np.radians(radius*180/height))
    out_depth, out_semantic, added = depth.copy(), semantic.copy(), np.zeros(depth.shape, bool)
    half = int(np.ceil(radius))+1
    for row in range(height):
        best = np.full(width, -np.inf)
        source_row, source_col = np.zeros(width, int), np.zeros(width, int)
        for other in range(max(0, row-half), min(height, row+half+1)):
            columns = nearest_col[other]
            if columns[0] < 0:
                continue
            cosine = sin_phi[row]*sin_phi[other]+cos_phi[row]*cos_phi[other]*np.cos((longitude-columns)*2*np.pi/width)
            better = cosine > best
            best[better], source_row[better], source_col[better] = cosine[better], other, columns[better]
        selected = ~observed[row] & (best >= threshold-1e-14)
        added[row] = selected
        out_depth[row, selected] = depth[source_row[selected], source_col[selected]]
        out_semantic[row, selected] = semantic[source_row[selected], source_col[selected]]
    return out_depth, out_semantic, added


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Diagnostic run with capture/, masks/, action_values.json')
    parser.add_argument('--output', type=Path, required=True, help='New/empty presentation image directory')
    parser.add_argument('--width', type=int, default=1440, help='Display-only unwrap width; default 0.25 degrees per pixel')
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--point-radius', type=float, default=1.5, help='Display-only spherical footprint radius in elevation pixels, 0 disables')
    parser.add_argument('--masks-only', action='store_true', help='Only render the 3-row/2-column mask figure; can update an existing output directory')
    args = parser.parse_args(argv)
    if args.width < 1 or args.height < 1 or not np.isfinite(args.point_radius) or not 0 <= args.point_radius <= 10:
        parser.error('Require positive dimensions and display radius in [0,10]')
    source, output = args.input.resolve(), args.output.resolve()
    if output.exists() and (not output.is_dir() or (any(output.iterdir()) and not args.masks_only)):
        parser.error('Output must be empty or absent')
    metadata = json.loads((source/'capture/meta.json').read_text())
    masks_meta = json.loads((source/'masks/meta.json').read_text())
    report = json.loads((source/'pipeline.json').read_text())
    actions = json.loads((source/'action_values.json').read_text())
    with np.load(source/'action_features.npz', allow_pickle=False) as features:
        counts = features['sample_count'].copy()
        fractions = features['semantic_mode_fraction'].copy()
    if [a['action'] for a in actions] != ['front', 'L30', 'R30', 'DOWN']:
        parser.error('Expected current four-direction direct-pixel output')
    if not report['action_frame'] == 'heading_FRD':
        parser.error('This demo uses the default gravity-aligned action frame')
    colors, classes = np.asarray(masks_meta['colors'], np.uint8), masks_meta['classes']
    scene = Path(metadata['trajectory']).parent.name
    frame = metadata.get('frame', 'single frame')
    output.mkdir(parents=True, exist_ok=args.masks_only)
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.patches import Patch, Rectangle
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.titlesize': 13,
                         'axes.labelsize': 11, 'savefig.facecolor': 'white'})
    provenance = {'Purpose': 'PRESENTATION ONLY', 'Scene': scene, 'Frame': str(frame),
                  'Cameras': ','.join(CAMERAS), 'PointCloudGenerated': 'false'}
    fig, axes = plt.subplots(3, 2, figsize=(10.8, 16.8))
    used = set()
    for col, name in enumerate(CAMERAS):
        rgb = np.array(Image.open(source/'capture'/name/'rgb.png').convert('RGB'))
        semantic = np.load(source/'masks'/name/'semantic.npy', allow_pickle=False)
        used.update(np.unique(semantic).tolist())
        overlay = rgb.copy(); active = semantic > 0
        overlay[active] = (.45*rgb[active]+.55*colors[semantic[active]]).astype(np.uint8)
        for row, (pixels, title) in enumerate(((rgb, 'RGB'), (colors[semantic], 'Semantic mask'), (overlay, 'Overlay'))):
            axes[row, col].imshow(pixels, interpolation='nearest')
            axes[row, col].set_title(f'{"Front" if col == 0 else "Down"} view | {title}', pad=10)
            axes[row, col].axis('off')
    handles = [Patch(facecolor=colors[i]/255, label=classes[i]) for i in sorted(used)]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .045), ncol=min(6, len(handles)), frameon=False)
    fig.suptitle(f'{scene} | Frame {frame}\nFront / down GSA semantic masks', fontsize=20, fontweight='bold', y=.98)
    fig.text(.5, .015, 'GroundingDINO + MobileSAM | Unmodified class IDs | Source images: 256 x 256', ha='center', color='#596477', fontsize=10)
    fig.subplots_adjust(top=.91, bottom=.105, left=.025, right=.975, hspace=.20, wspace=.06)
    fig.savefig(output/'front_down_semantic_masks.png', dpi=200, metadata=provenance); plt.close(fig)
    if args.masks_only:
        print(json.dumps(dict(output=str(output/'front_down_semantic_masks.png'), rows=['RGB', 'Semantic mask', 'Overlay'],
                              columns=['FrontCamera', 'DownCamera']), ensure_ascii=False))
        return 0

    depth, semantic, semantic_base, source_counts = unwrap_pixels(source/'capture', source/'masks', metadata,
        args.width, args.height, .5, 80., .4)
    display_depth, display_semantic, added = display_footprints(depth, semantic_base, args.point_radius)
    palette = np.vstack((EMPTY, colors))
    cmap = plt.get_cmap('viridis').copy(); cmap.set_bad(EMPTY/255)
    norm = Normalize(0, RADIUS_M)
    np.savez_compressed(output/'unwrap_data.npz', depth_m=depth, semantic_id=semantic,
        display_depth_m=display_depth, display_semantic_id=display_semantic, display_added_mask=added,
        known_preference_mask=semantic_base != semantic, purpose=np.array('presentation_only'),
        row_zero=np.array('negative_elevation'), yaw_edges_deg=np.linspace(-180, 180, args.width+1),
        elevation_edges_deg=np.linspace(-90, 90, args.height+1))
    fig, axes = plt.subplots(2, 1, figsize=(16, 15.3))
    for panel, (ax, title) in enumerate(zip(axes, ('Radial depth | KDE within 15 m', 'Semantic mask | GSA categories'))):
        value = np.ma.masked_invalid(display_depth) if panel == 0 else palette[display_semantic+1]
        image = ax.imshow(value, origin='lower', extent=(-180, 180, -90, 90), interpolation='nearest', aspect='equal',
                          **({'cmap': cmap, 'norm': norm} if panel == 0 else {}))
        ax.set(title=title, xlabel='Yaw (degrees; left negative, front 0, right positive)', ylabel='Elevation (degrees)',
               xticks=[-180, -90, -30, 0, 30, 90, 180], yticks=[-90, -80, -45, 0, 45, 90])
        ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
        ax.grid(color='white', alpha=.1, linewidth=.7)
        half = report['window_deg']/2
        for i, (action, direction, callout) in enumerate(zip(actions, DIRECTION_FRD, [(-3, 45), (-85, 24), (85, 24), (100, -55)])):
            yaw = [0, -30, 30, 0][i]; elevation = -90 if i == 3 else 0
            if i == 3:
                ax.add_patch(Rectangle((-180, -90), 360, half, facecolor='#ffcd55', alpha=.07, edgecolor='none'))
                ax.add_patch(Rectangle((-180, -90), 360, half, facecolor='none', edgecolor='#ffcd55', linewidth=1.6))
            else:
                t = np.linspace(0, 2*np.pi, 181)
                right_axis = np.array([-direction[1], direction[0], 0.])
                # Unit directions describe only the cap boundary, not scene geometry.
                boundary = np.cos(np.radians(half))*direction+np.sin(np.radians(half))*(
                    np.cos(t)[:, None]*right_axis+np.sin(t)[:, None]*np.array([0., 0., -1.]))
                ax.plot(np.degrees(np.arctan2(boundary[:, 1], boundary[:, 0])), np.degrees(np.arcsin(-boundary[:, 2])),
                        color='#ffcd55', linewidth=1.6)
            ax.plot(yaw, elevation, '+', color='white', markersize=7, markeredgewidth=1.6, clip_on=False)
            metric = f'{action["depth_m"]:.2f} m' if action['depth_m'] is not None else 'no return'
            text = f'{action["action"]} | {metric}\n{action["semantic_name"]}\nvote {fractions[i]:.0%}, n={counts[i]}'
            ax.annotate(text, (yaw, elevation), xytext=callout, ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle='round,pad=.45', facecolor='white', edgecolor='#cdd5df', alpha=.97),
                arrowprops=dict(arrowstyle='-', color='#ffcd55', linewidth=1.2))
    fig.subplots_adjust(top=.9, bottom=.145, left=.07, right=.9, hspace=.21)
    cax = fig.add_axes([.92, axes[0].get_position().y0, .015, axes[0].get_position().height])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, label='Distance from UAV (m)')
    unwrap_used = sorted(set(semantic[semantic >= 0].tolist()))
    handles = [Patch(facecolor=EMPTY/255, label='No valid return within 15 m')]+[
        Patch(facecolor=colors[i]/255, label=classes[i]) for i in unwrap_used]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .048), ncol=min(5, len(handles)), frameon=False)
    fig.suptitle(f'{scene} | Frame {frame}\nUnwrapped RGB-D with four action labels', fontsize=21, fontweight='bold', y=.98)
    fig.text(.5, .925, 'Front + down views only | Radius 15 m | Full cone angle 20 degrees', ha='center', fontsize=12, color='#465166')
    fig.text(.5, .018, 'PRESENTATION ONLY: known labels emphasized; small display footprints enlarged.\nAction labels are unchanged direct-pixel pipeline results. Unobserved regions remain empty.',
             ha='center', fontsize=10, color='#596477', linespacing=1.6)
    fig.savefig(output/'unwrapped_actions.png', dpi=200, metadata=provenance); plt.close(fig)
    validation = dict(presentation_only=True, point_cloud_generated=False, scene=scene, trajectory=metadata['trajectory'], frame=frame,
        cameras=list(CAMERAS), source=str(source), actions=actions,
        action_values_sha256=hashlib.sha256((source/'action_values.json').read_bytes()).hexdigest(),
        source_pixels_within_15m=source_counts, unwrap_width=args.width, unwrap_height=args.height,
        angle_resolution_deg=[360/args.width, 180/args.height], display_point_radius_pixels=args.point_radius,
        display_footprint_metric='great-circle distance on the angular raster',
        added_display_pixels=int(added.sum()), raw_observed_cells=int(np.isfinite(depth).sum()),
        semantic_known_preference_cells=int((semantic_base != semantic).sum()), numeric_action_labels_modified=False)
    write_json(output/'demo.json', validation)
    print(json.dumps(dict(output=str(output), images=['front_down_semantic_masks.png', 'unwrapped_actions.png'],
                          scene=scene, frame=frame, purpose='presentation_only'), ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
