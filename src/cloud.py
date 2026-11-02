"""Point-cloud export and portable rendering without a display server/Open3D."""
import json
import os

import numpy as np

from src.config import ROOT, write_json
from src.geometry import backproject, voxel_representatives


def depth_statistics(depth):
    finite = depth[np.isfinite(depth)]
    positive = finite[finite > 0]
    return {"shape": list(depth.shape), "finite_pixels": len(finite), "positive_pixels": len(positive),
            "min_m": float(finite.min()) if len(finite) else None,
            "max_m": float(finite.max()) if len(finite) else None,
            "std_m": float(finite.std(dtype=np.float64)) if len(finite) else None,
            "suspect": not len(positive) or (len(finite) > 0 and float(np.ptp(finite)) < 1e-6)}


def fuse_action_observations(observations, segmentation, body_pose, settings, pixel_masks,
                             action_rotation, window_deg):
    """In-memory fusion: create scene XYZ only for the selected action pixels."""
    from src.actions import action_membership
    spec = {'xyz': (np.float32, (3,)), 'rgb': (np.uint8, (3,)), 'view_id': (np.uint8, ()),
            'semantic_id': (np.uint16, ()), 'instance_id': (np.uint16, ()), 'confidence': (np.float32, ())}
    chunks = {key: [] for key in spec}
    counts = {}
    for view_id, observation in enumerate(observations):
        name = observation.name
        counts[name] = 0
        if not pixel_masks[name].any():
            continue
        xyz, v, u = backproject(observation.depth_perspective_m, observation.metadata, body_pose,
                                settings['stride'], settings['min_depth'], settings['max_depth'], pixel_masks[name])
        keep = action_membership(xyz @ action_rotation, window_deg).any(axis=1)
        xyz, v, u = xyz[keep], v[keep], u[keep]
        semantic, instance, confidence = segmentation[name][1:4]
        values = dict(xyz=xyz, rgb=observation.rgb[v, u], semantic_id=semantic[v, u],
                      instance_id=instance[v, u], confidence=confidence[v, u],
                      view_id=np.full(len(xyz), view_id, np.uint8))
        counts[name] = len(xyz)
        for key, value in values.items():
            chunks[key].append(value)
    data = {key: np.concatenate(parts) if parts else np.empty((0,)+spec[key][1], spec[key][0])
            for key, parts in chunks.items()}
    chosen = voxel_representatives(data['xyz'], data['semantic_id'], data['confidence'], settings['voxel_size'])
    data = {key: value[chosen] for key, value in data.items()}
    data['action_membership'] = action_membership(data['xyz'] @ action_rotation, window_deg)
    return data, counts


def fuse(capture, masks, settings, labeled_only=False, *, pixel_masks=None,
         action_rotation=None, action_window_deg=None):
    from PIL import Image
    meta = json.loads((capture / "meta.json").read_text())
    meta["source_capture"] = str(capture.resolve())
    seg = json.loads((masks / "meta.json").read_text()) if masks is not None else None
    if labeled_only and seg is None:
        raise ValueError("--labeled-only requires semantic masks")
    if "vehicle_pose_world_ned" not in meta:
        raise ValueError("Capture lacks vehicle pose. Re-capture with the updated step 1 script.")
    keys = ["xyz", "rgb", "view_id"] + (["semantic_id", "instance_id", "confidence"] if seg is not None else [])
    chunks = {k: [] for k in keys}
    views, view_counts = list(meta["cameras"]), {}
    statistics = {}
    if not views:
        raise ValueError("Capture has no cameras")
    for view_id, camera_name in enumerate(views):
        camera = meta["cameras"][camera_name]
        depth = np.load(capture / camera_name / "depth_perspective_m.npy")
        rgb = np.array(Image.open(capture / camera_name / "rgb.png").convert("RGB"))
        arrays = [np.load(masks / camera_name / f"{name}.npy")
                  for name in ("semantic", "instances", "confidence")] if seg is not None else []
        if rgb.shape[:2] != depth.shape or any(a.shape != depth.shape for a in arrays):
            raise ValueError(f"Mask/RGB/depth resolution mismatch: {camera_name}")
        if seg is not None and (np.any(arrays[0] < 0) or np.any(arrays[0] >= len(seg["classes"]))):
            raise ValueError(f"Invalid semantic IDs: {camera_name}")
        xyz, v, u = backproject(depth, camera, meta["vehicle_pose_world_ned"], settings["stride"],
                                settings["min_depth"], settings["max_depth"],
                                pixel_mask=pixel_masks[camera_name] if pixel_masks is not None else None)
        if action_window_deg is not None:
            from src.actions import action_membership
            keep = action_membership(xyz @ action_rotation, action_window_deg).any(axis=1)
            xyz, v, u = xyz[keep], v[keep], u[keep]
        statistics[camera_name] = depth_statistics(depth)
        values = {"xyz": xyz, "rgb": rgb[v, u], "view_id": np.full(len(xyz), view_id, dtype=np.uint8)}
        if seg is not None:
            values.update(zip(("semantic_id", "instance_id", "confidence"), (a[v, u] for a in arrays)))
        keep = values["semantic_id"] != 0 if labeled_only else np.ones(len(xyz), dtype=bool)
        view_counts[camera_name] = int(keep.sum())
        for key, value in values.items():
            chunks[key].append(value[keep])
    data = {key: np.concatenate(value) for key, value in chunks.items()}
    selected = voxel_representatives(data["xyz"], data.get("semantic_id", np.zeros(len(data["xyz"]), np.uint16)),
                                     data.get("confidence", np.zeros(len(data["xyz"]), np.float32)), settings["voxel_size"])
    meta["cloud_diagnostics"] = {"depth_statistics": statistics,
                                 "depth_suspect": all(s["suspect"] for s in statistics.values()),
                                 "unsynchronized_live_pose": meta.get("source") == "airsim_rpc" and not meta.get("pose_synchronized_by_pause")}
    data = {key: value[selected] for key, value in data.items()}
    if action_window_deg is not None:
        from src.actions import ACTION_DISPLAY_ORDER, RADIUS_M, action_membership
        data['action_membership'] = action_membership(data['xyz'] @ action_rotation, action_window_deg)
        meta['action_scope'] = dict(kind='four_action_cones', full_point_cloud_generated=False,
                                   radius_m=RADIUS_M, window_deg=action_window_deg,
                                   window_shape='spherical_cap', action_names=[name for name, _ in ACTION_DISPLAY_ORDER],
                                   points_per_action=data['action_membership'].sum(axis=0).tolist(),
                                   points_before_voxel=sum(view_counts.values()),
                                   body_to_action_rotation=np.asarray(action_rotation).tolist())
    return data, meta, seg, views, view_counts


def write_ply(path, data, colors):
    dtype = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"),
             ("blue", "u1")]
    attributes = [(key, typ) for key, typ in (("semantic_id", "<u2"), ("instance_id", "<u2"),
                  ("confidence", "<f4"), ("view_id", "u1")) if key in data]
    dtype += attributes
    vertices = np.empty(len(data["xyz"]), dtype=dtype)
    for i, key in enumerate(("x", "y", "z")):
        vertices[key] = data["xyz"][:, i]
    for i, key in enumerate(("red", "green", "blue")):
        vertices[key] = colors[:, i]
    for key, _ in attributes:
        vertices[key] = data[key]
    header = "ply\nformat binary_little_endian 1.0\ncomment frame drone_body_FRD meters\n"
    header += f"element vertex {len(vertices)}\n"
    for name, typ in dtype:
        ply_type = {"<f4": "float", "u1": "uchar", "<u2": "ushort"}[typ]
        header += f"property {ply_type} {name}\n"
    with path.open("wb") as stream:
        stream.write((header + "end_header\n").encode("ascii"))
        vertices.tofile(stream)


def render(output, data, seg=None, max_points=60000, diagnostics=None):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    if max_points < 1:
        raise ValueError("render_max_points must be positive")
    rng = np.random.default_rng(0)
    indices = rng.choice(len(data["xyz"]), min(len(data["xyz"]), max_points), replace=False)
    xyz = data["xyz"][indices]
    title = "Drone-centered semantic cloud" if seg is not None else "Drone-centered RGB depth cloud"
    diagnostics = diagnostics or {}
    if diagnostics.get("depth_suspect"):
        title += " | SUSPECT DEPTH"
    if diagnostics.get("unsynchronized_live_pose"):
        title += " | UNSYNCHRONIZED POSE"
    if not len(xyz):
        title += " | EMPTY CLOUD"
    fig = go.Figure()
    if seg is not None:
        labels = data["semantic_id"][indices]
        colors = np.asarray(seg["colors"], dtype=np.uint8)
        point_colors = colors[labels]
        for label in np.unique(labels):
            points = xyz[labels == label]
            color = "rgb(%d,%d,%d)" % tuple(colors[label])
            fig.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="markers",
                                      marker={"size": 2, "color": color}, name=seg["classes"][label],
                                      hovertemplate="x=%{x:.2f} y=%{y:.2f} down=%{z:.2f}m<extra>%{fullData.name}</extra>"))
    else:
        point_colors = data["rgb"][indices]
        fig.add_trace(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers",
                                  marker={"size": 2, "color": ["rgb(%d,%d,%d)" % tuple(c) for c in point_colors]},
                                  name="RGB depth", hovertemplate="x=%{x:.2f} y=%{y:.2f} down=%{z:.2f}m<extra></extra>"))
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode="markers+text", text=["drone"],
                              marker={"color": "red", "size": 7, "symbol": "diamond"}, name="drone origin"))
    fig.update_layout(title=title + " (meters)", template="plotly_white",
                      scene={"xaxis_title": "X Forward (m)", "yaxis_title": "Y Right (m)",
                             "zaxis_title": "Z Down (m)", "zaxis": {"autorange": "reversed"},
                             "aspectmode": "data"})
    fig.write_html(output / "cloud.html", include_plotlyjs=True, auto_open=False)
    figure = plt.figure(figsize=(11, 8))
    ax = figure.add_subplot(projection="3d")
    ax.set_position([.02, .08, .76, .82])
    ax.scatter(*xyz.T, c=point_colors / 255., s=.5, rasterized=True)
    ax.scatter([0], [0], [0], c="red", s=30, marker="x")
    ax.set(xlabel="X Forward (m)", ylabel="Y Right (m)", zlabel="Z Down (m)",
           title=title)
    ax.invert_zaxis()
    if seg is not None:
        from matplotlib.lines import Line2D
        handles = [Line2D([0], [0], marker="o", linestyle="", color=colors[label] / 255.,
                          markersize=5, label=seg["classes"][label]) for label in np.unique(labels)]
        figure.legend(handles=handles, loc="center right", fontsize=8, frameon=False)
    if len(xyz):
        span = np.ptp(np.concatenate([xyz, np.zeros((1, 3))]), axis=0)
        ax.set_box_aspect(np.maximum(span, 1))
    figure.savefig(output / "cloud.png", dpi=160, bbox_inches="tight", pad_inches=.4)
    plt.close(figure)


def export(output, data, capture_meta, seg, views, view_counts, settings):
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "cloud.npz", **data)
    if seg is not None:
        colors = np.asarray(seg["colors"], dtype=np.uint8)[data["semantic_id"]]
        write_ply(output / "semantic.ply", data, colors)
    write_ply(output / "rgb.ply", data, data["rgb"])
    diagnostics = capture_meta.get("cloud_diagnostics", {})
    report = {
        "frame": "drone_body_FRD", "axes": {"x": "forward", "y": "right", "z": "down"},
        "units": "meters", "points": len(data["xyz"]),
        "kind": "semantic" if seg is not None else "rgb_depth", "views": views, "view_counts_before_voxel": view_counts,
        "status": "empty" if not len(data["xyz"]) else ("depth_suspect" if diagnostics.get("depth_suspect") else "generated"),
        **diagnostics,
        "settings": settings, "depth_quality": capture_meta.get("depth_quality", "unspecified"),
        "source": capture_meta.get("source"), "source_capture": capture_meta.get("source_capture"),
        "pose_synchronized_by_pause": capture_meta.get("pose_synchronized_by_pause", False),
        "vehicle_pose_world_ned": capture_meta["vehicle_pose_world_ned"],
        "fusion": "highest-confidence representative per semantic class/voxel" if seg is not None else "first sampled point per voxel",
    }
    if seg is not None:
        report.update(classes=seg["classes"], colors=seg["colors"], source_capture=seg.get("capture"),
                      gsa_commit=seg.get("gsa_commit"), instance_id_scope="per view; use (view_id, instance_id) as key")
    if 'action_scope' in capture_meta:
        report['action_scope'] = capture_meta['action_scope']
    write_json(output / "meta.json", report)
    return report


def save_depth_cloud(capture, settings=None, render_cloud=True):
    """Standard acquisition output: RGB depth cloud, without segmentation models."""
    settings = dict(settings if settings is not None else json.loads((ROOT / "config/resources.example.json").read_text())["cloud"])
    output = capture / "depth_cloud"
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Depth cloud output is not empty: {output}")
    data, meta, _, views, counts = fuse(capture, None, settings)
    report = export(output, data, meta, None, views, counts, settings)
    if render_cloud:
        render(output, data, max_points=settings["render_max_points"], diagnostics=meta["cloud_diagnostics"])
    print(f"Depth cloud: {len(data['xyz']):,} points; status={report['status']}; {output}", flush=True)
    return {"directory": str(output), "points": report["points"], "status": report["status"],
            "depth_suspect": report["depth_suspect"], "unsynchronized_live_pose": report["unsynchronized_live_pose"]}
