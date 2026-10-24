# Adapting Grounded-Segment-Anything into Generating 3D Semantic-Depth Point Cloud Pipeline 

Research code for exploring Grounded-Segment-Anything-assisted 3D semantic point-cloud generation in OpenUAV environments.

## Milestone 1: OpenUAV five-view RGB-D capture

This branch implements the data-acquisition boundary for the project and deliberately does **not** include GSA or 3D reconstruction yet.

The implementation follows the camera layout and image requests used by the OpenUAV/TravelUAV codebase:

- Cameras: `FrontCamera`, `LeftCamera`, `RightCamera`, `RearCamera`, `DownCamera`
- RGB: `airsim.ImageType.Scene`, uncompressed uint8
- Depth: `airsim.ImageType.DepthPerspective`, floating-point meters
- OpenUAV camera settings: 256×256, 90° FOV for both Scene and DepthPerspective
- Camera pose: queried from `simGetCameraInfo`; AirSim reports it in the world NED frame

Reference implementation:

- TravelUAV `airsim_plugin/AirVLNSimulatorClientTool.py`
- TravelUAV `airsim_plugin/AirVLNSimulatorServerTool.py`
- AirSim Image APIs: https://microsoft.github.io/AirSim/image_apis/

### Why metric depth is preserved

TravelUAV's training-side helper converts `DepthPerspective` to an 8-bit visualization by clipping to 0–100 m. That is appropriate for its model input but loses metric precision. This project therefore saves the original floating-point depth as `depth_perspective_m.npy`; later PRs will use it for 3D back-projection.

## Environment

The recommended Conda environment is named **`rgbd-sem3d`**, reflecting the project's RGB-D semantic 3D point-cloud pipeline while keeping the name short.

The current environment baseline is:

- Python `3.10.22`
- CUDA `11.8` / PyTorch `cu118`
- PyTorch `2.0.1+cu118`
- torchvision `0.15.2+cu118`
- torchaudio `2.0.2+cu118`

Create and activate the environment:

```bash
conda create -n rgbd-sem3d python=3.10.22 -y
conda activate rgbd-sem3d
```

Install the CUDA 11.8 PyTorch stack explicitly:

```bash
pip install \
  torch==2.0.1+cu118 \
  torchvision==0.15.2+cu118 \
  torchaudio==2.0.2+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
```

Then install the remaining pinned project dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt` also points pip to the official PyTorch CUDA 11.8 wheel index and pins the same `+cu118` PyTorch versions. The dependency baseline is derived from the reference environment used for this project rather than from an unconstrained latest-version install.

### Capture one frame

With an OpenUAV-compatible AirSim scene already running:

```bash
python scripts/capture_openuav.py \
  --ip 127.0.0.1 \
  --port 41451 \
  --output outputs/frame_0001
```

If OpenUAV launches the scene on a different RPC port, pass that port explicitly.

### Output

```text
outputs/frame_0001/
├── FrontCamera/
│   ├── rgb.png
│   ├── depth_perspective_m.npy
│   └── camera.json
├── LeftCamera/
├── RightCamera/
├── RearCamera/
├── DownCamera/
└── meta.json
```

Each `camera.json` stores image size, FOV, a pinhole intrinsic matrix derived from the horizontal FOV, `simGetCameraInfo` pose, image-response poses/timestamps, and AirSim's projection matrix.

## Project roadmap

1. OpenUAV/AirSim five-view RGB-D capture.
2. Grounded-Segment-Anything semantic-mask frontend.
3. Metric RGB-D semantic back-projection and five-view fusion.
4. Open3D point-cloud visualization and demo pipeline.
