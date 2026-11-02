# 两个场景的深度全量检查与 AirSim 补采

## 检查结论

2026-10-25 对 本机 TravelUAV 数据副本 中两个场景的全部已有轨迹进行了检查。**深度文件存在且与 RGB 帧对应，但没有找到能用于正常场景重建的深度。** 需要尝试通过 AirSim 重新采集，并检查服务端实际返回的深度。

| 场景 | 轨迹数 | 五视角深度 PNG 总数 | 整张恒为 2 的 PNG | 所有 PNG 像素取值 | 深度张量份数 | 张量最小 / 最大值（米） |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| BrushifyCountryRoads | 320 | 101,755 | 101,364 | 0、1、2 | 320 | 0 / 0.78515625 |
| NordicHarbour | 508 | 106,865 | 105,424 | 0、1、2 | 508 | 0 / 0.78515625 |

逐一解码了全部 **208,620 张 PNG**，并读取全部 **828 份 `depth_imgs.tensor`**。五个标准视角均无缺失深度目录、无 RGB 对应深度帧缺失、无额外深度帧；PNG 解码和张量读取均无错误，张量均为有限值。在轨迹顶层也没有发现其他深度文件。

PNG 的 uint8=2 按数据集的 0–100 m 编码仅对应约 0.784 m，其他少量像素为 0 或 1。`depth_imgs.tensor` 同样未提供正常场景深度范围，不能靠切换到张量恢复几何。当前检查只确认已有数据异常，尚不能确定异常发生在渲染、RPC 返回还是历史保存过程。

完整本机结果在 `outputs/depth_audit_20261025/`：`summary.json` 为场景汇总，`trajectories.csv` 为逐轨迹汇总，`trajectories.jsonl` 包含每个视角的像素直方图、帧配对和张量统计。输出不纳入 Git。

重新检查时，激活 `rgbd-sem3d`，从项目根目录运行并使用新输出目录：

```bash
python scripts/audit_depth.py --config config/resources.json \
  --scenes BrushifyCountryRoads NordicHarbour \
  --check-tensors --workers 4 \
  --output outputs/depth_audit_repeat_0001
```

`--dataset-root` 可覆盖数据位置。退出码 0 表示检查成功执行；深度质量需查看报告。

## 手动启动场景

[scripts/recapture_depth.py](../scripts/recapture_depth.py) 连接已运行的 AirSim，不负责启动或关闭场景。应使用与轨迹采集时相同版本的地图、无人机出生点/NED 原点和五视角布局；脚本无法自动验证地图版本或原点。

本机可在单独终端启动一个场景。`SIM_ROOT` 对应配置中的 `paths.simulator_root`；`GPU` 是 Unreal 使用的显卡编号，按当前可用资源调整。配置模板包含 `Drone_1`、端口 41451、256×256 / 90° 的五视角 Scene 和 DepthPerspective。

```bash
cd /path/to/GSA_for_AirSim_3DSemanticDepthPointCloud
SIM_ROOT=/path/to/traveluav_env
GPU=4

"$SIM_ROOT/BrushifyCountryRoads/BrushifyCountryRoads/Binaries/Linux/BrushifyCountryRoads" \
  BrushifyCountryRoads -RenderOffscreen -NoSound -NoVSync \
  -GraphicsAdapter="$GPU" -settings="$PWD/config/airsim_settings.example.json"
```

保持进程运行，另开终端补采。完成后在启动终端用 Ctrl+C 结束该场景，再启动 NordicHarbour；两条启动命令使用相同 RPC 端口，应分开运行：

```bash
"$SIM_ROOT/NordicHarbour/NordicHarbour/Binaries/Linux/NordicHarbour" \
  NordicHarbour -RenderOffscreen -NoSound -NoVSync \
  -GraphicsAdapter="$GPU" -settings="$PWD/config/airsim_settings.example.json"
```

也可以沿用 TravelUAV 启动器，用 `--ip`、`--port`、`--vehicle-name` 指向实际服务。这里直接调用已有可执行文件，不修改公共仿真程序。NordicHarbour 已连接到按此命令启动的真实服务完成补采，但深度仍异常，见下方现场记录；BrushifyCountryRoads 尚未现场补采。

## 在已有轨迹位姿补采

在补采终端准备环境：

```bash
conda activate rgbd-sem3d
cd /path/to/GSA_for_AirSim_3DSemanticDepthPointCloud
export SEM3D_CONFIG="$PWD/config/resources.json"
source env/activate.sh
```

先查看计划；`--dry-run` 只读取日志，不连接仿真、不移动无人机、不写采集文件：

```bash
python scripts/recapture_depth.py \
  --trajectory BrushifyCountryRoads/0008c004-9c02-40d3-928f-b7228c17a39d \
  --frames 000210 --dry-run
```

确认 BrushifyCountryRoads 的 RPC 服务就绪后，补采此前异常的同一个样本点：

```bash
python scripts/recapture_depth.py \
  --trajectory BrushifyCountryRoads/0008c004-9c02-40d3-928f-b7228c17a39d \
  --frames 000210 \
  --ip 127.0.0.1 --port 41451 --vehicle-name Drone_1 \
  --check-planar --output outputs/recapture_country_0001
```

切换到 NordicHarbour 后：

```bash
python scripts/recapture_depth.py \
  --trajectory NordicHarbour/00ac351b-13de-4e3b-a453-10fd3948719b \
  --frames 000115 \
  --ip 127.0.0.1 --port 41451 --vehicle-name Drone_1 \
  --check-planar --output outputs/recapture_nordic_0001
```

`--trajectory` 支持相对于 `paths.dataset_root` 的路径或绝对路径。`--frames 000005 000010` 可选择多个记录帧；用 `--all-frames` 替代 `--frames ...` 可回放该轨迹所有已保存前视角 RGB 对应的位姿。`--all-frames --every 5` 表示每 5 个已保存帧选择 1 个，不是把帧编号每次加 5。输出目录通常必须为空或不存在。如果失败目录只有 `recapture.json` 且没有保存任何帧，可直接重试，旧报告会保留在 `previous_attempts` 中；已有图像的目录不会被覆盖。

脚本读取 `log/<frame>.json` 中的 NED 位置与 xyzw 四元数，先在暂停状态初始化渲染目标，再用 `simSetKinematics` 设置位姿并清零速度。每次推进预热帧前都重新定位，避免物理运动误差在预热期间累积。默认预热 2 帧，可通过 `--warmup-frames` 增加。最终位置误差超过 0.25 m 或姿态误差超过 2° 时，最多尝试 3 轮定位，仍未达到要求则停止；报告记录请求位置、实际位置和误差，不放宽原阈值。五视角 RGB 与 DepthPerspective 在同一个 `simGetImages` 请求中获取，仿真保持暂停，并保存实际相机内外参、时间戳和无人机位姿。

结束、RPC 报错或 Ctrl+C 时，会尝试恢复采集前的无人机运动状态和暂停状态；这需要 RPC 连接仍可用。脚本不接管飞控、不解锁起飞、不写原始轨迹目录。使用与主采集入口相同的本地 RPC 适配，不修改 AirSim 安装包。

## 输出与诊断

```text
outputs/recapture_country_0001/
├── recapture.json
└── 000210/
    ├── meta.json
    ├── source_log.json
    ├── depth_cloud/
    │   ├── rgb.ply / cloud.npz
    │   ├── cloud.html / cloud.png
    │   └── meta.json
    └── <Camera>/
        ├── rgb.png
        ├── depth_perspective_m.npy
        ├── camera.json
        └── depth_planar_m.npy       # 仅 --check-planar 时生成
```

主深度直接保存 RPC 返回的 **float32 米制值**，不进行 uint8 编码。RGB 同时重新采集；后续分割应使用这次保存的 RGB。不同运行的渲染、物体状态和曝光可能变化，不能将新深度直接配到旧 RGB 上。

每帧默认输出 RGB 着色的深度点云到 `depth_cloud/`，无需 GSA 掩码，坐标为无人机本体 FRD、单位米。`cloud.npz` 包含 `xyz`、`rgb`、`view_id`；PLY 可用常见点云工具打开，HTML 支持交互查看，PNG 为静态预览。参数读取配置的 `cloud` 部分，`--no-render` 仅跳过 HTML/PNG，PLY/NPZ 仍是标准输出。`recapture.json` 每帧的 `depth_cloud` 条目记录输出位置、点数和状态。

常量深度也会导出诊断点云，元数据标记 `depth_suspect`，预览标题显示 `SUSPECT DEPTH`；过滤后没有点则输出状态为 `empty` 的零点 PLY/NPZ 和空预览。能够导出文件不代表恢复了场景几何。

已有采集数据可独立补导出，不必重新连接仿真或运行语义分割：

```bash
python scripts/fuse_cloud.py \
  --input outputs/recapture_nordic_0001/000115 \
  --output outputs/nordic_depth_cloud_0002
```

`meta.json` 和 `recapture.json` 包含每个视角的有限/正深度像素数量、最小值、最大值、标准差及 `suspect` 标记。如果某帧五个 DepthPerspective 视角均为常量或没有正深度，仍保留原始结果，汇总标记为 `depth_suspect`，退出码为 2；其他采集错误退出码为 1。退出码 0 仅表示通过基础检查，仍应检查统计和几何预览。

`--check-planar` 额外获取 DepthPlanar，帮助区分不同深度通道的问题。它表示相机平面方向的深度，DepthPerspective 表示沿像素射线的距离，两者不能直接互换。诊断图可能使用服务端默认分辨率，不参与现有融合。如果两种通道仍返回常量，需要检查场景深度渲染和 AirSim 服务端。脚本不会改变比例来制造看似正常的结果。两种深度的定义见 [AirSim 官方图像接口说明](https://microsoft.github.io/AirSim/image_apis/)。

确认新深度正常后，直接接入原 pipeline 后两步：

```bash
export CUDA_VISIBLE_DEVICES=4
python scripts/segment_gsa.py \
  --input outputs/recapture_country_0001/000210 \
  --output outputs/recapture_country_0001/masks_000210 --device cuda:0

python scripts/fuse_cloud.py \
  --input outputs/recapture_country_0001/000210 \
  --masks outputs/recapture_country_0001/masks_000210 \
  --output outputs/recapture_country_0001/cloud_000210
```

## NordicHarbour 现场补采记录

2026-10-25，已连接用户运行的 NordicHarbour，采集轨迹 `00ac351b-13de-4e3b-a453-10fd3948719b` 的 `000115` 帧。输出位于 `outputs/recapture_nordic_0001/000115/`，汇总为 `outputs/recapture_nordic_0001/recapture.json`。

- 初次运行记录到位置误差 27.078 m。逐步排查观察到：设置位姿后，物理位置立即更新，但渲染机体位置需推进一帧才更新。旧版本没有在每个预热帧前重新定位，也没有对超限结果重试；初次失败缺少实际 XYZ，无法从旧报告唯一确定 27 m 偏差的具体来源。
- 修复后同一帧位置误差 `2.384185791015625e-7 m`，姿态误差约 `3.167e-5°`，RGB 已正常保存；先前的失败报告保留在 `previous_attempts`。
- 五视角 DepthPerspective（256×256）和 DepthPlanar（256×144）均直接返回全图 `1.0 m`。标准差均为 0，汇总状态 `depth_suspect`、退出码 2。**这次已完成接口采集，但没有恢复有效几何深度。**
- 本次异常值在 RPC 返回后、保存文件之前即已存在，不是 PNG 编码或 `.npy` 保存造成的。具体服务端、渲染或 RPC 序列化原因尚未确定，不据此声称已修复场景深度。

当时 25 项回归测试通过，包括模拟慢帧位移、定位重试、失败目录重试和已有采集文件保护。两个真实样本的 `--dry-run` 和全部既有数据审计也已完成。后续再次运行应使用新输出目录，例如 `outputs/recapture_nordic_0002`。

2026-10-26 已将深度点云接入采集和补采的标准输出，32 项回归测试通过。正常深度的 Carla_Town01 离线导入默认生成 39,251 点，保存在 `outputs/depth_cloud_smoke_20261026/capture/depth_cloud/`。上述 NordicHarbour 已有采集也补导出了 808 点的诊断点云，位于 `outputs/recapture_nordic_0001/000115/depth_cloud/`，其深度仍为常量 1 m，文件及预览均标记异常。
