# OpenUAV RGB-D → 方向语义与深度

统一入口为 [run_pipeline.py](../scripts/run_pipeline.py)，Python 接口为 [GSAActionPipeline](../src/pipeline.py)。从 GSA 的逐像素语义、深度和标定直接计算指定方向特征，**不构建 XYZ 点云、体素或完整球面**。点云导出使用 README 中的独立三阶段流程。

## 输入与标定

通过 `--cameras` 选择 `FrontCamera`、`LeftCamera`、`RightCamera`、`RearCamera`、`DownCamera` 的任意非空组合。默认前/下视图；只读取被选中的目录：

```text
sample/
├── frontcamera/000035.png
├── frontcamera_depth/000035.png
├── downcamera/000035.png
├── downcamera_depth/000035.png
└── log/000035.json             # 可选
```

其他视角使用 `leftcamera`、`rightcamera`、`rearcamera` 及对应 `_depth` 文件夹。仅有一个完整帧时可省略 `--frame`；否则指定编号。

深度 PNG 为历史 TravelUAV uint8 格式，反解为 `pixel * 100 / 255` 米；0、255 丢弃，量化步长约 0.392 米。RGB 与深度必须配准、尺寸一致。也支持单帧采集目录 `meta.json` + `<Camera>/rgb.png` + `<Camera>/depth_perspective_m.npy`，此时不传 `--frame`。

默认 FOV 为 90°，外参见 [openuav_cameras.json](../config/openuav_cameras.json)。其他标定用 `--camera-layout` 和 `dataset.fov_deg` 配置覆盖；米制采集目录使用自身元数据。

可选日志提供 `sensors.state.position` 及 xyzw 四元数 `sensors.state.orientation`。无日志时假设机体水平，使用虚拟机体原点并标记 `world_pose_known=false`。默认方向坐标系按重力对齐，前方跟随航向；`--no-gravity-aligned` 改为机体轴。实时采集须用 `--pause` 同步位姿。

## 命令行与输出

先按 [README](../README.md#installation) 安装环境和模型。在项目根目录执行：

```bash
conda activate rgbd-sem3d
export SEM3D_CONFIG="$PWD/config/resources.example.json"
source env/activate.sh
export CUDA_VISIBLE_DEVICES=0

python scripts/run_pipeline.py \
  --input /path/to/TravelUAV/Carla_Town02/01fa26b7-98aa-4284-b75d-d1fad1f81ee2 \
  --frame 000035 --no-ram-tags
```

不启用 RAM 时，stdout 为动作列表，以下仅示意字段：

```json
[
  {"action":"front","depth_m":8.0,"semantic_id":2,"semantic_name":"road"},
  {"action":"L30","depth_m":7.0,"semantic_id":10,"semantic_name":"fence"},
  {"action":"R30","depth_m":9.0,"semantic_id":1,"semantic_name":"building"},
  {"action":"DOWN","depth_m":null,"semantic_id":-1,"semantic_name":"no_return"}
]
```

默认方向及顺序为 `front, L30, R30, DOWN`。包含前/下视图时，CLI 默认启用 [RAM](RAM_TAGS.md)，返回 `{"actions":[...],"ram_tags":{"front":{...},"down":{...}}}`。其他视图组合默认跳过 RAM；显式 `--ram-tags` 要求同时选择前和下。Python `.run()` 默认不启用 RAM。

| 参数 | 含义 |
| --- | --- |
| `--cameras LeftCamera RearCamera` | 仅读取这两个视角，不要求其他视角文件存在 |
| `--directions config/directions.example.json` | 自定义方向 JSON；覆盖资源配置中的 `actions.directions` |
| `--window-deg 20` | 圆锥全张角，默认 20°，半角 10° |
| `--kde-bandwidth 0.4` | 径向 KDE 核标准差，米 |
| `--no-ram-tags` | 跳过 RAM，只返回动作列表 |
| `--output outputs/run_0001` | 保存 capture/masks、action_features.npz、action_values.json/csv、pipeline.json；仍不生成点云 |
| `--verbose` | 启用模型进度和错误堆栈，默认静默 |
| `--stream` | JSONL 常驻进程，复用已加载模型 |

方向文件是非空列表，例如只查询后方和向下：

```json
[
  {"name":"rear","yaw_deg":180,"elevation_deg":0},
  {"name":"down","yaw_deg":0,"elevation_deg":-90}
]
```

yaw 向右为正，范围 `[-180,180]`；elevation 向上为正，范围 `[-90,90]`。名称必须非空且唯一，输出顺序与列表一致。所有方向共用全张角与固定 15 米半径。

选中的图像不一定能看到指定方向。缺少回波输出 `null/-1/no_return`；有观测但未识别输出语义 `0/unknown`。无回波不等于 15 米内自由空间。机体近水平、标准安装时，前图通常能覆盖前/左30°/右30°，下图覆盖竖直下方；遮挡、相机偏移、姿态和无效深度会影响实际覆盖。

`--stream` 先输出 ready，再逐行接收 `{"id":1,"input":"/path/to/sample","frame":"000035"}`，返回 `rows`、运行报告及可选 RAM 标签。默认不写中间文件；输出目录须为空或不存在。

## 直接计算方法

设相机射线为 r、深度为 d、相机相对无人机偏移为 t、旋转为 R，数学上的位置为 `p=t+d*R*r`。实际只计算标量：

- `rho² = ||t||² + 2*d*t·(R*r) + d²*||r||²`。
- 每个方向 a 的投影 `p·a = t·a + d*(R*r)·a`。
- 使用 `rho<=15` 和 `p·a>=rho*cos(window_deg/2)` 选择像素，不创建场景 XYZ 数组。

标定射线缓存复用；没有入选像素的视图跳过 GSA。相机偏移、姿态和 range-depth/z-depth 均参与计算。

每个圆锥直接池化，无细分网格：深度取全部入选像素径向距离的 Gaussian KDE 密度峰；语义每像素一票，包含 unknown，并列取较小 ID。stride、相机深度上下限沿用 `cloud.stride/min_depth/max_depth`，默认 2、0.5、80 米。15 米限制按无人机中心另算；`cloud.voxel_size` 和渲染设置不参与这个接口。

## Python 接口

```python
from src.pipeline import GSAActionPipeline

pipeline = GSAActionPipeline(
    device="cuda:0",
    cameras=["RearCamera"],
    directions=[{"name": "rear", "yaw_deg": 180, "elevation_deg": 0}],
)
rows = pipeline.run("/path/to/sample", frame="000035")
# 后续样本继续使用同一对象，避免重新加载模型。
```

`last_features` 提供方向计数和投票，`last_samples` 仅含像素索引、标量距离及方向归属；`last_report` 含角度、坐标系、计时与质量信息。`last_cloud` 固定为 None。诊断 NPZ 保存方向名称、轴向量和角度。

前/下两张 RGB 的标签可以单独调用 `RAMViewTagger.tag(front_rgb, down_rgb)`，不需要深度。RAM 使用完整 RGB，与方向语义是不同输出，不替换 GSA 类别。

测试覆盖全部 31 种非空视角组合、诊断采集往返、自定义方向、相机偏移、姿态、深度语义与 15 米边界。独立 XYZ 反投影只用于验证标量方法，另有测试禁止正常 wrapper 调用点云融合或体素函数。单元测试使用固定语义图代替模型，真实模型需额外进行 GPU smoke run。
