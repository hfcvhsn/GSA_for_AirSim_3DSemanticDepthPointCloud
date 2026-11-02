# 前/下语义掩码与动作标签展开图

演示入口：[scripts/render_sample_demo.py](../scripts/render_sample_demo.py)。**只用于展示**，不参与正常 wrapper，不生成或保存点云。

NewYorkCity 示例为轨迹 `00016766-0a99-4dda-9f89-8d62f40973f6`、帧 `000110`，与此前 20 场景验证使用的样本一致。按主 README 准备环境后，在项目根目录运行：

```bash
source env/activate.sh
export CUDA_VISIBLE_DEVICES=0

python scripts/run_pipeline.py \
  --input /path/to/TravelUAV/NewYorkCity/00016766-0a99-4dda-9f89-8d62f40973f6 \
  --frame 000110 --no-ram-tags \
  --output outputs/demo_newyorkcity/pipeline

python scripts/render_sample_demo.py \
  --input outputs/demo_newyorkcity/pipeline \
  --output outputs/demo_newyorkcity/figures
```

将 `/path/to/TravelUAV` 替换为自己的数据根目录。输出目录需为空或不存在；换新名称重复运行。演示图不随源码上传。此脚本用于默认前/下四方向展示，不支持自定义方向组合。

主要输出两张 PNG：

| 图片 | 内容 |
| --- | --- |
| `front_down_semantic_masks.png` | 三行两列：第一列前视图、第二列下视图；三行依次为 RGB、GSA 原始语义掩码和透明叠加，2160×3360 |
| `unwrapped_actions.png` | 语义与径向深度的球面展开图，叠加 front/L30/R30/DOWN 的真实动作输出，3200×3060 |

语义配色沿用模型类别表：灰色为 unknown；展开图深色背景表示 15 米内没有有效观测。只使用前/下两视图，不补其他相机。窗口为全张角 20° 的三维圆锥；DOWN 在 -90° 极点，边界为覆盖全部 yaw 的南极冠。

标注三行依次为动作与 KDE 深度、获胜语义、投票占比及像素数 n。数值直接读取本次 wrapper 的结果，保持原来的 stride=2 像素池化。展开背景使用这两张图的全部有效像素，分辨率 1440×720，两轴均为 0.25°；全背景只在该演示脚本中计算。

为改善稀疏显示，背景格中已有已识别类别时优先展示该已识别类别，并为每个观察格扩大少量显示足迹。足迹按球面大圆距离计算，默认半径为 1.5 个纬度像素（0.375°），正确处理 yaw 接缝和极区。`--point-radius 0` 关闭扩点。**这些策略只影响显示背景，不改变动作值、原始语义或深度数组。**

`unwrap_data.npz` 分开保存原始和显示数组，包含扩点掩码及已识别类别优先掩码；`demo.json` 记录样本、原始动作文件 SHA256、四个标签、相机范围和展示策略。

该样本的下视图被当前 GSA 整图预测为 building，原始预测原样保留；这些是模型预测，不是仿真真值标签。front 在 15 米圆锥内无有效像素，L30/R30 的多数票为 unknown，图中同样保留这些结果。
