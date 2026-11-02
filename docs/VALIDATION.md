# 发布验证记录

2026-11-02，以远端 `main` 的 `744d00edcb0da5164433475cc4c7448c383895f8` 为基点整理发布版本。保留原有采集、RPC 修复与点云几何逻辑；完整点云未新增严格无人机中心半径裁剪。

## 回归测试

```bash
python -m pytest tests -q
```

结果：**118 passed**。测试覆盖：

- 五视角全部 31 种非空组合；输入中删除未选择的视图，只读取选中图像，并验证保存后的采集目录可以再次输入。
- 自定义方向、JSON 输出、配置继承、默认前/下 RAM 行为，以及单视图自动跳过 RAM。
- 直接标量几何与独立 XYZ 参考的比较：姿态、相机偏移、range/z-depth、15 米边界、空回波与 unknown。
- 禁止直接方向 wrapper 调用反投影、点云融合及体素函数。
- RGB/语义点云的 NPZ、PLY、PNG、HTML 输出；异常深度诊断；采集与补采位姿/RPC 适配。

14 条警告来自 Matplotlib 对 Pyparsing 旧接口的调用，不影响测试结果。单元测试中的语义模型使用固定图像替身；下面额外验证真实模型。

## 实际模型与 GPU

环境：Python 3.10.22、PyTorch 2.0.1+cu118、torchvision 0.15.2+cu118、NumPy 1.26.4、RTX4090。使用现有依赖与本地权重，没有重新下载或修改公共数据。

- `python -m pip check`：通过。
- `python scripts/check_env.py --cuda --models`：MobileSAM 严格权重加载通过，GroundingDINO CUDA 算子数值检查通过。
- 数据样本：`NewYorkCity/00016766-0a99-4dda-9f89-8d62f40973f6`，帧 `000110`。

| 流程 | 结果 |
| --- | --- |
| 默认前/下 RGB-D → GSA + RAM | 完成四方向输出和两张 RGB 的 RAM 标签；710 个采样像素参与方向池化；未生成点云 |
| 单左视图 + 自定义 `left_side`，yaw=-90°、elevation=0° | 完成实际 GSA 推理；20 个采样像素，KDE 深度 13.51094 米，语义 unknown；RAM 自动跳过 |
| 单后视图 + 后方查询 | 正常返回 no_return，跳过无目标像素的模型推理 |
| 五视角导入 → GSA 分割 → 点云导出 | 五视角分割均完成；使用 `--max-depth 15` 导出 18,596 点；NPZ 含 xyz/rgb/view_id/semantic_id/instance_id/confidence；PLY、PNG 与 HTML 均检查通过 |

原始输出保留在本机独立发布工作目录的 `outputs/release_smoke/`，不纳入 Git。完整点云的 `--max-depth 15` 是相机深度筛选；直接方向特征的 15 米限制按无人机中心计算。二者范围定义不同。

本次没有启动或操作实时仿真环境。已有两个异常深度场景的限制见 [补采说明](DEPTH_AUDIT_AND_RECAPTURE.md)。本记录验证当前已安装环境，未宣称在空白服务器上重新安装了整套依赖。
