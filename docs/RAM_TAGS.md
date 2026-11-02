# 前视图、下视图 RAM 标签接口

使用官方 [Recognize Anything Model](https://github.com/xinyu1205/recognize-anything)，固定源码提交 `7cb804a8609e9f4b1a50b7f31436d2df40bb9481`，模型为 **RAM Swin-L / 384**。复用原始预处理、4,585 类词表、各类别阈值和 `generate_tag`，前/下两张图合成一个 batch。不是 RAM++，也不是把 GSA 固定类别当作 RAM 标签。

## 1. 构建资源

在主项目 cu118 环境中，从项目根目录执行。已存在对应版本时跳过下载，不安装 RAM 上游的整份训练依赖，以保留本项目 timm/torch/transformers 版本。

```bash
mkdir -p third_party assets/checkpoints

git clone https://github.com/xinyu1205/recognize-anything.git third_party/recognize-anything
git -C third_party/recognize-anything checkout --detach 7cb804a8609e9f4b1a50b7f31436d2df40bb9481
pip install -r env/requirements-ram.txt -c env/constraints.txt

HF_HUB_OFFLINE=0 hf download xinyu1205/Recognize_Anything-Tag2Text \
  ram_swin_large_14m.pth --repo-type space \
  --revision 7c76cec1377de0d46df8091a032c4b667aea4459 \
  --local-dir assets/checkpoints
```

权重约 5.63 GB，下载后为普通文件。SHA256：`15c729c793af28b9d107c69f85836a1356d76ea830d4714699fb62e55fcc08ed`，来源和版本记录在 [sources.json](../config/sources.json)。RAM 的 tokenizer 直接复用项目已准备的本地 BERT，不另行下载。

默认资源路径已经写进 `resources.example.json` 的 `ram` 节。使用其他位置时在自己的 `resource.local.json` 添加覆盖项；相对路径仍以项目根目录为基准：

```json
{
  "ram": {
    "source": "/path/to/recognize-anything",
    "checkpoint": "/path/to/ram_swin_large_14m.pth"
  }
}
```

这是 RAM 的独立设置，不需添加空的强制 `paths` 条目。不使用 RAM 时不加载源码、权重或模型。

## 2. 单独查询两张 RGB

```bash
python scripts/tag_rgb_views.py \
  --front-rgb /path/to/sample/frontcamera/000035.png \
  --down-rgb /path/to/sample/downcamera/000035.png
```

只需两张 RGB，不需要深度、姿态或 GSA 分割。stdout 为：

```json
{
  "front": {"tags": ["example tag"], "tags_zh": ["示例标签"]},
  "down": {"tags": ["example tag"], "tags_zh": ["示例标签"]}
}
```

示例仅说明字段。英文标签使用 `tags`，中文使用 `tags_zh`，均为字符串列表，无命中时为空列表。没有将固定类别 ID 或模型置信度伪装成标签。默认无日志、不写文件，支持 `--config`、`--device`、`--verbose` 和可选的 `--output tags.json`。

## 3. 统一 wrapper

```bash
python scripts/run_pipeline.py --input /path/to/front_down_rgbd_sample
```

默认前/下输入同时输出 `{"actions":[四方向结果],"ram_tags":{"front":{...},"down":{...}}}`。输入必须包含前/下的 RGB 和深度；只有一个完整帧可省略 `--frame`。`--no-ram-tags` 跳过 RAM，恢复仅动作列表的输出。

RAM 始终处理完整 RGB，即使对应方向没有 15 米内深度点也会输出图像标签。GSA 的动作语义仍来自配置的 GroundingDINO 类别及 MobileSAM 掩码，两者是不同输出，不自动把 RAM 标签替换成分割提示词。

## 4. Python 与模型复用

```python
from src.ram_tags import RAMViewTagger

tagger = RAMViewTagger(device="cuda:0")
tags = tagger.tag("front.png", "down.png")
front_tags = tags["front"]["tags"]
down_tags = tags["down"]["tags"]
# 下一对图片继续复用 tagger，不重复加载模型。
```

参数也可为 PIL 图像或 `uint8 (H,W,3)` RGB NumPy 数组。RGB 数组不做 BGR 转换；OpenCV 输入需调用方先转为 RGB。

统一核心 API：

```python
from src.pipeline import GSAActionPipeline

pipeline = GSAActionPipeline(device="cuda:0")
result = pipeline.run("/path/to/sample", ram_tags=True)
# 或仅用同一对象的 RGB 接口：
tags = pipeline.tag_rgb_views(front_rgb, down_rgb)
```

RAM 首次调用加载一次，之后复用。命令行连续样本可用 `--stream`，避免每个进程重复加载 GSA 和 RAM。只需 GSA 感知时，核心 `.run()` 默认不启用 RAM；命令行 wrapper 在选中视图包含前和下时默认启用，其他视图组合默认跳过。
