# Krea2 SuperEdit

**用参考图和自然语言，训练自己的图像编辑模型。**

Krea2 SuperEdit 是基于 Krea 2 与 Diffusers 的多参考图编辑训练工具，面向指令编辑、参考风格学习和多图条件生成。将参考图、编辑指令与目标图组织成数据集，即可训练自己的编辑适配器，并通过同一套 Pipeline 完成推理。

从准备数据、训练 LoRA 到查看预览、保存模型，整个流程由独立训练入口完成。

## 核心能力

- **多张参考图，共同指导编辑。** 同时输入待编辑图、风格图或其他辅助参考，让模型学习不同参考信息与编辑指令之间的关系。
- **适应真实数据的尺寸。** 同一个 batch 可以包含不同宽高比、分辨率和参考图数量的样本，无需预先整理成固定尺寸桶。
- **可学习的参考表示。** 通过虚拟 token 学习参考信息，并支持按样本随机关闭对原始参考图的直接读取，用于探索参考风格的提取与迁移。
- **训练过程可观察、可继续。** 支持固定种子预览、W&B 记录、EMA 和断点续训，便于比较不同阶段的编辑效果。
- **按硬件资源选择训练方式。** 提供权重量化、梯度检查点和多卡 ZeRO-2；注意力自动选择 FA4、Flex Attention 或 SDPA。

## 快速开始

### 1. 安装与模型准备

在 Python 3.10+、CUDA 环境中安装依赖：

```bash
git clone https://github.com/chinoll/krea2-superedit
cd krea2-superedit
pip install -r requirements.txt
```

默认使用 `krea/Krea-2-Raw`。请先在 Hugging Face 获得模型访问权限，并通过 `hf auth login` 登录。

模型既可以从 Diffusers 仓库或本地目录加载，也可以直接使用原始 `.safetensors` 文件。原始权重会在训练启动时自动转换并加载，无需提前导出一份新权重。

### 2. 准备数据

使用 JSONL 文件描述数据，每行对应一组「参考图 + 指令 + 目标图」：

```jsonl
{"id":"edit-0001","target":{"image":"targets/0001.png","caption":"Repaint the scene in the watercolor style of the second reference."},"references":[{"role":"source","image":"references/source.png"},{"role":"reference","image":"references/style.png"}]}
```

`target` 是期望得到的编辑结果；`source` 是待编辑图，`reference` 是风格、主体等辅助参考。图像路径相对于 JSONL 文件所在目录解析。只使用一张参考图时，保留对应条目即可。

数据处理支持不同尺寸的图像与 RGBA 素材。更多样例见 [manifest.example.jsonl](configs/manifest.example.jsonl)。

### 3. 启动训练

以 [configs/train.yaml](configs/train.yaml) 为起点。单卡首次运行可修改其中以下字段，其余配置保留：

```yaml
output_dir: output/krea2edit

distributed:
  deepspeed_zero2: false

data:
  manifest: data/train/manifest.jsonl

model:
  name_or_path: krea/Krea-2-Raw

train:
  batch_size: 1
  steps: 2000
  loss:
    weights:
      fm: 1.0
      perceptual: 0.0
      pixel: 0.0
      self_flow: 0.0

logging:
  backend: null
```

这个起点使用 Flow Matching 训练。需要时，可在配置中组合感知损失、像素损失和 Self-Flow，并启用 W&B 记录。

```bash
accelerate launch --num_processes 1 train.py --config configs/train.yaml
```

多卡训练时，将 `distributed.deepspeed_zero2` 设为 `true`，并在 [Accelerate 配置](configs/accelerate_zero2.yaml) 中设置 GPU 数量：

```bash
accelerate launch --config_file configs/accelerate_zero2.yaml \
  train.py --config configs/train.yaml
```

仓库也提供 [8 卡 Muon 训练配置](configs/train_8gpu_muon.yaml)，可按数据路径与设备情况调整。

## 使用训练好的模型

`Krea2EditPipeline` 负责模型加载、参考图编码和图像生成。加载训练导出的适配器后，即可使用与训练一致的编辑流程：

```python
import torch
from PIL import Image
from krea2edit import Krea2EditPipeline

pipe = Krea2EditPipeline.from_pretrained(
    "krea/Krea-2-Raw",
    torch_dtype=torch.bfloat16,
).to("cuda")
pipe.load_edit_adapter("output/krea2edit/checkpoint-00000250/lora")

image = pipe(
    prompt="Repaint the scene in the watercolor style of the second reference.",
    references=[
        Image.open("source.png").convert("RGB"),
        Image.open("style.png").convert("RGB"),
    ],
    height=1024,
    width=1024,
    num_inference_steps=20,
    guidance_scale=4.5,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]

image.save("edited.png")
```

适配器包含 LoRA 和可学习的虚拟 token，使用 `load_edit_adapter` 一起恢复。启用 EMA 后，也可以加载 checkpoint 中的 `lora_ema/`。

输出宽高使用 16 的倍数。传入 `references=[]` 可进行文生图。

## 控制模型如何学习参考图

每个编辑样本的输入顺序为 `参考图 | 虚拟 token | 文本 | 目标图`。虚拟 token 来自可学习的 embedding，用于承接参考图的信息。

| 训练模式 | 文本与目标图如何读取参考信息 |
| --- | --- |
| 默认模式 | 同时读取原始参考图与虚拟 token |
| 切断模式 | 关闭对原始参考图的直接读取，通过虚拟 token 获取参考信息 |

两种模式都保留完整输入，文本与目标图始终可以互相读取。参考图只读取参考图；虚拟 token 读取参考图和虚拟 token，不读取文本与目标图。这一策略作用于 DiT，Qwen-VL 的图文编码保持原样。

`model.num_virtual_tokens` 控制虚拟 token 数量，默认 16；`train.reference_attention_dropout` 控制每个样本进入切断模式的概率，`0` 表示不切断，`1` 表示全部切断。推理时可通过 `drop_reference_attention=True` 选择切断模式。

参考图固定在 `t=0`，推理默认复用其 KV cache，减少每一步对参考图的重复计算。虚拟 token、文本和目标图随当前时间步重新计算。

## 预览、保存与继续训练

训练支持从独立验证集定期生成预览，将参考图、生成结果和目标图放在同一张图中，方便观察编辑是否符合指令、是否学到了参考特征。在配置的 `sample` 部分设置验证集与样本 ID，再开启 `sample.enabled` 即可使用。

训练产物保存到 `output_dir`：

| 内容 | 用途 |
| --- | --- |
| `samples/` | 训练过程中的预览图 |
| `checkpoint-*/lora/` | Pipeline 使用的 LoRA 与虚拟 token 权重 |
| `checkpoint-*/lora_ema/` | 开启 EMA 后导出的平滑权重 |
| 完整 `checkpoint-*` 目录 | 恢复模型、优化器及训练进度 |

将 `train.resume_from` 设为完整 checkpoint 目录即可继续训练。项目同时支持 ComfyUI 格式导出；包含虚拟 token 的模型需要配套自定义节点。

训练选项集中在 [默认配置](configs/train.yaml) 与 [多卡配置](configs/train_8gpu_muon.yaml) 中，包括 LoRA、量化、损失函数、优化器和采样设置。

## 许可证

项目代码采用 [GNU GPL v3.0](LICENSE)。模型权重遵循各自的上游许可。
