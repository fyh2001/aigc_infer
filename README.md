# Qwen Image Edit - SGLang Diffusion

基于 [SGLang Diffusion](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen) 的 Qwen Image Edit 推理程序。

## 支持的模型

| 模型 | HuggingFace 路径 | 说明 |
|------|------------------|------|
| Qwen-Image-Edit | `Qwen/Qwen-Image-Edit` | 基础版 |
| Qwen-Image-Edit-2509 | `Qwen/Qwen-Image-Edit-2509` | 增强版 |
| Qwen-Image-Edit-2511 | `Qwen/Qwen-Image-Edit-2511` | 最新版（推荐） |

## 安装

```bash
# 推荐使用 uv
uv pip install 'sglang[diffusion]' --prerelease=allow

# 或者使用 pip
pip install -r requirements.txt
```

## 使用方法

### 基本用法

```bash
python run_qwen_image_edit.py \
    --model-path Qwen/Qwen-Image-Edit-2511 \
    --image-path input.jpg \
    --prompt "Turn it into a watercolor painting."
```

### 使用 LoRA 适配器

```bash
python run_qwen_image_edit.py \
    --model-path Qwen/Qwen-Image-Edit-2511 \
    --image-path input.jpg \
    --prompt "Transform into anime." \
    --lora-path prithivMLmods/Qwen-Image-Edit-2511-Anime
```

### 自定义参数

```bash
python run_qwen_image_edit.py \
    --model-path Qwen/Qwen-Image-Edit-2511 \
    --image-path input.jpg \
    --prompt "Add snow to the scene." \
    --output-dir results/ \
    --height 1024 --width 1024 \
    --steps 40 --cfg-scale 4.0 --seed 123
```

### 低显存 GPU (CPU Offload)

```bash
python run_qwen_image_edit.py \
    --model-path Qwen/Qwen-Image-Edit-2511 \
    --image-path input.jpg \
    --prompt "Make the sky sunset colors." \
    --text-encoder-cpu-offload --pin-cpu-memory
```

### 使用 URL 作为输入图片

```bash
python run_qwen_image_edit.py \
    --model-path Qwen/Qwen-Image-Edit-2511 \
    --image-path "https://example.com/photo.jpg" \
    --prompt "Remove the background."
```

## 也可以直接使用 CLI

SGLang 也提供了内置的 CLI 命令：

```bash
sglang generate \
    --model-path Qwen/Qwen-Image-Edit-2511 \
    --prompt "Transform into anime." \
    --image-path input.jpg \
    --save-output
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model-path` | `Qwen/Qwen-Image-Edit-2511` | 模型路径 |
| `--image-path` | (必填) | 输入图片路径或 URL |
| `--prompt` | (必填) | 编辑指令 |
| `--negative-prompt` | None | 负面提示词 |
| `--lora-path` | None | LoRA 权重路径 |
| `--output-dir` | `outputs/` | 输出目录 |
| `--height` / `--width` | 自动 | 输出图片尺寸 |
| `--steps` | 模型默认 | 去噪步数 |
| `--cfg-scale` | 模型默认 | Guidance Scale |
| `--seed` | 42 | 随机种子 |
| `--num-outputs` | 1 | 每个 prompt 生成的图片数 |
| `--num-gpus` | 1 | 使用的 GPU 数量 |
| `--text-encoder-cpu-offload` | False | 将文本编码器卸载到 CPU |
| `--pin-cpu-memory` | False | 固定 CPU 内存加速传输 |

## 硬件要求

- NVIDIA GPU（推荐 A100/H100，4090 也可运行）
- 使用 `--text-encoder-cpu-offload` 可在低显存 GPU 上运行
- 也支持 AMD GPU (ROCm)、Apple Silicon (MPS)
