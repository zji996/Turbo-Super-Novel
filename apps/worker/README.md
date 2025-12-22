# apps/worker

后台 Celery Worker（TurboDiffusion 推理任务）。

## Run

在 repo_root 执行：

```bash
cp .env.example .env
uv sync --project apps/worker
uv run --project apps/worker --directory apps/worker celery -A celery_app:celery_app worker -l info
```

推荐（单一入口，自动加载 `.env` 并同时启动 API+Worker）：
```bash
bash scripts/tsn_up.sh
```

DF11 文本编码器（默认支持）：

- Worker 默认包含 DF11 运行时依赖（`cupy-cuda12x` / `safetensors` / `dahuffman` 等）。
- 若存在 `models/text-encoder-df11/models_t5_umt5-xxl-enc-df11.safetensors` 且 `TD_TEXT_ENCODER_FORMAT=df11`，推理会优先使用 DF11；否则自动回退到 BF16 `.pth`。

GPU/FlashAttention（可选）：

`flash-attn` 在 `torch==2.9.*` 上目前没有对应的预编译 wheel，会触发源码编译并要求本机 CUDA Toolkit 版本与 PyTorch 编译版本一致（例如 torch `+cu128` 需要 CUDA 12.8 的 `nvcc`）。为了避免本机编译，本项目将 `flash-attn` 放在可选依赖组 `cuda` 中，并在该组内将 `torch/torchvision` 约束到 `<2.9/<0.24`（从而使用已有的 `flash-attn` wheel）：

```bash
uv sync --project apps/worker --group cuda
```

SageAttention / SageSLA（可选）：

TurboDiffusion 的 `sagesla` 依赖 SpargeAttn（Python 包名 `spas_sage_attn`，包含 CUDA 扩展）。已在 worker 中提供可选依赖组 `sagesla`：

```bash
uv sync --project apps/worker --group sagesla
```

注意：该扩展需要本机 `nvcc` 版本与当前 PyTorch 的 CUDA 版本一致；否则会在编译阶段报 CUDA mismatch（和 `flash-attn` 源码编译类似）。

依赖：

- Redis：作为 Celery broker/result backend
- MinIO/S3：输入图片与输出视频存储
- TurboDiffusion + GPU 依赖：用于实际推理（见 `third_party/TurboDiffusion/README.md`）

## Test

```bash
uv run --project apps/worker --directory apps/worker --group dev pytest
```
