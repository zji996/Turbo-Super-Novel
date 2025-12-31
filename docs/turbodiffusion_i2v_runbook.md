# TurboDiffusion（Wan2.2 I2V）跑通手册

本项目当前跑通的链路是 **图生视频（I2V）**：

- 前端/客户端上传图片 + prompt 到 API
- API 写入 MinIO（S3 协议）并投递 Celery 任务
- Worker 从 MinIO 拉取输入，调用 TurboDiffusion 推理，产出 mp4 再上传 MinIO
- API 提供查询接口并返回输出视频的 presigned URL

## 0. 重要前提

### 0.1 GPU / CUDA 版本匹配（关键）

`turbo_diffusion_ops` / `spas_sage_attn`（SageSLA / SpargeAttn）都是 CUDA 扩展，**编译时的 nvcc 版本需要与 PyTorch 的 CUDA 版本一致**。

例如本仓库默认 worker 环境里是 `torch==2.8.0+cu128`，对应 `torch.version.cuda == 12.8`：

```bash
uv run --project apps/worker --directory apps/worker python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

如果系统里只有 `nvcc 13.x`（比如 `/usr/local/cuda-13.1/bin/nvcc`），会遇到 CUDA mismatch，导致扩展无法编译。

下面给出一个 **不需要 sudo** 的方案：安装 CUDA 12.8 toolkit 到用户目录，并用它来编译扩展。

## 1. 一次性准备：CUDA 工具链 + CUTLASS（用于构建 turbo_diffusion_ops）

说明：

- 推理运行时不依赖 `third_party/`：上游 Python 代码已 vendoring 到 `libs/ai/videogen`。
- 量化权重需要 `turbo_diffusion_ops`（CUDA 扩展），其构建依赖 CUTLASS。
- 本仓库默认会把 CUTLASS 下载/解压到 `data/cutlass-v4.3.0`（或你也可手动设置 `CUTLASS_DIR` 指向任意 CUTLASS v4.3.0 checkout）。

## 2. 安装 CUDA 12.8 toolkit（含 nvcc 12.8，免 sudo）

下载 CUDA 12.8 runfile（约 5GB），并只安装 toolkit 到 `~/.local/cuda-12.8`：

```bash
mkdir -p data/cuda_installers
wget -O data/cuda_installers/cuda_12.8.1_570.124.06_linux.run \
  https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_570.124.06_linux.run

chmod +x data/cuda_installers/cuda_12.8.1_570.124.06_linux.run
data/cuda_installers/cuda_12.8.1_570.124.06_linux.run \
  --silent --toolkit --toolkitpath="$HOME/.local/cuda-12.8" --override --no-man-page

"$HOME/.local/cuda-12.8/bin/nvcc" --version
```

说明：

- 这里只装 toolkit（不装 driver）；driver 由系统/容器层维护。
- `data/` 已在 `.gitignore` 忽略，用于缓存 runfile 合理。

## 3. 编译并安装 `turbo_diffusion_ops`（限制编译并行度）

`turbo_diffusion_ops` 的源码已 vendoring 到 `libs/ai/ops`，推理运行时不依赖子模块源码。

仓库已提供一个一键脚本（会自动选择匹配 `torch.version.cuda` 的 `~/.local/cuda-<version>`）：

```bash
bash scripts/build_turbodiffusion_ops.sh
```

建议设置：

- `MAX_JOBS=5`：限制编译并行（照顾 64GB 内存）
- `USE_NINJA=1`：加速编译（本项目用 pip 安装 `ninja`，不依赖系统包）

```bash
export CUDA_HOME="$HOME/.local/cuda-12.8"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export MAX_JOBS=5
export USE_NINJA=1
export CUTLASS_DIR="<repo_root>/data/cutlass-v4.3.0"

# 确保 worker venv 里有 pip / ninja
uv run --project apps/worker --directory apps/worker python -m ensurepip --upgrade
uv run --project apps/worker --directory apps/worker python -m pip install -U pip setuptools wheel ninja

# torch 的动态库目录加入 LD_LIBRARY_PATH（避免 import 扩展时报 libc10.so 找不到）
export LD_LIBRARY_PATH="$(uv run --project apps/worker --directory apps/worker python -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"):$LD_LIBRARY_PATH"

# 编译并安装 turbo_diffusion_ops
uv run --project apps/worker --directory apps/worker \
  python -m pip install -e ../../libs/ai/ops --no-build-isolation
```

## 3.1 启用 SageSLA（可选，但强烈推荐）

TurboDiffusion 上游 inference 默认使用 `--attention_type sagesla`（SageSLA 是基于 SageAttention 的更快 SLA forward）。

本仓库会在启动 worker 时默认尝试安装 `cuda,sagesla` 依赖组；如果本机 CUDA 工具链不匹配导致 `uv sync` 失败，会自动降级为仅 `cuda`（仍可跑，但注意力算子会回退到 SLA）。

如果你希望 **默认就启用 SageSLA**，需要确保使用 nvcc 12.8（与 `torch==*+cu128` 对齐），然后安装 `sagesla` 组：

```bash
export CUDA_HOME="$HOME/.local/cuda-12.8"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

uv sync --project apps/worker --group sagesla
```

验证（应当能 import `spas_sage_attn`）：

```bash
uv run --project apps/worker --directory apps/worker python -c "import spas_sage_attn; print('spas_sage_attn ok')"
```

## 4. 本地 smoke test

跑通标志：生成 `data/turbodiffusion/smoke/i2v_0_0.mp4`。

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

uv run --project apps/worker \
  python scripts/test/smoke_turbodiffusion_i2v.py \
    --input-index 0 --prompt-index 0 --num-steps 1
```

## 5. 彻底本地化（避免 HuggingFace 联网）

TurboDiffusion 上游的 UMT5 tokenizer 默认通过 `transformers.AutoTokenizer.from_pretrained("google/umt5-xxl")` 加载；
如果本机无法访问 `huggingface.co`，会出现日志中的重试/超时警告。

本仓库已支持把 tokenizer 缓存到本地并启用离线模式：

1) 下载 tokenizer 文件到 `models/2v/`（只拉 tokenizer，不拉模型权重）：

```bash
uv run --project apps/api --group tools scripts/setup/cache_umt5_tokenizer.py
```

2) 在 `.env` 中开启离线并指向本地 tokenizer 目录：

```bash
TD_HF_OFFLINE=1
TD_UMT5_TOKENIZER_DIR=<repo_root>/models/2v/text-encoder/umt5-xxl-tokenizer
```

说明：

- 也可以用 `TD_UMT5_TOKENIZER` 指定 HF repo id 或本地路径（优先级低于 `TD_UMT5_TOKENIZER_DIR`）。
- `TD_HF_OFFLINE=1` 会设置 `HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE`，任何缺失文件都会直接报错（更容易排查）。

补充：

- umT5 文本编码默认走 CUDA（更快）；如果显存紧张可设置 `TD_UMT5_DEVICE=cpu`。

## 6. 启动服务（API + Worker + 基础设施）

基础设施（redis/pg/minio）：

```bash
docker compose -f infra/docker-compose.dev.yml up -d
```

启动 API + worker（非 Docker，一键脚本）：

```bash
bash scripts/tsn_up.sh
```

数据库（可选）：

- 如果设置了 `DATABASE_URL`，后端会在首次写入时自动创建开发用表。

Worker 并发（默认做了限制）：

- `scripts/tsn_manage.sh` 默认设置了 `CELERY_CONCURRENCY=5`（可通过环境变量覆盖）。
- 如果只有一张 GPU，且单个任务就会占满显存，建议把 `CELERY_CONCURRENCY` 调到 `1`。

示例：

```bash
CELERY_CONCURRENCY=1 bash scripts/tsn_up.sh
```

## 7. API 调用（I2V）

创建任务（multipart/form-data）：

- `POST /v1/turbodiffusion/wan22-i2v/jobs`
- 字段：`prompt`（必填）、`image`（必填）、`seed`、`num_steps`、`quantized`

查询任务：

- `GET /v1/turbodiffusion/jobs/{job_id}`
- 成功时返回 `output_url`（1h 有效的 presigned URL，可直接给前端播放/下载）
