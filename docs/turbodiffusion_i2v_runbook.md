# TurboDiffusion（Wan2.2 I2V）跑通手册

本项目当前跑通的链路是 **图生视频（I2V）**：

- 前端/客户端上传图片 + prompt 到 API
- API 写入 MinIO（S3 协议）并投递 Celery 任务
- Worker 从 MinIO 拉取输入，调用 TurboDiffusion 推理，产出 mp4 再上传 MinIO
- API 提供查询接口并返回输出视频的 presigned URL

## 0. 重要前提

### 0.1 GPU / CUDA 版本匹配（关键）

`turbo_diffusion_ops` 是 CUDA 扩展，**编译时的 nvcc 版本需要与 PyTorch 的 CUDA 版本一致**。

例如本仓库默认 worker 环境里是 `torch==2.8.0+cu128`，对应 `torch.version.cuda == 12.8`：

```bash
uv run --project apps/worker --directory apps/worker python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

如果系统里只有 `nvcc 13.x`（比如 `/usr/local/cuda-13.1/bin/nvcc`），会遇到 CUDA mismatch，导致扩展无法编译。

下面给出一个 **不需要 sudo** 的方案：安装 CUDA 12.8 toolkit 到用户目录，并用它来编译扩展。

## 1. 一次性准备：拉齐 third_party 子模块

TurboDiffusion 上游包含 `cutlass` 子模块，先初始化：

```bash
git submodule update --init --recursive third_party/TurboDiffusion
```

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

`turbo_diffusion_ops` 随 `third_party/TurboDiffusion` 一起安装/构建。

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

# 确保 worker venv 里有 pip / ninja
uv run --project apps/worker --directory apps/worker python -m ensurepip --upgrade
uv run --project apps/worker --directory apps/worker python -m pip install -U pip setuptools wheel ninja

# torch 的动态库目录加入 LD_LIBRARY_PATH（避免 import 扩展时报 libc10.so 找不到）
export LD_LIBRARY_PATH="$(uv run --project apps/worker --directory apps/worker python -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"):$LD_LIBRARY_PATH"

# 编译并安装 TurboDiffusion（会构建 turbo_diffusion_ops）
uv run --project apps/worker --directory apps/worker \
  python -m pip install -e ../../third_party/TurboDiffusion --no-build-isolation
```

## 4. 本地 smoke test（直接用 third_party 里的 assets）

跑通标志：生成 `data/turbodiffusion/smoke/i2v_0_0.mp4`。

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

uv run --project apps/worker --directory apps/worker \
  python scripts/smoke_turbodiffusion_i2v.py \
    --inputs-dir third_party/TurboDiffusion/assets/i2v_inputs \
    --input-index 0 --prompt-index 0 --num-steps 1
```

补充：

- umT5 文本编码默认走 CPU（避免 GPU 显存峰值），可用 `TD_UMT5_DEVICE=cuda` 强制上 GPU（不建议在 32GB 显存机器上开启）。

## 5. 启动服务（API + Worker + 基础设施）

基础设施（redis/pg/minio）：

```bash
docker compose -f infra/docker-compose.dev.yml up -d
```

启动 API + worker（非 Docker，一键脚本）：

```bash
bash scripts/tsn_up.sh
```

Worker 并发（默认做了限制）：

- `scripts/tsn_manage.sh` 默认设置了 `CELERY_CONCURRENCY=5`（可通过环境变量覆盖）。
- 如果只有一张 GPU，且单个任务就会占满显存，建议把 `CELERY_CONCURRENCY` 调到 `1`。

示例：

```bash
CELERY_CONCURRENCY=1 bash scripts/tsn_up.sh
```

## 6. API 调用（I2V）

创建任务（multipart/form-data）：

- `POST /v1/turbodiffusion/wan22-i2v/jobs`
- 字段：`prompt`（必填）、`image`（必填）、`seed`、`num_steps`、`quantized`

查询任务：

- `GET /v1/turbodiffusion/jobs/{job_id}`
- 成功时返回 `output_url`（1h 有效的 presigned URL，可直接给前端播放/下载）

