# Quickstart (TurboDiffusion Wan2.2 I2V)

当前项目核心目标：在本机托管并跑通 `models/turbodiffusion/TurboWan2.2-I2V-A14B-720P` 的 I2V 推理链路（API → Worker → MinIO）。

## 0. 前置条件

- Linux + NVIDIA GPU（已安装驱动，`nvidia-smi` 可用）
- `uv`、`docker`、`docker compose`
- 首次使用需初始化子模块：`third_party/TurboDiffusion`

## 1. 一次性准备：子模块

```bash
git submodule update --init --recursive third_party/TurboDiffusion
```

## 2. 配置环境变量（推荐：统一用根目录 .env）

```bash
cp .env.example .env
```

默认配置已满足本地开发：Redis `6379` / Postgres `5432` / MinIO `9000` / API `8000`。

说明：

- 推荐只维护这一个根目录 `.env`，`apps/*/env.example` 仅作为兜底示例/兼容，不建议用户再创建多个 `.env`。

## 3. 启动基础设施（Redis / Postgres / MinIO）

```bash
docker compose -f infra/docker-compose.dev.yml up -d
```

MinIO Console：`http://localhost:9001`（默认账号密码来自 `.env` 的 `S3_ACCESS_KEY/S3_SECRET_KEY`）。

如果你在 `.env` 配了 `DATABASE_URL`，后端会在首次写入时自动创建开发用表（无需额外迁移步骤）。

## 4. 安装依赖（API + Worker）

```bash
uv sync --project apps/api
uv sync --project apps/worker -p python3.12
```

## 5. 下载模型到本机（写入 models/）

默认会下载量化（quant）权重，并写入：
`models/turbodiffusion/TurboWan2.2-I2V-A14B-720P/`

```bash
uv run --project apps/api scripts/download_turbodiffusion_models.py \
  --model TurboWan2.2-I2V-A14B-720P --quantized
```

## 6.（推荐）编译 turbo_diffusion_ops（量化权重需要）

量化权重使用 int8 Linear，通常需要 `turbo_diffusion_ops`（CUDA 扩展）。详细步骤见：

- `docs/turbodiffusion_i2v_runbook.md`

## 7. GPU Smoke Test（直接跑一次 I2V）

```bash
uv run --project apps/worker scripts/smoke_turbodiffusion_i2v.py --input-index 0 --prompt-index 0
```

成功标志：输出类似 `data/turbodiffusion/smoke/i2v_0_0.mp4` 的文件路径。

## 8. 启动 API + Worker（非 Docker）

```bash
bash scripts/tsn_up.sh
```

查看状态：

```bash
bash scripts/tsn_manage.sh status
```

## 9. 调用 API（I2V）

创建任务：

```bash
curl -sS -X POST "http://localhost:8000/v1/turbodiffusion/wan22-i2v/jobs" \
  -F 'prompt=a cute cat, cinematic lighting' \
  -F "image=@assets/turbodiffusion/i2v_inputs/i2v_input_0.jpg" \
  | jq
```

查询任务（把返回的 `job_id` 替换进去）：

```bash
curl -sS "http://localhost:8000/v1/turbodiffusion/jobs/<job_id>" | jq
```

当返回 `output_url` 时，可直接在浏览器打开播放/下载。
