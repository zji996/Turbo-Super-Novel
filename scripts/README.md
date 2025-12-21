# scripts

脚本目录（无独立环境，必须借用某个 app 的环境运行）。

示例：

```bash
uv run --project apps/api scripts/example.py
```

## TurboDiffusion 模型下载

借用 `apps/api` 环境运行（需要先 `uv sync --project apps/api`）：

```bash
uv run --project apps/api scripts/download_turbodiffusion_models.py --model TurboWan2.2-I2V-A14B-720P --quantized
```

## TurboDiffusion I2V Smoke Test

借用 `apps/worker` 环境运行（需要 GPU）：

```bash
uv sync --project apps/worker -p python3.12
uv run --project apps/worker scripts/smoke_turbodiffusion_i2v.py --input-index 0 --prompt-index 0
```

## Init DB (dev)

```bash
uv run --project apps/api scripts/init_db.py
```

## 一键启动/停止（非 Docker）

日志会写入 `logs/`（仅 `.log` 文件）；PID 文件写入 `data/tsn_runtime/pids/`（用于按仓库路径停止）。

```bash
# 一键启动 API + Worker
bash scripts/tsn_up.sh

# 一键停止 API + Worker
bash scripts/tsn_down.sh

# 单独启动/停止
bash scripts/tsn_api_start.sh
bash scripts/tsn_api_stop.sh
bash scripts/tsn_worker_start.sh
bash scripts/tsn_worker_stop.sh

# 查看状态
bash scripts/tsn_manage.sh status
```
