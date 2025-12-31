# Worker 部署指南（Capability-based）

## 关键环境变量

- `WORKER_CAPABILITIES`: Worker 支持的能力（逗号分隔），例如 `tts,videogen`
- `CAP_GPU_MODE`:
  - `resident`: 任务完成后保持当前 Capability 的模型常驻（直到切换）
  - `ondemand`: 每个任务完成后立即卸载模型并清理 CUDA 缓存

## 启动（推荐脚本）

在 `repo_root`：

```bash
bash scripts/tsn_up.sh worker
```

或只启动 worker：

```bash
WORKER_CAPABILITIES=tts,videogen CAP_GPU_MODE=resident \
bash scripts/tsn_manage.sh start worker
```

脚本会自动追加 `-Q celery,cap.tts,cap.videogen` 并默认以单并发运行（可用 `CELERY_CONCURRENCY` 显式覆盖）。

## 直接启动（调试）

```bash
WORKER_CAPABILITIES=tts,videogen CAP_GPU_MODE=ondemand \
uv run --project apps/worker --directory apps/worker \
  celery -A celery_app:celery_app worker -l info \
  -Q celery,cap.tts,cap.videogen --concurrency 1 --prefetch-multiplier 1
```

