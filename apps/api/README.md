# apps/api

FastAPI 服务（提交 TurboDiffusion 任务 / 查询结果）。

## Run

在 repo_root 执行：

```bash
cp .env.example .env
uv sync --project apps/api
uv run --project apps/api --directory apps/api uvicorn main:app --reload
```

推荐（单一入口，自动加载 `.env` 并同时启动 API+Worker）：
```bash
bash scripts/tsn_up.sh
```

依赖：

- Redis：Celery broker/result backend
- MinIO/S3：输入图片与输出视频存储

## Test

```bash
uv run --project apps/api --directory apps/api --group dev pytest
```
