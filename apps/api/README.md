# apps/api

FastAPI 服务（提交 TurboDiffusion 任务 / 查询结果）。

## Run

在 repo_root 执行：

```bash
uv sync --project apps/api
uv run --project apps/api --directory apps/api uvicorn main:app --reload
```

依赖：

- Redis：Celery broker/result backend
- MinIO/S3：输入图片与输出视频存储

## Test

```bash
uv run --project apps/api --directory apps/api --group dev pytest
```
