# apps/api

FastAPI 服务（示例骨架）。

## Run

在 repo_root 执行：

```bash
uv sync --project apps/api
uv run --project apps/api --directory apps/api uvicorn main:app --reload
```

## Test

```bash
uv run --project apps/api --directory apps/api --group dev pytest
```
