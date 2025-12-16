# apps/worker

后台 Worker（示例骨架）。

## Run

在 repo_root 执行：

```bash
uv sync --project apps/worker
uv run --project apps/worker --directory apps/worker python main.py
```

## Test

```bash
uv run --project apps/worker --directory apps/worker --group dev pytest
```
