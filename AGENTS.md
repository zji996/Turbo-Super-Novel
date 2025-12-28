# Monorepo 规范

## 目录结构

```
repo_root/
├─ apps/         # 独立应用（各有 pyproject.toml）
├─ libs/         # 共享代码库
├─ infra/        # Docker / K8s / Terraform
├─ scripts/      # 自动化脚本（无独立环境）
├─ docs/         # 全局文档
├─ assets/       # 静态资源
├─ models/       # 模型权重 (Git Ignored, 通过 MODELS_DIR 读取)
├─ data/         # 临时缓存 (Git Ignored, 通过 DATA_DIR 读取)
├─ logs/         # 本地日志 (Git Ignored, 通过 LOG_DIR 读取)
└─ third_party/  # Git submodule
```

## 命令模板

| 任务 | 命令 |
|------|------|
| 启动 API | `uv run --project apps/api --directory apps/api uvicorn main:app --reload` |
| 启动 Worker | `uv run --project apps/worker --directory apps/worker python main.py` |
| 同步依赖 | `uv sync --project apps/api` |
| 测试 API | `uv run --project apps/api --directory apps/api --group dev pytest` |
| 测试 Worker | `uv run --project apps/worker --directory apps/worker --group dev pytest` |
| 代码质量 | `ruff check --fix` 或 `ruff format` |

**脚本运行（借用环境）：**
- 重型任务 (CUDA/PyTorch)：`uv run --project apps/worker scripts/xxx.py`
- S3 任务 (boto3)：`uv run --project apps/api scripts/xxx.py`

## 依赖规则

**apps 引用 libs：** 在 `pyproject.toml` 声明 editable 依赖

```toml
[project]
dependencies = ["fastapi", "py_core"]

[tool.uv.sources]
py_core = { path = "../../libs/pycore", editable = true }
```

**导入路径：** `from libs.pycore import xxx`（绝对路径）

## 环境变量

`.env` 禁止提交，`.env.example` 必须提交。必需变量：
- `MODELS_DIR`, `DATA_DIR`, `LOG_DIR`
- `S3_ENDPOINT`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`, `S3_BUCKET_NAME`

## 禁止事项 (MUST NOT)

- `apps/A` import `apps/B`（横向引用）
- `libs` import `apps`（反向引用）
- `apps/` 下创建 `common/` 或 `utils/`（应放 `libs/`）
- `scripts/` 包含 `pyproject.toml`
- 裸跑 `uv run scripts/xxx.py`（必须借用 app 环境）
- 新增顶层目录
- 硬编码路径或密钥
- `.env` 提交 Git
- Crontab/脚本清理 S3 文件（用 Lifecycle Rules）
- Dockerfile 使用 `COPY ../libs`（构建上下文必须为 repo_root）
- 持久化数据存 `data/`（必须上传 S3/MinIO）
