# Monorepo 规范

## 目录结构

```
repo_root/
├─ apps/         # 独立应用（各有 pyproject.toml）
│   ├─ api/      # FastAPI 后端
│   ├─ worker/   # Celery GPU Worker
│   └─ web/      # React 前端
├─ libs/         # 共享代码库
│   ├─ ai/       # AI 模型封装 (tts, videogen, ops, dfloat11)
│   └─ foundation/   # 基础组件 (core, db, capabilities)
├─ infra/        # Docker Compose
├─ scripts/      # 自动化脚本（无独立环境，借用 app）
│   ├─ setup/    # 模型下载/环境设置
│   ├─ test/     # Smoke 测试
│   └─ tools/    # 独立工具
├─ docs/         # 全局文档
├─ assets/       # 静态资源
├─ models/       # 模型权重 (Git Ignored)
├─ data/         # 临时缓存 (Git Ignored)
├─ logs/         # 本地日志 (Git Ignored)
└─ third_party/  # Git submodule
```

## 命令模板

| 任务 | 命令 |
|------|------|
| 一键启动 | `bash scripts/tsn_up.sh` |
| 一键停止 | `bash scripts/tsn_down.sh` |
| 服务管理 | `bash scripts/tsn_manage.sh start\|stop\|status api\|worker\|web` |
| 同步依赖 | `uv sync --project apps/api` |
| 测试 | `uv run --project apps/api pytest` |
| 代码质量 | `ruff check --fix` 或 `ruff format` |

**脚本运行（借用环境）：**

```bash
# 模型下载
uv run --project apps/api scripts/setup/download_turbodiffusion_models.py

# Smoke 测试
uv run --project apps/api scripts/test/smoke_e2e_capabilities.py --api-base http://localhost:8000

# 工具
uv run --project apps/api scripts/tools/init_db.py
```

## 依赖规则

**apps 引用 libs：** 在 `pyproject.toml` 声明 editable 依赖

```toml
[project]
dependencies = ["fastapi", "tsn-core"]

[tool.uv.sources]
tsn-core = { path = "../../libs/foundation/core", editable = true }
```

**导入路径：** `from core import xxx`（绝对路径；不使用 `libs.*` 命名空间）

## 环境变量

`.env` 禁止提交，`.env.example` 必须提交。核心变量：
- `CAP_GPU_MODE`, `WORKER_CAPABILITIES` - Worker 配置
- `S3_ENDPOINT`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`, `S3_BUCKET_NAME` - 存储
- `DATABASE_URL`, `CELERY_BROKER_URL` - 基础设施

## 禁止事项

- `apps/A` import `apps/B`（横向引用）
- `libs` import `apps`（反向引用）
- `scripts/` 包含 `pyproject.toml`
- 裸跑 `uv run scripts/xxx.py`（必须借用 app 环境）
- 硬编码路径或密钥
- `.env` 提交 Git
