# Turbo-Super-Novel

Monorepo 结构遵循 `AGENTS.md` 约定：

- `apps/`：可运行应用（独立环境）
- `libs/`：共享代码库（被 apps 引用）
- `infra/`：Docker/K8s/Terraform
- `scripts/`：自动化脚本（借用 app 环境）
- `docs/`：全局文档
- `assets/`：静态资源/示例素材
- `models/`、`data/`、`logs/`：本地运行目录（已在 `.gitignore` 忽略）

快速开始（在 repo_root 执行）：

- API：`uv run --project apps/api --directory apps/api uvicorn main:app --reload`
- Worker：`uv run --project apps/worker --directory apps/worker python main.py`
