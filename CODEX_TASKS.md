# CODEX_TASKS: Worker 架构清理与完善

> **状态**: 清理阶段已完成

---

## 已完成 ✅

- [x] Capability-based Worker 架构实现
- [x] `ModelManager` 互斥切换 + `CAP_GPU_MODE`
- [x] `TTSWorker` / `VideoGenWorker` 迁移
- [x] 多队列路由 + 启动脚本改造
- [x] TTS `unload()` 实现

---

## 清理任务 (不考虑向后兼容)

### 1. 删除旧代码

- [x] 删除 `apps/worker/tasks.py` (re-export 兼容模块)
- [x] 删除 `apps/worker/tts_tasks.py` (re-export 兼容模块)
- [x] 更新 `apps/worker/celery_app.py` 的 `include` 列表，移除旧模块引用

### 2. 统一任务命名

当前混用两套命名：
- 旧: `tts.synthesize`, `novel.scene.tts`, `turbodiffusion.wan22_i2v.generate`
- 新: 应该统一为 `cap.{capability}.{action}`

**改为**：
- `cap.tts.synthesize`
- `cap.tts.scene` (或 `cap.tts.novel_scene`)
- `cap.videogen.generate`

**需要同步修改**：
- [x] `apps/worker/capabilities/tts.py` - `@celery_app.task(name=...)`
- [x] `apps/worker/capabilities/videogen.py` - `@celery_app.task(name=...)`
- [x] `apps/worker/celery_app.py` - `task_routes`
- [x] `apps/api/routes/tts.py` - 调用任务的地方（无直接 `send_task` 调用）
- [x] `apps/api/routes/videogen.py` - 调用任务的地方
- [x] `apps/api/routes/novel.py` - 调用任务的地方
- [x] `libs/foundation/capabilities/src/capabilities/providers/local.py` - LocalProvider

### 3. 清理 `.env.example`

- [x] 移除旧的 `GPU_MODE` 注释块（或精简为一行说明已废弃）
- [x] 移除 `TD_RESIDENT_GPU` 相关注释
- [x] 只保留 `CAP_GPU_MODE` 作为推荐配置

### 4. VideoGen 模型真正卸载 (可选)

当前 `VideoGenWorker.unload_models()` 只标记 `_loaded=False`：
- videogen 库内部每次调用自行加载/卸载
- 如需 resident 模式生效，需重构 `videogen/inference.py`
- **建议**：暂时保持现状，后续有需求再优化

### 5. 目录结构对齐

删除文档中未实现的引用：
- [x] `apps/worker/capabilities/registry.py` (未创建；文档无引用)
- [x] `apps/worker/capabilities/postprocess.py` (未创建；文档无引用)

### 6. 测试更新

- [x] 检查 `apps/worker/tests/test_tts_tasks.py` 是否过时
- [x] 检查 `apps/worker/tests/test_worker.py` 是否过时
- [x] 确保测试覆盖新的任务名

---

## 运行示例

```bash
# 启动 Worker（支持 TTS + VideoGen）
WORKER_CAPABILITIES=tts,videogen \
CAP_GPU_MODE=resident \
bash scripts/tsn_manage.sh start worker
```

---

## 前端迁移计划

### 当前状态

前端 Studio 通过 HTTP API 调用后端，后端 API 路由内部使用 Celery `send_task` 投递任务：

| 前端组件 | 调用的 API | 后端任务 | 状态 |
|---------|-----------|---------|------|
| `TTSStudio.tsx` | `/v1/tts/jobs/with-profile` | `cap.tts.synthesize` | ✅ 已迁移 |
| `I2VStudio.tsx` | `/v1/videogen/wan22-i2v/jobs` | `cap.videogen.generate` | ✅ 已迁移 |
| `ImageStudio/` | `/v1/imagegen/jobs` | 远程 Z-Image API | ✅ 无需改动 |

### 验证清单

前端 → 后端 HTTP API → Celery 任务 全链路已更新为新命名，需验证：

- [ ] TTS Studio: 创建任务 → 轮询状态 → 播放音频
- [ ] I2V Studio: 上传图片 → 创建任务 → 轮询状态 → 播放视频
- [ ] Image Studio: 创建任务 → 轮询状态 → 显示图片

### 可选优化

1. **前端 services 文件整理**
   - 当前 `api.ts` 只包含 VideoGen，`tts.ts` 包含 TTS
   - 建议：统一命名为 `videogen.ts`，与后端保持一致

2. **类型定义同步**
   - 检查 `apps/web/src/types/` 是否与后端 Pydantic 模型一致
   - 可考虑使用 OpenAPI 生成类型

---

## 更新日志

- **2025-12-31**: Phase 1-2 完成，进入清理阶段
- **2025-12-31**: 清理阶段完成：统一 task 命名、删除旧 shim、精简 env，并补齐测试覆盖
- **2025-12-31**: 添加前端迁移验证计划
