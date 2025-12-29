# libs/ai/videogen

TurboDiffusion 集成相关的共享代码：

- 模型文件结构与路径约定（基于 `MODELS_DIR`）
- HuggingFace 直链下载清单（供 `scripts/` 使用）
- Wan2.2 I2V 推理封装（已将上游 `rcm/imaginaire/SLA` 代码 vendoring 到 `libs/ai/videogen`，运行时不依赖 `third_party/`）
