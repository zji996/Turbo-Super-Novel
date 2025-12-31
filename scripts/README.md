# scripts

脚本目录，按功能分类。

## 服务管理（根目录）

核心脚本直接放在 `scripts/` 根目录：

```bash
# 一键启动 API + Worker + Web
bash scripts/tsn_up.sh

# 一键停止
bash scripts/tsn_down.sh

# 精细控制
bash scripts/tsn_manage.sh start api
bash scripts/tsn_manage.sh start worker
bash scripts/tsn_manage.sh start web
bash scripts/tsn_manage.sh stop api
bash scripts/tsn_manage.sh status
```

## setup/ - 环境设置

模型下载、tokenizer 缓存等一次性设置脚本。借用 `apps/api` 或 `apps/worker` 环境：

```bash
# 下载 TurboDiffusion 模型
uv run --project apps/api scripts/setup/download_turbodiffusion_models.py --model TurboWan2.2-I2V-A14B-720P --quantized

# 下载 GLM-TTS 模型
uv run --project apps/api scripts/setup/download_glm_tts_models.py

# 缓存 UMT5 tokenizer（离线推理用）
uv run --project apps/api scripts/setup/cache_umt5_tokenizer.py

# 压缩 T5 到 DF11 格式（可选加速）
uv run --project apps/worker scripts/setup/compress_turbodiffusion_t5_dfloat11.py
```

## test/ - 测试脚本

E2E smoke test 和单模块测试：

```bash
# E2E smoke（需要 API + Worker 已启动）
uv run --project apps/api scripts/test/smoke_e2e_capabilities.py --api-base http://localhost:8000

# TurboDiffusion 单模块 smoke（仅 GPU 推理）
uv run --project apps/worker scripts/test/smoke_turbodiffusion_i2v.py --input-index 0 --prompt-index 0
```

## tools/ - 独立工具

```bash
# 初始化数据库
uv run --project apps/api scripts/tools/init_db.py

# 视频拼接工具
uv run --project apps/worker scripts/tools/concat_videos.py --help
```

## build_turbodiffusion_ops.sh

构建 TurboDiffusion CUDA 扩展（`turbo_diffusion_ops`）：

```bash
bash scripts/build_turbodiffusion_ops.sh
```

详见 `docs/turbodiffusion_i2v_runbook.md`。
