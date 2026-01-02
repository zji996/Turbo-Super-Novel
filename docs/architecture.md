# Turbo-Super-Novel 架构文档

本文档描述 Turbo-Super-Novel (TSN) 项目的整体系统架构。

---

## 1. 系统层次

TSN 采用**分层架构**，从上到下分为以下几个层次：

### 1.1 前端层 (`apps/web`)

基于 Vite + React + CSS 构建的 Web 应用，提供以下功能模块：

| 模块 | 路由 | 说明 |
|------|------|------|
| **TTS Studio** | `/tools/tts` | 文本转语音，支持 Speaker Profile |
| **Image Studio** | `/tools/image` | 图像生成（远程 Z-Image API）|
| **I2V Studio** | `/tools/i2v` | 图生视频（TurboDiffusion）|
| **LLM Studio** | `/tools/llm` | 对话测试/Prompt 调优 |

**前端特性**：
- **懒验证 + 健康跟踪**：任务成功/失败驱动能力状态，侧边栏实时显示
- **AI 增强**：各 Studio 支持 `enhance_prompt` 开关，调用 LLM 优化提示词

### 1.2 API 层 (`apps/api`)

基于 FastAPI 构建的后端服务：

```
apps/api/routes/
├── capabilities.py   # 能力探测 /v1/capabilities/status
├── imagegen.py       # 图像生成 API
├── videogen.py       # 视频生成 API
├── tts.py            # TTS API
├── llm.py            # LLM 代理 API
└── novel.py          # Novel 项目 API
```

**核心设计：Capability Router**

根据配置将请求路由到不同的 Provider：
- **Remote Provider**：转发到远程 API（Z-Image、Deepseek LLM）
- **Local Provider**：通过本机 Celery Worker 执行

### 1.3 能力路由层 (`libs/foundation/capabilities`)

```
libs/foundation/capabilities/
├── config.py         # 从环境变量加载配置
├── router.py         # CapabilityRouter
└── providers/
    ├── base.py       # 基类
    ├── local.py      # Celery 调用
    ├── remote.py     # HTTP 转发
    └── llm.py        # LLM（含 enhance_prompt）
```

**支持的 Capabilities**：

| Capability | Provider | 说明 |
|------------|----------|------|
| `tts` | local | GLM-TTS 语音合成 |
| `imagegen` | remote | Z-Image 图像生成 |
| `videogen` | local | TurboDiffusion I2V |
| `llm` | remote | Deepseek/OpenAI 对话 |

### 1.4 Worker 层 (`apps/worker`)

基于 Celery 的异步任务执行器（GPU）：

```
apps/worker/
├── celery_app.py       # Celery 配置
├── model_manager.py    # GPU 模型管理（互斥切换）
└── capabilities/
    ├── tts.py          # cap.tts.synthesize
    └── videogen.py     # cap.videogen.generate
```

**Capability-based 架构**：
- 单进程单并发（`--concurrency 1`）
- 多队列监听（`cap.tts, cap.videogen`）
- 动态模型切换 + 显存释放

---

## 2. 能力健康状态机

前端采用**懒验证 + 任务驱动**模式：

```
┌──────────┐    首次探测      ┌──────────┐
│ unknown  │ ──────────────▶ │ available│ ◀─── 任务成功
└──────────┘   Worker在线    └────┬─────┘
      │                            │
      │ Worker不在线               │ 任务失败
      ▼                            ▼
┌──────────┐                 ┌──────────┐
│unavailable│ ◀──────────────│  error   │
└──────────┘                 └──────────┘
```

**优点**：
- 不主动测试，避免 Worker 能力切换
- 真实反映运行时健康状态

---

## 3. LLM 辅助能力

所有生成类 API 支持 `enhance_prompt` 参数：

```json
POST /v1/imagegen/jobs
{
  "prompt": "一只猫",
  "enhance_prompt": true  // 调用 LLM 优化
}
```

**Context 类型**：
| Context | 用途 |
|---------|------|
| `image_generation` | 图像提示词优化 |
| `video_generation` | 视频提示词优化 |
| `tts_text` | TTS 文本分句优化 |

---

## 4. 基础设施

```yaml
# infra/docker-compose.dev.yml
services:
  redis:      # Celery Broker
  postgres:   # 任务状态持久化
  minio:      # S3 兼容存储
```

---

## 5. 目录结构

```
Turbo-Super-Novel/
├── apps/
│   ├── api/          # FastAPI 后端
│   ├── web/          # React 前端
│   └── worker/       # Celery Worker
├── libs/
│   ├── ai/           # AI 模型封装
│   │   ├── tts/      # GLM-TTS
│   │   ├── videogen/ # TurboDiffusion
│   │   ├── ops/      # CUDA 扩展
│   │   └── dfloat11/ # DF11 量化
│   └── foundation/
│       ├── capabilities/  # 能力路由
│       ├── core/          # 通用工具
│       └── db/            # 数据库模型
├── scripts/
│   ├── setup/        # 模型下载
│   ├── test/         # Smoke 测试
│   └── tools/        # 工具脚本
├── docs/             # 文档
└── infra/            # Docker Compose
```

---

## 6. 环境配置

```bash
# Worker 配置
WORKER_CAPABILITIES=tts,videogen
CAP_GPU_MODE=resident  # resident | ondemand

# 基础设施
CELERY_BROKER_URL=redis://localhost:6379/0
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/tsn
S3_ENDPOINT=http://localhost:9000

# Capability 路由
CAP_TTS_PROVIDER=local
CAP_IMAGEGEN_PROVIDER=remote
CAP_IMAGEGEN_REMOTE_URL=http://192.168.8.24:8000
CAP_IMAGEGEN_REMOTE_API_KEY=xxx
CAP_VIDEOGEN_PROVIDER=local
CAP_LLM_PROVIDER=remote
CAP_LLM_REMOTE_URL=https://api.deepseek.com/v1
CAP_LLM_REMOTE_API_KEY=sk-xxx
```

---

## 7. 扩展性

| 扩展方式 | 说明 |
|----------|------|
| **新增 Capability** | 在 `libs/foundation/capabilities` 添加配置 |
| **切换 Provider** | 环境变量 `CAP_*_PROVIDER=local\|remote` |
| **多机部署** | API 和 Worker 分离，Redis 通信 |
| **新增 Studio** | 前端添加页面 + 路由 |

---

*最后更新：2026-01-02*
