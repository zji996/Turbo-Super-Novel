# Turbo-Super-Novel 架构文档

本文档描述 Turbo-Super-Novel (TSN) 项目的整体系统架构。

---

## 架构总览

![架构总览](/home/zji/.gemini/antigravity/brain/be7e6701-4104-4280-951a-8220b2c8bfa1/architecture_diagram.png)

---

## 1. 系统层次

TSN 采用**分层架构**，从上到下分为以下几个层次：

### 1.1 前端层 (`apps/web`)

基于 Vite + React + TailwindCSS 构建的 Web 应用，提供以下功能模块：

| 模块 | 说明 |
|------|------|
| **首页/仪表板** | 项目概览、快速入口 |
| **TTS Studio** | 文本转语音工作台，支持 GLM-TTS 等多种引擎 |
| **Image Studio** | 图像生成工作台，支持远程 Z-Image API |
| **Video Studio** | 视频生成工作台，基于 TurboDiffusion I2V |
| **Novel Workflow** | 有声小说/视频项目的端到端工作流 |

### 1.2 API 层 (`apps/api`)

基于 FastAPI 构建的后端服务，提供 RESTful API：

```
apps/api/
├── main.py           # FastAPI 入口，挂载路由
├── celery_app.py     # Celery 应用配置
├── routes/
│   ├── capabilities.py    # 通用 capability 路由
│   ├── imagegen.py        # 图像生成 API
│   ├── tts.py             # TTS API
│   ├── novel.py           # Novel 项目 API
│   └── llm.py             # LLM 代理 API
└── s3.py             # S3/MinIO 存储工具
```

**核心设计：Capability Router**

API 层的核心是 **Capability Router**，它根据配置将请求路由到不同的 Provider：

- **Remote Provider**：转发请求到远程 API（如 Z-Image、远程 TTS 服务）
- **Local Provider**：通过本机 Celery Worker 执行任务

### 1.3 能力路由层 (`libs/foundation/capabilities`)

提供统一的能力抽象和路由机制：

```
libs/foundation/capabilities/
├── config.py         # 配置加载（从环境变量）
├── router.py         # CapabilityRouter 实现
└── providers/
    ├── base.py       # 基类接口
    ├── local.py      # 本地执行（Celery）
    ├── remote.py     # 远程转发（HTTP）
    └── llm.py        # LLM 特殊处理
```

**支持的 Capabilities：**

| Capability | Provider 选项 | 说明 |
|------------|---------------|------|
| `tts` | local / remote | 文本转语音 |
| `imagegen` | local / remote | 图像生成 |
| `videogen` | local / remote | 视频生成 |
| `llm` | remote only | 大语言模型 |

### 1.4 Worker 层 (`apps/worker`)

基于 Celery 的异步任务执行器，运行在 GPU 机器上：

- **TTS Engine**：GLM-TTS 模型推理
- **Video Engine**：TurboDiffusion I2V 模型推理

> **注意**：图像生成 (imagegen) 目前设计为通过远程 Z-Image API 提供，不在本地 Worker 中执行。

---

## 2. 远程服务集成

### 2.1 LLM API (OpenAI/Deepseek)

用于 Prompt 优化、脚本生成等文本处理任务。

```
CAP_LLM_PROVIDER=remote
CAP_LLM_REMOTE_URL=https://api.deepseek.com/v1
CAP_LLM_REMOTE_API_KEY=sk-xxxx
```

### 2.2 Image Gen API (Z-Image)

用于高质量图像生成，部署在局域网机器上。

```
CAP_IMAGEGEN_PROVIDER=remote
CAP_IMAGEGEN_REMOTE_URL=http://192.168.8.24:8000
CAP_IMAGEGEN_REMOTE_API_KEY=test_123456
```

**Z-Image API 概览：**

| 端点 | 方法 | 说明 |
|------|------|------|
| `/v1/images/generate` | POST | 创建生成任务 |
| `/v1/tasks/{task_id}` | GET | 查询任务状态 |
| `/v1/tasks/{task_id}/cancel` | POST | 取消任务 |
| `/v1/history` | GET | 获取历史批次 |
| `/v1/history/{batch_id}` | GET | 获取批次详情 |
| `/generated-images/{path}` | GET | 访问生成的图片 |

---

## 3. 本机服务 (Worker)

### 3.1 TTS Engine (GLM-TTS)

- 支持多种中文语音合成
- 支持自定义音色（prompt audio）
- 输出格式：WAV (44100Hz)

### 3.2 Video Engine (TurboDiffusion I2V)

- Image-to-Video 生成
- 支持 DF11 量化模型（显存优化）
- 输出格式：MP4 (16fps)

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

## 5. 数据流

### 5.1 图像生成流程

```mermaid
sequenceDiagram
    participant User as 用户
    participant Web as 前端
    participant API as TSN API
    participant Router as Capability Router
    participant ZImage as Z-Image API

    User->>Web: 输入 Prompt
    Web->>API: POST /v1/imagegen/jobs
    API->>Router: get_capability_router("imagegen")
    Router->>ZImage: POST /v1/images/generate
    ZImage-->>Router: task_id
    Router-->>API: task_id
    API-->>Web: job_id
    
    loop 轮询状态
        Web->>API: GET /v1/imagegen/jobs/{job_id}
        API->>ZImage: GET /v1/tasks/{task_id}
        ZImage-->>API: status, image_url
        API-->>Web: status, image_url
    end
    
    Web->>User: 显示生成的图片
```

### 5.2 TTS 生成流程

```mermaid
sequenceDiagram
    participant User as 用户
    participant Web as 前端
    participant API as TSN API
    participant Celery as Celery Worker
    participant TTS as GLM-TTS Engine

    User->>Web: 输入文本
    Web->>API: POST /v1/tts/jobs
    API->>Celery: 提交任务
    Celery->>TTS: 模型推理
    TTS-->>Celery: WAV 文件
    Celery->>S3: 上传音频
    Celery-->>API: 任务完成
    
    Web->>API: GET /v1/tts/jobs/{job_id}
    API-->>Web: status, audio_url
    Web->>User: 播放音频
```

---

## 6. 目录结构

```
Turbo-Super-Novel/
├── apps/
│   ├── api/          # FastAPI 后端服务
│   ├── web/          # Vite + React 前端
│   └── worker/       # Celery Worker
├── libs/
│   ├── ai/           # AI 模型封装
│   │   ├── tts/      # TTS 引擎
│   │   ├── imagegen/ # 图像生成（预留）
│   │   └── videogen/ # 视频生成引擎
│   ├── business/     # 业务逻辑
│   │   └── novel/    # Novel 项目管理
│   └── foundation/   # 基础设施
│       ├── capabilities/  # 能力路由
│       ├── core/          # 通用工具
│       └── db/            # 数据库模型
├── docs/             # 文档
├── infra/            # 基础设施配置
├── scripts/          # 启动/管理脚本
├── third_party/      # 第三方依赖
└── .env              # 环境配置
```

---

## 7. 环境配置

关键环境变量：

```bash
# GPU 模式
GPU_MODE=balanced  # fast | balanced | lowvram

# 服务连接
CELERY_BROKER_URL=redis://localhost:6379/0
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/tsn
S3_ENDPOINT=http://localhost:9000

# Capability 路由
CAP_TTS_PROVIDER=local
CAP_IMAGEGEN_PROVIDER=remote
CAP_IMAGEGEN_REMOTE_URL=http://192.168.8.24:8000
CAP_IMAGEGEN_REMOTE_API_KEY=test_123456
CAP_VIDEOGEN_PROVIDER=local
CAP_LLM_PROVIDER=remote
```

---

## 8. 扩展性

系统设计支持以下扩展：

1. **新增 Capability**：在 `libs/foundation/capabilities` 中添加配置和 Provider
2. **切换 Provider**：通过环境变量切换 local/remote
3. **多机部署**：API 和 Worker 可分离部署，通过 Redis 通信
4. **存储后端**：支持本地存储或 S3/MinIO

---

*最后更新：2025-12-30*
