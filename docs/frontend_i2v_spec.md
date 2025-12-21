# 前端：I2V MVP + 为「视频小说平台」留扩展位

目标（当前）：先把 **图生视频（I2V）** 跑通，前端以最小可用为原则实现“上传图片 → 提交任务 → 轮询 → 播放/下载结果”闭环。

定位（后续演进）：I2V 会长期保留，但它应当被实现为「生成工具/工作流」中的一个子项，未来会叠加「视频小说平台」能力（素材库、角色/场景管理、分镜/镜头列表、批量生成与队列、项目沉淀与复用）。

## 0. 信息架构建议（MVP 不必全做）

- **Tools（生成工具）**：面向一次性生成/实验（例如 I2V、T2V、风格化等）
- **Projects（项目）**：面向视频小说生产（novel → chapters → scenes → shots → jobs）
- **Assets（素材库）**：图/视频/音频/参考图/提示词模板等，可在 Tools/Projects 复用

MVP 只需先实现 `Tools/I2V` 一个页面；导航/路由可以预留 `Projects` / `Assets` 的入口（可先显示为 disabled/Coming Soon）。

## 1. 产品形态（建议）

MVP：一个页面即可（建议命名为 I2V Studio / Image-to-Video）：

- 左侧：输入区（图片上传、prompt、参数）
- 右侧：任务状态与结果（进度/错误、视频播放器、下载按钮）

扩展（为视频小说留空间）：

- 页面结构建议按「可插拔的 panel」组织：InputPanel / ParamsPanel / JobPanel / ResultPanel
- JobCard 的字段设计尽量通用（支持未来的 T2V、分镜批量等）：`job_type`、`inputs`、`params`、`status`、`output_url`

## 2. 关键交互流程

### 2.1 提交任务

1. 用户选择一张图片（jpg/png）
2. 输入 prompt（必填，支持多行）
3. 选择参数（可选）：
   - `num_steps`：默认 4（建议下拉 1–4）
   - `seed`：默认 0（支持随机按钮）
   - `quantized`：默认 true（本项目默认下载的是 quant 权重）
4. 点击“生成视频”
5. 前端调用 `POST /v1/turbodiffusion/wan22-i2v/jobs`，拿到 `job_id`

提交时建议立即创建一个本地 job card 并开始轮询。

### 2.2 轮询与状态展示

轮询接口：`GET /v1/turbodiffusion/jobs/{job_id}`

建议：

- 轮询间隔：2s（RUNNING）/ 5s（PENDING）/ 10s（长时间无变化）
- 最长超时：例如 30min（超时后提示用户“任务可能仍在后台运行”并提供继续查询入口）
- 展示两套状态：
  - Celery 状态（`payload.status`）：PENDING / STARTED / SUCCESS / FAILURE
  - DB 状态（`payload.db.status`，若存在）：CREATED / SUBMITTED / STARTED / DOWNLOADED / RUNNING / UPLOADED / SUCCEEDED / FAILED

### 2.3 成功态（播放/下载）

当接口返回 `output_url` 时：

- `<video controls src={output_url} />` 直接播放
- “下载”按钮：直接打开 `output_url` 或 `a[href]` 下载
- 可选：展示输入缩略图、prompt、参数、耗时（前端可自行计时）

### 2.4 失败态

当 Celery 状态为 FAILURE 或响应中带 `error` 时：

- 展示错误信息（原样显示，注意换行）
- 给出重试按钮（保留原参数、允许换 seed）

## 3. API 契约（当前后端实现）

### 3.1 创建任务

`POST /v1/turbodiffusion/wan22-i2v/jobs`（`multipart/form-data`）

- `prompt: string`（必填）
- `image: file`（必填）
- `seed: int`（默认 0）
- `num_steps: int`（默认 4）
- `quantized: bool`（默认 true）

响应（示例字段）：

- `job_id`
- `status: "submitted"`
- `input.bucket/key`
- `output.bucket/key`
- `db.persisted/db.error`

### 3.2 查询任务

`GET /v1/turbodiffusion/jobs/{job_id}`

响应可能包含：

- `status`（Celery 状态）
- `db`（DB 信息，可能没有）
- `result`（成功时）
- `output_url`（成功时，1h 有效）
- `error`（失败时）

### 3.3 模型与环境检查（可选：前端诊断面板）

`GET /v1/turbodiffusion/models`

可用于展示：

- 模型文件是否存在（`exists`）
- 模型文件路径（用于运维排障）

`GET /health` / `GET /paths` 可用于基础连通性与目录诊断。

## 4. 工程建议（非强制）

- API Base URL 用环境变量配置（如 `VITE_API_BASE_URL` / `NEXT_PUBLIC_API_BASE_URL`）
- 本地开发优先使用同源反代（避免 CORS）；或后端加 CORS 中间件（如需要再做）
- 轮询逻辑抽象成 hook / composable，页面只负责渲染
- Job 列表保存在 localStorage（刷新后可继续查询）

## 5. 面向「视频小说平台」的扩展建议（先写在代码结构里，不要求 MVP 实现）

建议把 I2V 看成一个可复用的 “shot generator”：

- **Project → Shot → Job**：一个镜头（shot）可以触发一次或多次生成任务（job），I2V 只是 job 的一种类型
- **复用能力**：同一套轮询/状态组件，复用到后续的 T2V、批量镜头生成、重试/变体等

路由示例：

- `/tools/i2v`：当前 MVP（保留并持续可用）
- `/projects`：视频小说项目列表（后续）
- `/projects/:id`：项目详情（章节/场景/镜头）（后续）
- `/assets`：素材库（后续）
