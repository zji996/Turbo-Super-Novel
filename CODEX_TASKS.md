# CODEX_TASKS: 懒验证 + 健康跟踪 + LLM Studio

> **目标**: 重构能力状态管理为"懒验证 + 健康跟踪"模式，新增 LLM Studio

---

## 核心设计：懒验证 + 健康跟踪

**原则**：
1. **不主动测试能力** — 避免 Worker 在多能力间切换
2. **任务成功/失败驱动状态** — 真实反映能力健康度
3. **初始用 Celery inspect 快速探测** — 仅检查 Worker 在线，不实际运行任务

```
┌─────────────────────────────────────────────────────────────────┐
│                      能力状态机                                  │
│                                                                  │
│  ┌──────────┐    首次探测      ┌──────────┐                     │
│  │ unknown  │ ──────────────▶ │ available│ ◀─── 任务成功       │
│  └──────────┘   Worker在线    └────┬─────┘                      │
│       │                             │                            │
│       │ Worker不在线                │ 任务失败                   │
│       ▼                             ▼                            │
│  ┌──────────┐                 ┌──────────┐                       │
│  │unavailable│ ◀───────────── │  error   │                      │
│  └──────────┘                 └──────────┘                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 第一部分：前端健康状态管理

### 1.1 状态 Hook

**新增**: `apps/web/src/hooks/useCapabilityHealth.ts`

```typescript
interface CapabilityHealth {
  status: 'unknown' | 'available' | 'unavailable';
  lastSuccess: number | null;  // timestamp
  lastError: string | null;
}

// Context + Hook 管理全局状态
export function useCapabilityHealth() {
  const [health, setHealth] = useState<Record<string, CapabilityHealth>>();
  
  // 初始化：调用 /v1/capabilities/status 快速探测
  useEffect(() => { initFromProbe(); }, []);
  
  // 任务完成时调用
  const reportSuccess = (cap: string) => { ... };
  const reportFailure = (cap: string, error: string) => { ... };
  
  return { health, reportSuccess, reportFailure };
}
```

### 1.2 各 Studio 集成

```typescript
// I2VStudio.tsx
const { reportSuccess, reportFailure } = useCapabilityHealth();

// 任务成功
onJobSuccess(() => reportSuccess('videogen'));

// 任务失败
onJobError((err) => reportFailure('videogen', err.message));
```

---

## 第二部分：后端简化

### 2.1 保留现有探测（仅检查 Worker 在线）

现有 `/v1/capabilities/status` 已实现 `_celery_worker_available()`，**无需改动**。

### 2.2 删除测试端点

原计划的 `/v1/capabilities/{cap}/test` **不再需要**，删除相关代码。

---

## 第三部分：LLM Studio

### 3.1 功能

- 简单对话界面
- 支持 system prompt
- 参数调节（temperature, max_tokens）
- 对话历史（sessionStorage）

### 3.2 API

使用现有 LLM 能力路由:

```python
@router.post("/v1/llm/chat")
async def llm_chat(messages: list[dict], temperature: float = 0.7):
    llm = get_capability_router("llm")
    return await llm.chat(messages, temperature=temperature)
```

### 3.3 前端

**新增**: `apps/web/src/pages/LLMStudio.tsx`

- 对话消息列表
- 输入框 + 发送按钮
- 参数侧边栏
- 清空对话按钮

---

## 实施清单

### Phase 1: 健康跟踪 Hook

- [x] `useCapabilityHealth.ts` 状态管理 Hook
- [x] `CapabilityHealthContext` Provider
- [x] 更新 `CapabilityStatusIndicator` 使用新 Hook
- [x] 删除 TTL 缓存逻辑

### Phase 2: Studio 集成

- [x] `I2VStudio` 任务完成回调 → reportSuccess/Failure
- [x] `TTSStudio` 任务完成回调
- [x] `ImageStudio` 任务完成回调

### Phase 3: LLM Studio

- [x] 后端: `/v1/llm/chat` 端点（如不存在）
- [x] 前端: `LLMStudio.tsx` 页面
- [x] 路由: 添加到 App.tsx

### Phase 4: 清理

- [x] 删除原计划的 `/v1/capabilities/{cap}/test` 端点代码（如已添加）
- [x] 删除后端 TTL 缓存变量

---

## 验证

```bash
# 启动服务
bash scripts/tsn_manage.sh start api worker web

# 前端访问
open http://localhost:5173

# 验证流程：
# 1. 打开 I2V Studio，提交任务
# 2. 任务成功后，侧边栏 videogen 状态变绿
# 3. 如任务失败，状态变红
```
