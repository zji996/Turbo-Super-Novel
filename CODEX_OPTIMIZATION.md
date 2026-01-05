# Codex 优化任务

> 代码审查发现的优化点，按优先级排列

---

## P0: 统一 Job Polling Hook

**问题**: 三个 Hook 实现了几乎相同的轮询逻辑

- `apps/web/src/hooks/useJobPolling.ts` (143行)
- `apps/web/src/hooks/useImageGenJob.ts` (219行)  
- `apps/web/src/hooks/useTTSJob.ts` (132行)

**任务**:
1. 创建 `useGenericJobPolling<T>` 泛型 Hook，提供:
   - 可配置的轮询间隔和超时
   - 泛型状态类型支持
   - 自动清理
2. 重构 `useImageGenJob` 和 `useTTSJob` 使用新 Hook
3. 保留 `useJobPolling` 用于 I2V，后续迁移

**验证**: ImageStudio、TTSStudio、I2VStudio 功能正常

---

## P0: 统一 Job 状态类型

**问题**: 不同模块使用不同的状态类型

```typescript
// types/job.ts
type CeleryStatus = 'PENDING' | 'STARTED' | 'SUCCESS' | 'FAILURE';

// types/imagegen.ts  
type ImageGenStatus = 'PENDING' | 'STARTED' | 'PROGRESS' | 'SUCCESS' | 'FAILURE' | 'REVOKED' | 'CANCELLED';

// useTTSJob.ts - 硬编码
if (['SUCCEEDED', 'FAILED'].includes(status)) { ... }
```

**任务**:
1. 创建 `types/common.ts`:
   - 定义 `BaseJobStatus` 联合类型
   - 创建 `isTerminalStatus()`, `isPendingStatus()` 工具函数
2. 重构现有类型使用统一基础类型
3. 更新所有硬编码状态检查

---

## P1: 抽取共享 Studio 组件

**问题**: Studio 页面存在重复 UI 模式

| 模式 | 出现位置 |
|------|----------|
| 标题 + 描述头部 | 所有 Studio |
| 错误提示框 | 所有 Studio |
| 提交按钮 (带 loading) | 所有 Studio |

**任务**:
1. 创建 `components/studio/StudioHeader.tsx`
2. 创建 `components/studio/ErrorAlert.tsx`
3. 创建 `components/studio/SubmitButton.tsx`
4. 重构各 Studio 使用共享组件

---

## P1: LLMStudio 状态提取

**问题**: `apps/web/src/pages/LLMStudio.tsx` 状态管理分散

**任务**:
1. 创建 `hooks/useLLMSession.ts`:
   - 封装 sessionStorage 读写
   - 管理消息历史和参数持久化
2. 添加消息自动滚动到底部
3. (可选) 添加消息编辑/删除功能

---

## P2: 后端 capabilities.py 优化

**问题**: `apps/api/routes/capabilities.py` 重复代码

```python
# 这种模式重复多次
if getattr(cap, "provider_type", "local") == "remote" and hasattr(cap, "request_json"):
    return await cap.request_json(...)
```

**任务**:
1. 创建辅助函数处理 remote vs local 分支
2. 简化重复的 try/except + ImportError 处理

---

## P2: 类型导出整理

**问题**: `types/index.ts` 导出不完整

**任务**:
1. 更新 `types/index.ts` 导出所有类型文件
2. 确保所有导入使用 barrel import: `from '../types'`

---

## P3: 补充 Hook 测试

**任务**: 使用 vitest 添加以下测试:
- `useJobPolling.test.ts`
- `useImageGenJob.test.ts`  
- `useTTSJob.test.ts`

---

## 执行顺序建议

1. **P0 任务** 先做，影响核心功能
2. **P1 任务** 提升代码质量
3. **P2-P3** 按需执行
