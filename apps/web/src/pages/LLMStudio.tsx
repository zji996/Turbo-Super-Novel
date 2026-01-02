import { useCallback, useEffect, useMemo, useState } from 'react';
import type { ChatMessage } from '../services/llm';
import { chatLLM, firstAssistantText } from '../services/llm';
import { useCapabilityHealth } from '../hooks/useCapabilityHealth';

const STORAGE_KEY = 'tsn_llm_studio_v1';

interface StoredState {
    systemPrompt: string;
    model: string;
    temperature: number;
    maxTokens: number;
    messages: ChatMessage[];
}

function loadState(): StoredState | null {
    try {
        const raw = sessionStorage.getItem(STORAGE_KEY);
        if (!raw) return null;
        return JSON.parse(raw) as StoredState;
    } catch {
        return null;
    }
}

export function LLMStudio() {
    const { reportFailure, reportSuccess } = useCapabilityHealth();

    const stored = useMemo(() => loadState(), []);

    const [systemPrompt, setSystemPrompt] = useState(stored?.systemPrompt || '');
    const [model, setModel] = useState(stored?.model || '');
    const [temperature, setTemperature] = useState(stored?.temperature ?? 0.7);
    const [maxTokens, setMaxTokens] = useState(stored?.maxTokens ?? 1024);
    const [messages, setMessages] = useState<ChatMessage[]>(stored?.messages || []);

    const [input, setInput] = useState('');
    const [isSending, setIsSending] = useState(false);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const next: StoredState = { systemPrompt, model, temperature, maxTokens, messages };
        try {
            sessionStorage.setItem(STORAGE_KEY, JSON.stringify(next));
        } catch {
            // ignore
        }
    }, [systemPrompt, model, temperature, maxTokens, messages]);

    const clear = useCallback(() => {
        setMessages([]);
        setError(null);
        setInput('');
    }, []);

    const send = useCallback(async () => {
        const userText = input.trim();
        if (!userText || isSending) return;

        setIsSending(true);
        setError(null);

        const nextMessages: ChatMessage[] = [...messages, { role: 'user', content: userText }];
        setMessages(nextMessages);
        setInput('');

        const payloadMessages: ChatMessage[] = [
            ...(systemPrompt.trim() ? [{ role: 'system' as const, content: systemPrompt.trim() }] : []),
            ...nextMessages,
        ];

        try {
            const resp = await chatLLM(payloadMessages, {
                model: model.trim() || undefined,
                temperature,
                max_tokens: maxTokens,
            });
            const text = firstAssistantText(resp) || '(empty response)';
            setMessages((prev) => [...prev, { role: 'assistant', content: text }]);
            reportSuccess('llm');
        } catch (e) {
            const message = e instanceof Error ? e.message : String(e);
            setError(message);
            reportFailure('llm', message);
        } finally {
            setIsSending(false);
        }
    }, [input, isSending, messages, systemPrompt, model, temperature, maxTokens, reportFailure, reportSuccess]);

    return (
        <div className="animate-fade-in">
            <div className="flex items-center justify-between mb-6">
                <div>
                    <h1 className="text-3xl font-bold">💬 LLM Studio</h1>
                    <p className="text-[var(--color-text-secondary)] mt-1">简单对话界面 · 支持 system prompt 与参数调节</p>
                </div>
                <button onClick={clear} className="btn-secondary">
                    清空对话
                </button>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div className="lg:col-span-2 space-y-4">
                    <div className="card">
                        <div className="h-[520px] overflow-y-auto space-y-3">
                            {messages.length === 0 ? (
                                <div className="text-sm text-[var(--color-text-muted)]">暂无对话，输入内容开始。</div>
                            ) : (
                                messages.map((m, idx) => (
                                    <div
                                        key={idx}
                                        className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}
                                    >
                                        <div
                                            className={`max-w-[85%] rounded-lg px-3 py-2 text-sm border ${m.role === 'user'
                                                    ? 'bg-[var(--color-accent-primary)]/10 border-[var(--color-accent-primary)]/30'
                                                    : 'bg-[var(--color-bg-tertiary)] border-[var(--color-border)]'
                                                }`}
                                        >
                                            <div className="text-xs text-[var(--color-text-muted)] mb-1">
                                                {m.role}
                                            </div>
                                            <div className="whitespace-pre-wrap text-[var(--color-text-primary)]">
                                                {m.content}
                                            </div>
                                        </div>
                                    </div>
                                ))
                            )}
                        </div>
                    </div>

                    {error && (
                        <div className="p-4 rounded-lg bg-[var(--color-error)]/10 border border-[var(--color-error)]/20">
                            <p className="text-sm text-[var(--color-error)]">{error}</p>
                        </div>
                    )}

                    <div className="card">
                        <div className="flex gap-3">
                            <textarea
                                value={input}
                                onChange={(e) => setInput(e.target.value)}
                                placeholder="输入消息..."
                                className="flex-1 h-20 p-3 rounded-lg bg-[var(--color-bg-tertiary)] border border-[var(--color-border)] resize-none"
                            />
                            <button
                                onClick={send}
                                disabled={!input.trim() || isSending}
                                className="btn-primary px-5"
                            >
                                {isSending ? '发送中...' : '发送'}
                            </button>
                        </div>
                        <div className="text-xs text-[var(--color-text-muted)] mt-2">
                            {input.length} 字符
                        </div>
                    </div>
                </div>

                <div className="space-y-4">
                    <div className="card">
                        <h3 className="font-semibold mb-3">参数</h3>

                        <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Model (optional)</label>
                        <input
                            value={model}
                            onChange={(e) => setModel(e.target.value)}
                            placeholder="e.g. deepseek-chat"
                            className="w-full mb-4 p-2 rounded-lg bg-[var(--color-bg-tertiary)] border border-[var(--color-border)]"
                        />

                        <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Temperature</label>
                        <input
                            type="number"
                            min={0}
                            max={2}
                            step={0.1}
                            value={temperature}
                            onChange={(e) => setTemperature(Number(e.target.value))}
                            className="w-full mb-4 p-2 rounded-lg bg-[var(--color-bg-tertiary)] border border-[var(--color-border)]"
                        />

                        <label className="block text-sm text-[var(--color-text-secondary)] mb-1">Max tokens</label>
                        <input
                            type="number"
                            min={1}
                            max={32768}
                            step={1}
                            value={maxTokens}
                            onChange={(e) => setMaxTokens(Number(e.target.value))}
                            className="w-full mb-4 p-2 rounded-lg bg-[var(--color-bg-tertiary)] border border-[var(--color-border)]"
                        />
                    </div>

                    <div className="card">
                        <h3 className="font-semibold mb-3">System Prompt</h3>
                        <textarea
                            value={systemPrompt}
                            onChange={(e) => setSystemPrompt(e.target.value)}
                            placeholder="(optional) 在这里输入 system prompt..."
                            className="w-full h-48 p-3 rounded-lg bg-[var(--color-bg-tertiary)] border border-[var(--color-border)] resize-none"
                        />
                    </div>
                </div>
            </div>
        </div>
    );
}

