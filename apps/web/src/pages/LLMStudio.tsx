import { useCallback, useEffect, useRef, useState } from 'react';
import type { ChatMessage } from '../services/llm';
import { chatLLM, firstAssistantText } from '../services/llm';
import { useCapabilityHealth } from '../hooks/useCapabilityHealth';
import { ErrorAlert, StudioHeader, SubmitButton } from '../components';
import { useLLMSession } from '../hooks/useLLMSession';

export function LLMStudio() {
    const { reportFailure, reportSuccess } = useCapabilityHealth();

    const {
        systemPrompt,
        setSystemPrompt,
        model,
        setModel,
        temperature,
        setTemperature,
        maxTokens,
        setMaxTokens,
        messages,
        setMessages,
        clearMessages,
    } = useLLMSession();

    const [input, setInput] = useState('');
    const [isSending, setIsSending] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const messagesContainerRef = useRef<HTMLDivElement | null>(null);
    const messagesEndRef = useRef<HTMLDivElement | null>(null);
    const shouldAutoScrollRef = useRef(true);

    useEffect(() => {
        const el = messagesContainerRef.current;
        if (!el) return;

        const onScroll = () => {
            const distanceToBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
            shouldAutoScrollRef.current = distanceToBottom < 80;
        };

        el.addEventListener('scroll', onScroll);
        onScroll();

        return () => {
            el.removeEventListener('scroll', onScroll);
        };
    }, []);

    useEffect(() => {
        if (!shouldAutoScrollRef.current) return;
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages.length]);

    const clear = useCallback(() => {
        clearMessages();
        setError(null);
        setInput('');
    }, [clearMessages]);

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
    }, [input, isSending, messages, systemPrompt, model, temperature, maxTokens, reportFailure, reportSuccess, setMessages]);

    return (
        <div className="animate-fade-in">
            <StudioHeader
                title="💬 LLM Studio"
                description="简单对话界面 · 支持 system prompt 与参数调节"
                action={
                    <button onClick={clear} className="btn-secondary">
                        清空对话
                    </button>
                }
                className="mb-6"
            />

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div className="lg:col-span-2 space-y-4">
                    <div className="card">
                        <div
                            ref={messagesContainerRef}
                            className="h-[520px] overflow-y-auto space-y-3"
                        >
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
                            <div ref={messagesEndRef} />
                        </div>
                    </div>

                    <ErrorAlert message={error} />

                    <div className="card">
                        <div className="flex gap-3">
                            <textarea
                                value={input}
                                onChange={(e) => setInput(e.target.value)}
                                placeholder="输入消息..."
                                className="flex-1 h-20 p-3 rounded-lg bg-[var(--color-bg-tertiary)] border border-[var(--color-border)] resize-none"
                            />
                            <SubmitButton
                                onClick={send}
                                disabled={!input.trim() || isSending}
                                isLoading={isSending}
                                loadingText="发送中..."
                                className="px-5"
                            >
                                发送
                            </SubmitButton>
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
