import { apiPost } from './client';

export type ChatRole = 'system' | 'user' | 'assistant';

export interface ChatMessage {
    role: ChatRole;
    content: string;
}

export interface ChatCompletionResponse {
    choices?: Array<{
        message?: { content?: unknown };
    }>;
    [key: string]: unknown;
}

export async function optimizePrompt(text: string): Promise<string> {
    const data = await apiPost<{ optimized: string }>('/v1/llm/optimize-prompt', { text });
    return data.optimized;
}

export async function chatLLM(
    messages: ChatMessage[],
    opts: { model?: string; temperature?: number; max_tokens?: number } = {}
): Promise<ChatCompletionResponse> {
    return apiPost<ChatCompletionResponse>('/v1/llm/chat', {
        messages,
        model: opts.model || undefined,
        temperature: typeof opts.temperature === 'number' ? opts.temperature : undefined,
        max_tokens: typeof opts.max_tokens === 'number' ? opts.max_tokens : undefined,
    });
}

export function firstAssistantText(payload: ChatCompletionResponse): string | null {
    try {
        const choices = payload.choices || [];
        const content = choices[0]?.message?.content;
        if (content == null) return null;
        const text = String(content).trim();
        return text || null;
    } catch {
        return null;
    }
}
