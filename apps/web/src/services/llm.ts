const API_BASE = import.meta.env.VITE_API_BASE_URL || '';

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
    const resp = await fetch(`${API_BASE}/v1/llm/optimize-prompt`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
    });
    if (!resp.ok) throw new Error('Failed to optimize prompt');
    const data = await resp.json();
    return data.optimized;
}

export async function chatLLM(
    messages: ChatMessage[],
    opts: { model?: string; temperature?: number; max_tokens?: number } = {}
): Promise<ChatCompletionResponse> {
    const resp = await fetch(`${API_BASE}/v1/llm/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            messages,
            model: opts.model || undefined,
            temperature: typeof opts.temperature === 'number' ? opts.temperature : undefined,
            max_tokens: typeof opts.max_tokens === 'number' ? opts.max_tokens : undefined,
        }),
    });

    if (!resp.ok) {
        const error = await resp.text();
        throw new Error(`LLM chat failed: ${error}`);
    }

    return resp.json();
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
