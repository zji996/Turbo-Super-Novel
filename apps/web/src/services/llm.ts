const API_BASE = import.meta.env.VITE_API_BASE_URL || '';

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

