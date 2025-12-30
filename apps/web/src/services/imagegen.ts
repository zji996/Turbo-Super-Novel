const API_BASE = import.meta.env.VITE_API_BASE_URL || '';

export interface ImageGenModel {
    id: string;
    name: string;
}

export async function listImageGenModels(): Promise<ImageGenModel[]> {
    const resp = await fetch(`${API_BASE}/v1/capabilities/imagegen/models`);
    if (!resp.ok) return [];
    const data = await resp.json();
    return data.models || [];
}

export async function createImageGenJob(
    prompt: string,
    params?: Record<string, unknown>
) {
    const resp = await fetch(`${API_BASE}/v1/imagegen/jobs`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, params: params || {} }),
    });
    if (!resp.ok) throw new Error('Failed to create imagegen job');
    return resp.json();
}

export async function getImageGenJobStatus(jobId: string) {
    const resp = await fetch(`${API_BASE}/v1/imagegen/jobs/${jobId}`);
    if (!resp.ok) throw new Error('Failed to get imagegen job status');
    return resp.json();
}

