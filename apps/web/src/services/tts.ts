const API_BASE = import.meta.env.VITE_API_BASE_URL || '';

export interface PromptAudio {
    id: string;
    name: string;
    text: string;
    audio_url: string;
}

export interface CreateTTSJobRequest {
    text: string;
    provider?: string;
    prompt_text: string;
    prompt_audio_id: string;
    sample_rate?: number;
}

export async function listPromptAudios(): Promise<PromptAudio[]> {
    const resp = await fetch(`${API_BASE}/v1/tts/prompt-audios`);
    if (!resp.ok) throw new Error('Failed to list prompt audios');
    const data = await resp.json();
    return data.prompt_audios || [];
}

export async function uploadPromptAudio(
    name: string,
    text: string,
    audio: File
): Promise<PromptAudio> {
    const form = new FormData();
    form.append('name', name);
    form.append('text', text);
    form.append('audio', audio);

    const resp = await fetch(`${API_BASE}/v1/tts/prompt-audios`, {
        method: 'POST',
        body: form,
    });
    if (!resp.ok) throw new Error('Failed to upload prompt audio');
    return resp.json();
}

export async function createTTSJob(req: CreateTTSJobRequest) {
    const resp = await fetch(`${API_BASE}/v1/tts/jobs`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(req),
    });
    if (!resp.ok) throw new Error('Failed to create TTS job');
    return resp.json();
}

export async function getTTSJobStatus(jobId: string) {
    const resp = await fetch(`${API_BASE}/v1/tts/jobs/${jobId}`);
    if (!resp.ok) throw new Error('Failed to get TTS job status');
    return resp.json();
}

