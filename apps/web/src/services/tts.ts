import type {
    CreateSpeakerProfileRequest,
    CreateTTSJobWithProfileRequest,
    SpeakerProfile,
    TTSJob,
} from '../types';

const API_BASE = import.meta.env.VITE_API_BASE_URL || '';

export async function listSpeakerProfiles(): Promise<SpeakerProfile[]> {
    const resp = await fetch(`${API_BASE}/v1/tts/speaker-profiles`);
    if (!resp.ok) throw new Error('Failed to list speaker profiles');
    const data = await resp.json();
    return data.speaker_profiles || [];
}

export async function createSpeakerProfile(
    req: CreateSpeakerProfileRequest
): Promise<SpeakerProfile> {
    const form = new FormData();
    form.append('name', req.name);
    form.append('prompt_text', req.prompt_text);
    form.append('prompt_audio', req.prompt_audio);

    if (req.description) form.append('description', req.description);
    if (req.provider) form.append('provider', req.provider);
    if (req.sample_rate != null) form.append('sample_rate', String(req.sample_rate));
    if (req.is_default != null) form.append('is_default', String(req.is_default));
    if (req.config) form.append('config', JSON.stringify(req.config));

    const resp = await fetch(`${API_BASE}/v1/tts/speaker-profiles`, {
        method: 'POST',
        body: form,
    });
    if (!resp.ok) throw new Error('Failed to create speaker profile');
    return resp.json();
}

export async function deleteSpeakerProfile(profileId: string): Promise<void> {
    const resp = await fetch(`${API_BASE}/v1/tts/speaker-profiles/${profileId}`, {
        method: 'DELETE',
    });
    if (!resp.ok) throw new Error('Failed to delete speaker profile');
}

export async function createTTSJobWithProfile(
    req: CreateTTSJobWithProfileRequest
): Promise<TTSJob> {
    const resp = await fetch(`${API_BASE}/v1/tts/jobs/with-profile`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(req),
    });
    if (!resp.ok) throw new Error('Failed to create TTS job');
    return resp.json();
}

export async function getTTSJobStatus(jobId: string): Promise<TTSJob> {
    const resp = await fetch(`${API_BASE}/v1/tts/jobs/${jobId}`);
    if (!resp.ok) throw new Error('Failed to get TTS job status');
    return resp.json();
}
