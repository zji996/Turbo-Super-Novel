import type {
    CreateSpeakerProfileRequest,
    CreateTTSJobWithProfileRequest,
    SpeakerProfile,
    TTSJob,
} from '../types';

import { apiDelete, apiGet, apiPost, apiRequest } from './client';

export async function listSpeakerProfiles(): Promise<SpeakerProfile[]> {
    const data = await apiGet<{ speaker_profiles?: SpeakerProfile[] }>(
        '/v1/tts/speaker-profiles'
    );
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
    return apiRequest<SpeakerProfile>('POST', '/v1/tts/speaker-profiles', form);
}

export async function deleteSpeakerProfile(profileId: string): Promise<void> {
    await apiDelete(`/v1/tts/speaker-profiles/${profileId}`);
}

export async function createTTSJobWithProfile(
    req: CreateTTSJobWithProfileRequest
): Promise<TTSJob> {
    return apiPost<TTSJob>('/v1/tts/jobs/with-profile', req);
}

export async function getTTSJobStatus(jobId: string): Promise<TTSJob> {
    return apiGet<TTSJob>(`/v1/tts/jobs/${jobId}`);
}
