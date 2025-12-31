export interface TTSJob {
    job_id: string;
    status: string;
    output_url?: string;
    error?: string;
    audio_duration_seconds?: number;
    provider_type?: string;
    db?: {
        status?: string;
        error?: string | null;
    };
}

export interface SpeakerProfile {
    id: string;
    name: string;
    description?: string | null;
    provider: string;
    sample_rate: number;
    prompt_text: string;
    prompt_audio_url?: string | null;
    is_default: boolean;
    created_at: string;
    updated_at: string;
}

export interface CreateSpeakerProfileRequest {
    name: string;
    description?: string;
    provider?: string;
    sample_rate?: number;
    prompt_text: string;
    prompt_audio: File;
    config?: Record<string, unknown>;
    is_default?: boolean;
}

export interface CreateTTSJobWithProfileRequest {
    text: string;
    profile_id: string;
}
