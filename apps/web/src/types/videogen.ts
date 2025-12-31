import type { CeleryStatus, DBStatus } from './job';

export interface VideoGenJob {
    job_id: string;
    status: CeleryStatus;
    progress?: number;
    video_url?: string;
    error?: string;
    provider_type?: string;
    db?: {
        status?: DBStatus;
        error?: string | null;
    };
}
