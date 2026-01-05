import type { I2VParams, CreateJobResponse, VideoGenJob } from '../types';

import { apiGet, apiRequest } from './client';

/**
 * Create a new I2V (Image-to-Video) job
 */
export async function createI2VJob(
    image: File,
    prompt: string,
    params: I2VParams
): Promise<CreateJobResponse> {
    const formData = new FormData();
    formData.append('image', image);
    formData.append('prompt', prompt);
    formData.append('seed', String(params.seed));
    formData.append('num_steps', String(params.num_steps));
    formData.append('quantized', String(params.quantized));
    formData.append('duration_seconds', String(params.duration_seconds));
    return apiRequest<CreateJobResponse>('POST', '/v1/videogen/wan22-i2v/jobs', formData);
}

/**
 * Get the status of a job
 */
export async function getJobStatus(jobId: string): Promise<VideoGenJob> {
    return apiGet<VideoGenJob>(`/v1/videogen/jobs/${jobId}`);
}

/**
 * Get available models (for diagnostics)
 */
export async function getModels(): Promise<Record<string, { exists: boolean; path?: string }>> {
    return apiGet<Record<string, { exists: boolean; path?: string }>>('/v1/videogen/models');
}

/**
 * Health check
 */
export async function healthCheck(): Promise<{ status: string }> {
    return apiGet<{ status: string }>('/health');
}
