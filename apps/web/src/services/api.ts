import type { I2VParams, CreateJobResponse, VideoGenJob } from '../types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

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

    const response = await fetch(`${API_BASE_URL}/v1/videogen/wan22-i2v/jobs`, {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        const error = await response.text();
        throw new Error(`Failed to create job: ${error}`);
    }

    return response.json();
}

/**
 * Get the status of a job
 */
export async function getJobStatus(jobId: string): Promise<VideoGenJob> {
    const response = await fetch(`${API_BASE_URL}/v1/videogen/jobs/${jobId}`);

    if (!response.ok) {
        const error = await response.text();
        throw new Error(`Failed to get job status: ${error}`);
    }

    return response.json();
}

/**
 * Get available models (for diagnostics)
 */
export async function getModels(): Promise<Record<string, { exists: boolean; path?: string }>> {
    const response = await fetch(`${API_BASE_URL}/v1/videogen/models`);

    if (!response.ok) {
        throw new Error('Failed to get models');
    }

    return response.json();
}

/**
 * Health check
 */
export async function healthCheck(): Promise<{ status: string }> {
    const response = await fetch(`${API_BASE_URL}/health`);

    if (!response.ok) {
        throw new Error('Health check failed');
    }

    return response.json();
}
