/**
 * Image Generation Service
 *
 * Provides functions for interacting with the image generation API,
 * which proxies requests to the remote Z-Image API.
 */

import type { ImageGenJob, ImageGenParams, ImageGenStatus } from '../types';

const API_BASE = import.meta.env.VITE_API_BASE_URL || '';

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

export interface ImageGenHistoryItem {
    task_id: string;
    status: string;
    created_at: string;
    prompt: string;
    height: number;
    width: number;
    image_url?: string;
    seed?: number;
    batch_size?: number;
    success_count?: number;
    failed_count?: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// API Functions
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Create a new image generation job.
 */
export async function createImageGenJob(
    params: ImageGenParams
): Promise<ImageGenJob> {
    const body: Record<string, unknown> = { prompt: params.prompt };
    if (params.enhance_prompt !== undefined) body.enhance_prompt = params.enhance_prompt;
    if (params.width !== undefined) body.width = params.width;
    if (params.height !== undefined) body.height = params.height;
    if (params.num_inference_steps !== undefined)
        body.num_inference_steps = params.num_inference_steps;
    if (params.guidance_scale !== undefined) body.guidance_scale = params.guidance_scale;
    if (params.seed !== undefined) body.seed = params.seed;
    if (params.negative_prompt !== undefined) body.negative_prompt = params.negative_prompt;

    const resp = await fetch(`${API_BASE}/v1/imagegen/jobs`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });

    if (!resp.ok) {
        const error = await resp.text();
        throw new Error(`Failed to create imagegen job: ${error}`);
    }

    return resp.json();
}

/**
 * Get the status of an image generation job.
 */
export async function getImageGenJobStatus(jobId: string): Promise<ImageGenJob> {
    const resp = await fetch(`${API_BASE}/v1/imagegen/jobs/${jobId}`);

    if (!resp.ok) {
        const error = await resp.text();
        throw new Error(`Failed to get imagegen job status: ${error}`);
    }

    return resp.json();
}

/**
 * Cancel an image generation job.
 */
export async function cancelImageGenJob(
    jobId: string
): Promise<{ job_id: string; status: string; message: string }> {
    const resp = await fetch(`${API_BASE}/v1/imagegen/jobs/${jobId}/cancel`, {
        method: 'POST',
    });

    if (!resp.ok) {
        const error = await resp.text();
        throw new Error(`Failed to cancel imagegen job: ${error}`);
    }

    return resp.json();
}

/**
 * Get image generation history from the remote Z-Image API.
 */
export async function getImageGenHistory(
    limit = 20,
    offset = 0
): Promise<ImageGenHistoryItem[]> {
    const resp = await fetch(
        `${API_BASE}/v1/imagegen/history?limit=${limit}&offset=${offset}`
    );

    if (!resp.ok) {
        const error = await resp.text();
        throw new Error(`Failed to get imagegen history: ${error}`);
    }

    return resp.json();
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility Functions
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Check if a job is in a terminal state (completed or failed).
 */
export function isJobTerminal(status: ImageGenStatus): boolean {
    return ['SUCCESS', 'FAILURE', 'REVOKED', 'CANCELLED'].includes(status);
}

/**
 * Check if a job is in a pending/running state.
 */
export function isJobPending(status: ImageGenStatus): boolean {
    return ['PENDING', 'STARTED', 'PROGRESS'].includes(status);
}

/**
 * Get a human-readable status message.
 */
export function getStatusMessage(job: ImageGenJob): string {
    switch (job.status) {
        case 'PENDING':
            return '等待中...';
        case 'STARTED':
            return '处理中...';
        case 'PROGRESS':
            return `生成中 ${job.progress ?? 0}%`;
        case 'SUCCESS':
            return '生成完成';
        case 'FAILURE':
            return job.error_hint || job.error || '生成失败';
        case 'REVOKED':
        case 'CANCELLED':
            return '已取消';
        default:
            return '未知状态';
    }
}
