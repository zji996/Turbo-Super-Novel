export type ImageGenStatus =
    | 'PENDING'
    | 'STARTED'
    | 'PROGRESS'
    | 'SUCCESS'
    | 'FAILURE'
    | 'REVOKED'
    | 'CANCELLED';

export interface ImageGenJob {
    job_id: string;
    status: ImageGenStatus;
    progress?: number;
    image_url?: string;
    error?: string;
    error_hint?: string;
    provider_type?: string;
}

export interface ImageGenParams {
    prompt: string;
    enhance_prompt?: boolean;
    width?: number;
    height?: number;
    num_inference_steps?: number;
    guidance_scale?: number;
    seed?: number;
    negative_prompt?: string;
}
