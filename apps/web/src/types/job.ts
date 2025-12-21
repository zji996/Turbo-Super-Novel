/** Job types supported by the platform */
export type JobType = 'i2v' | 't2v';

/** Celery task status */
export type CeleryStatus = 'PENDING' | 'STARTED' | 'SUCCESS' | 'FAILURE';

/** Database job status (more detailed) */
export type DBStatus =
    | 'CREATED'
    | 'SUBMITTED'
    | 'STARTED'
    | 'DOWNLOADED'
    | 'RUNNING'
    | 'UPLOADED'
    | 'SUCCEEDED'
    | 'FAILED';

/** I2V job parameters */
export interface I2VParams {
    seed: number;
    num_steps: number;
    quantized: boolean;
}

/** Job input data */
export interface JobInputs {
    prompt: string;
    image_preview?: string; // Base64 or blob URL for local preview
}

/** Complete job record */
export interface Job {
    job_id: string;
    job_type: JobType;
    status: CeleryStatus;
    db_status?: DBStatus;
    output_url?: string;
    error?: string;
    inputs: JobInputs;
    params: I2VParams;
    created_at: number;
    updated_at?: number;
}

/** API response for job creation */
export interface CreateJobResponse {
    job_id: string;
    status: string;
    input?: {
        bucket?: string;
        key?: string;
    };
    output?: {
        bucket?: string;
        key?: string;
    };
    db?: {
        persisted?: boolean;
        error?: string;
    };
}

/** API response for job status query */
export interface JobStatusResponse {
    status: CeleryStatus;
    db?: {
        status?: DBStatus;
    };
    result?: unknown;
    output_url?: string;
    error?: string;
}

/** Default parameters for I2V */
export const DEFAULT_I2V_PARAMS: I2VParams = {
    seed: 0,
    num_steps: 4,
    quantized: true,
};
