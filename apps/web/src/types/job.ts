/** Job types supported by the platform */
export type JobType = 'i2v' | 't2v' | 'tts' | 'imagegen';

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
    /** Target clip duration in seconds */
    duration_seconds: number;
}

/** TTS job parameters */
export interface TTSParams {
    provider: string;
    sample_rate: number;
    prompt_audio_id: string;
    prompt_text: string;
}

/** Image generation parameters */
export interface ImageGenParams {
    model?: string;
    width?: number;
    height?: number;
    steps?: number;
}

/** I2V job input data */
export type I2VInputs = Record<string, unknown> & {
    prompt: string;
    image_preview?: string; // Base64 or blob URL for local preview
};

interface BaseJob<TJobType extends JobType, TParams, TInputs extends Record<string, unknown>> {
    job_id: string;
    job_type: TJobType;
    status: CeleryStatus;
    db_status?: DBStatus;
    output_url?: string;
    error?: string;
    inputs: TInputs;
    params: TParams;
    created_at: number;
    updated_at?: number;
}

export type I2VJob = BaseJob<'i2v' | 't2v', I2VParams, I2VInputs>;
export type TTSJob = BaseJob<'tts', TTSParams, Record<string, unknown>>;
export type ImageGenJob = BaseJob<'imagegen', ImageGenParams, Record<string, unknown>>;

/** Complete job record */
export type Job = I2VJob | TTSJob | ImageGenJob;

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
    duration_seconds: 5,
};
