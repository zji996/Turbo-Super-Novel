import { useState, useEffect, useCallback } from 'react';
import type { I2VJob } from '../types';
import { DEFAULT_I2V_PARAMS } from '../types';

const STORAGE_KEY = 'i2v-jobs';

function normalizeJob(raw: unknown): I2VJob | null {
    if (!raw || typeof raw !== 'object') return null;
    const job = raw as Partial<I2VJob>;
    if (typeof job.job_id !== 'string') return null;

    const paramsRaw = (job as { params?: unknown }).params;
    const params = (paramsRaw && typeof paramsRaw === 'object')
        ? (paramsRaw as Partial<I2VJob['params']>)
        : {};

    const inputsRaw = (job as { inputs?: unknown }).inputs;
    const inputs = (inputsRaw && typeof inputsRaw === 'object') ? (inputsRaw as Partial<I2VJob['inputs']>) : {};

    return {
        job_id: job.job_id,
        job_type: job.job_type || 'i2v',
        status: job.status || 'PENDING',
        db_status: job.db_status,
        video_url: job.video_url ?? (job as unknown as { output_url?: string }).output_url,
        error: job.error,
        inputs: {
            prompt: typeof inputs.prompt === 'string' ? inputs.prompt : '',
            image_preview: typeof inputs.image_preview === 'string' ? inputs.image_preview : undefined,
        },
        params: { ...DEFAULT_I2V_PARAMS, ...(params as Partial<typeof DEFAULT_I2V_PARAMS>) },
        created_at: typeof job.created_at === 'number' ? job.created_at : Date.now(),
        updated_at: typeof job.updated_at === 'number' ? job.updated_at : undefined,
    };
}

function normalizeJobs(raw: unknown): I2VJob[] {
    if (!Array.isArray(raw)) return [];
    return raw.map(normalizeJob).filter((job): job is I2VJob => job !== null);
}

/**
 * Hook for persisting jobs in localStorage
 */
export function useJobStorage() {
    const [jobs, setJobs] = useState<I2VJob[]>(() => {
        try {
            const stored = localStorage.getItem(STORAGE_KEY);
            if (stored) {
                return normalizeJobs(JSON.parse(stored));
            }
        } catch (error) {
            console.error('Failed to load jobs from localStorage:', error);
        }
        return [];
    });

    // Persist to localStorage whenever jobs change
    useEffect(() => {
        try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(jobs));
        } catch (error) {
            console.error('Failed to save jobs to localStorage:', error);
        }
    }, [jobs]);

    const addJob = useCallback((job: I2VJob) => {
        setJobs((prev) => [job, ...prev]);
    }, []);

    const updateJob = useCallback((jobId: string, updates: Partial<I2VJob>) => {
        setJobs((prev) =>
            prev.map((job) =>
                job.job_id === jobId ? { ...job, ...updates } : job
            )
        );
    }, []);

    const removeJob = useCallback((jobId: string) => {
        setJobs((prev) => prev.filter((job) => job.job_id !== jobId));
    }, []);

    const clearJobs = useCallback(() => {
        setJobs([]);
    }, []);

    const getJob = useCallback((jobId: string) => {
        return jobs.find((job) => job.job_id === jobId);
    }, [jobs]);

    return {
        jobs,
        addJob,
        updateJob,
        removeJob,
        clearJobs,
        getJob,
    };
}
