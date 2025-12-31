/**
 * useImageGenJob Hook
 *
 * Provides a simple interface for managing image generation jobs,
 * including submission, polling, and cancellation.
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import {
    createImageGenJob,
    getImageGenJobStatus,
    cancelImageGenJob,
    isJobTerminal,
    isJobPending,
} from '../services/imagegen';
import type { ImageGenJob, ImageGenParams, ImageGenStatus } from '../types';

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

export interface UseImageGenJobOptions {
    /** Polling interval in ms (default: 2000) */
    pollingInterval?: number;
    /** Maximum polling duration in ms (default: 10 minutes) */
    maxPollingDuration?: number;
    /** Callback when job completes successfully */
    onSuccess?: (job: ImageGenJob) => void;
    /** Callback when job fails */
    onError?: (error: Error, job?: ImageGenJob) => void;
    /** Callback on status update */
    onStatusUpdate?: (job: ImageGenJob) => void;
}

export interface UseImageGenJobResult {
    /** Current job state */
    job: ImageGenJob | null;
    /** Whether a job is being submitted */
    isSubmitting: boolean;
    /** Whether the job is being polled (in progress) */
    isPolling: boolean;
    /** Submit a new image generation job */
    submit: (
        prompt: string,
        params?: Omit<ImageGenParams, 'prompt'>
    ) => Promise<ImageGenJob | null>;
    /** Cancel the current job */
    cancel: () => Promise<void>;
    /** Clear the current job state */
    clear: () => void;
    /** Current status */
    status: ImageGenStatus | null;
    /** Progress percentage (0-100) */
    progress: number;
    /** Error message if any */
    error: string | null;
    /** Generated image URL */
    imageUrl: string | null;
}

// ─────────────────────────────────────────────────────────────────────────────
// Hook
// ─────────────────────────────────────────────────────────────────────────────

export function useImageGenJob(
    options: UseImageGenJobOptions = {}
): UseImageGenJobResult {
    const {
        pollingInterval = 2000,
        maxPollingDuration = 10 * 60 * 1000, // 10 minutes
        onSuccess,
        onError,
        onStatusUpdate,
    } = options;

    const [job, setJob] = useState<ImageGenJob | null>(null);
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [isPolling, setIsPolling] = useState(false);

    const pollingRef = useRef<ReturnType<typeof setTimeout> | null>(null);
    const startTimeRef = useRef<number | null>(null);

    // Cleanup polling on unmount
    useEffect(() => {
        return () => {
            if (pollingRef.current) {
                clearTimeout(pollingRef.current);
            }
        };
    }, []);

    const stopPolling = useCallback(() => {
        if (pollingRef.current) {
            clearTimeout(pollingRef.current);
            pollingRef.current = null;
        }
        startTimeRef.current = null;
        setIsPolling(false);
    }, []);

    const pollJobStatus = useCallback(
        async (jobId: string) => {
            // Check timeout
            if (
                startTimeRef.current &&
                Date.now() - startTimeRef.current > maxPollingDuration
            ) {
                console.warn(`ImageGen job ${jobId} polling timeout`);
                const timeoutJob: ImageGenJob = {
                    job_id: jobId,
                    status: 'FAILURE',
                    error: '轮询超时 - 任务可能仍在后台运行',
                };
                setJob(timeoutJob);
                stopPolling();
                onError?.(new Error('Polling timeout'), timeoutJob);
                return;
            }

            try {
                const updatedJob = await getImageGenJobStatus(jobId);
                setJob(updatedJob);
                onStatusUpdate?.(updatedJob);

                if (isJobTerminal(updatedJob.status)) {
                    stopPolling();
                    if (updatedJob.status === 'SUCCESS') {
                        onSuccess?.(updatedJob);
                    } else if (updatedJob.status === 'FAILURE') {
                        onError?.(new Error(updatedJob.error || 'Job failed'), updatedJob);
                    }
                } else {
                    // Schedule next poll
                    pollingRef.current = setTimeout(
                        () => pollJobStatus(jobId),
                        pollingInterval
                    );
                }
            } catch (error) {
                console.error(`Error polling imagegen job ${jobId}:`, error);
                // Continue polling on error (might be temporary network issue)
                pollingRef.current = setTimeout(
                    () => pollJobStatus(jobId),
                    pollingInterval * 2
                );
            }
        },
        [pollingInterval, maxPollingDuration, stopPolling, onSuccess, onError, onStatusUpdate]
    );

    const startPolling = useCallback(
        (jobId: string) => {
            stopPolling();
            startTimeRef.current = Date.now();
            setIsPolling(true);
            pollJobStatus(jobId);
        },
        [stopPolling, pollJobStatus]
    );

    const submit = useCallback(
        async (prompt: string, params?: Omit<ImageGenParams, 'prompt'>): Promise<ImageGenJob | null> => {
            setIsSubmitting(true);

            try {
                const newJob = await createImageGenJob({ prompt, ...(params || {}) });
                setJob(newJob);

                // Start polling if job is pending
                if (isJobPending(newJob.status)) {
                    startPolling(newJob.job_id);
                }

                return newJob;
            } catch (error) {
                console.error('Failed to submit imagegen job:', error);
                onError?.(error as Error);
                return null;
            } finally {
                setIsSubmitting(false);
            }
        },
        [startPolling, onError]
    );

    const cancel = useCallback(async () => {
        if (!job) return;

        try {
            await cancelImageGenJob(job.job_id);
            stopPolling();
            setJob((prev) =>
                prev ? { ...prev, status: 'CANCELLED' as ImageGenStatus } : null
            );
        } catch (error) {
            console.error('Failed to cancel imagegen job:', error);
            onError?.(error as Error);
        }
    }, [job, stopPolling, onError]);

    const clear = useCallback(() => {
        stopPolling();
        setJob(null);
    }, [stopPolling]);

    return {
        job,
        isSubmitting,
        isPolling,
        submit,
        cancel,
        clear,
        status: job?.status ?? null,
        progress: job?.progress ?? 0,
        error: job?.error ?? job?.error_hint ?? null,
        imageUrl: job?.image_url ?? null,
    };
}
