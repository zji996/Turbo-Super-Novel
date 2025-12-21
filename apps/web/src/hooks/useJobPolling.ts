import { useEffect, useRef, useCallback, useState } from 'react';
import type { Job, CeleryStatus, DBStatus } from '../types';
import { getJobStatus } from '../services/api';

interface UseJobPollingOptions {
    /** Callback when job status updates */
    onStatusUpdate?: (job: Job) => void;
    /** Callback when job completes (SUCCESS or FAILURE) */
    onComplete?: (job: Job) => void;
    /** Callback on error */
    onError?: (error: Error, job: Job) => void;
}

/** Polling intervals in ms */
const POLLING_INTERVALS = {
    RUNNING: 2000,   // 2s when running
    PENDING: 5000,   // 5s when pending
    IDLE: 10000,     // 10s when no change
};

/** Maximum polling duration (30 minutes) */
const MAX_POLLING_DURATION = 30 * 60 * 1000;

/**
 * Hook for polling job status with adaptive intervals
 */
export function useJobPolling(
    jobs: Job[],
    updateJob: (jobId: string, updates: Partial<Job>) => void,
    options: UseJobPollingOptions = {}
) {
    const { onStatusUpdate, onComplete, onError } = options;
    const pollingRef = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map());
    const startTimeRef = useRef<Map<string, number>>(new Map());
    const [isPolling, setIsPolling] = useState(false);

    const pollJob = useCallback(async (job: Job) => {
        const startTime = startTimeRef.current.get(job.job_id) || Date.now();

        // Check timeout
        if (Date.now() - startTime > MAX_POLLING_DURATION) {
            console.warn(`Job ${job.job_id} polling timeout`);
            updateJob(job.job_id, {
                status: 'FAILURE' as CeleryStatus,
                error: 'Polling timeout - job may still be running in the background'
            });
            return;
        }

        try {
            const response = await getJobStatus(job.job_id);

            const updates: Partial<Job> = {
                status: response.status,
                db_status: response.db?.status as DBStatus | undefined,
                output_url: response.output_url,
                error: response.error,
                updated_at: Date.now(),
            };

            updateJob(job.job_id, updates);

            const updatedJob = { ...job, ...updates };
            onStatusUpdate?.(updatedJob);

            // Check if job is complete
            if (response.status === 'SUCCESS' || response.status === 'FAILURE') {
                onComplete?.(updatedJob);
                stopPolling(job.job_id);
                return;
            }

            // Schedule next poll based on status
            const interval = response.status === 'STARTED'
                ? POLLING_INTERVALS.RUNNING
                : POLLING_INTERVALS.PENDING;

            scheduleNextPoll(job.job_id, updatedJob, interval);
        } catch (error) {
            console.error(`Error polling job ${job.job_id}:`, error);
            onError?.(error as Error, job);

            // Continue polling on error (might be temporary network issue)
            scheduleNextPoll(job.job_id, job, POLLING_INTERVALS.IDLE);
        }
    }, [updateJob, onStatusUpdate, onComplete, onError]);

    const scheduleNextPoll = useCallback((jobId: string, job: Job, interval: number) => {
        const existing = pollingRef.current.get(jobId);
        if (existing) {
            clearTimeout(existing);
        }

        const timeout = setTimeout(() => pollJob(job), interval);
        pollingRef.current.set(jobId, timeout);
        setIsPolling(pollingRef.current.size > 0);
    }, [pollJob]);

    const stopPolling = useCallback((jobId: string) => {
        const timeout = pollingRef.current.get(jobId);
        if (timeout) {
            clearTimeout(timeout);
            pollingRef.current.delete(jobId);
        }
        startTimeRef.current.delete(jobId);
        setIsPolling(pollingRef.current.size > 0);
    }, []);

    const startPolling = useCallback((job: Job) => {
        if (!startTimeRef.current.has(job.job_id)) {
            startTimeRef.current.set(job.job_id, Date.now());
        }
        // Start immediately
        pollJob(job);
    }, [pollJob]);

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            pollingRef.current.forEach((timeout) => clearTimeout(timeout));
            pollingRef.current.clear();
        };
    }, []);

    // Auto-start polling for incomplete jobs
    useEffect(() => {
        jobs.forEach((job) => {
            const isIncomplete = job.status !== 'SUCCESS' && job.status !== 'FAILURE';
            const isNotPolling = !pollingRef.current.has(job.job_id);

            if (isIncomplete && isNotPolling) {
                startPolling(job);
            }
        });
    }, [jobs, startPolling]);

    return {
        startPolling,
        stopPolling,
        isPolling,
    };
}
