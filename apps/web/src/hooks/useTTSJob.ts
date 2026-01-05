import { useCallback, useEffect, useState } from 'react';
import { createTTSJobWithProfile, getTTSJobStatus } from '../services/tts';
import type { BaseJobStatus, TTSJob } from '../types';
import { isTerminalStatus } from '../types';
import { useGenericJobPolling } from './useGenericJobPolling';

export interface TTSJobRequest {
    text: string;
    profile_id: string;
}

export interface TTSJobState {
    jobId: string | null;
    status: BaseJobStatus | null;
    outputUrl: string | null;
    error: string | null;
    isSubmitting: boolean;
    isPolling: boolean;
}

export function useTTSJob(
    pollingInterval: number = 2000,
    onSuccess?: (state: TTSJobState) => void,
    onError?: (error: Error) => void
) {
    const [state, setState] = useState<TTSJobState>({
        jobId: null,
        status: null,
        outputUrl: null,
        error: null,
        isSubmitting: false,
        isPolling: false,
    });

    const { startPolling, stopPolling, isPolling } = useGenericJobPolling<TTSJob>({
        pollingInterval,
        getStatus: getTTSJobStatus,
        getJobStatus: (job) => job.status ?? job.db?.status ?? 'PENDING',
        isTerminal: (status) => isTerminalStatus(status),
        onStatusUpdate: (job) => {
            const nextStatus = (job.status ?? job.db?.status ?? 'PENDING') as BaseJobStatus;
            const outputUrl = job.output_url || null;
            setState((prev) => ({
                ...prev,
                status: nextStatus,
                outputUrl,
            }));
        },
        onComplete: (job) => {
            const finalStatus = (job.status ?? job.db?.status ?? 'PENDING') as BaseJobStatus;
            const outputUrl = job.output_url || null;

            const finalState: TTSJobState = {
                jobId: job.job_id,
                status: finalStatus,
                outputUrl,
                error: job.error || null,
                isSubmitting: false,
                isPolling: false,
            };
            setState(finalState);

            if (finalStatus === 'SUCCESS' || finalStatus === 'SUCCEEDED') {
                onSuccess?.(finalState);
            } else if (finalStatus === 'FAILURE' || finalStatus === 'FAILED') {
                onError?.(new Error(job.error || 'TTS job failed'));
            }
        },
        onError: (error, jobId) => {
            console.error(`Polling error (job ${jobId}):`, error);
        },
    });

    const submit = useCallback(
        async (request: TTSJobRequest) => {
            setState((prev) => ({ ...prev, isSubmitting: true, error: null }));
            try {
                const resp = await createTTSJobWithProfile(request);
                setState((prev) => ({
                    ...prev,
                    jobId: resp.job_id,
                    status: 'SUBMITTED',
                    outputUrl: null,
                    error: null,
                    isSubmitting: false,
                    isPolling: true,
                }));
                startPolling(resp.job_id);
            } catch (e) {
                const error = e instanceof Error ? e : new Error(String(e));
                setState((prev) => ({
                    ...prev,
                    isSubmitting: false,
                    error: error.message,
                }));
                onError?.(error);
            }
        },
        [onError, startPolling]
    );

    const cancel = useCallback(() => {
        stopPolling();
        setState((prev) => ({ ...prev, isPolling: false }));
    }, [stopPolling]);

    const reset = useCallback(() => {
        cancel();
        setState({
            jobId: null,
            status: null,
            outputUrl: null,
            error: null,
            isSubmitting: false,
            isPolling: false,
        });
    }, [cancel]);

    useEffect(() => {
        setState((prev) => ({ ...prev, isPolling }));
    }, [isPolling]);

    return { state, submit, cancel, reset };
}
