import { useCallback, useEffect, useRef, useState } from 'react';
import { createTTSJobWithProfile, getTTSJobStatus } from '../services/tts';

export interface TTSJobRequest {
    text: string;
    profile_id: string;
}

export interface TTSJobState {
    jobId: string | null;
    status: string | null;
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

    const intervalRef = useRef<number | null>(null);
    const stateRef = useRef(state);

    useEffect(() => {
        stateRef.current = state;
    }, [state]);

    const submit = useCallback(
        async (request: TTSJobRequest) => {
            setState((prev) => ({ ...prev, isSubmitting: true, error: null }));
            try {
                const resp = await createTTSJobWithProfile(request);
                setState({
                    jobId: resp.job_id,
                    status: 'SUBMITTED',
                    outputUrl: null,
                    error: null,
                    isSubmitting: false,
                    isPolling: true,
                });
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
        [onError]
    );

    const cancel = useCallback(() => {
        if (intervalRef.current) {
            clearInterval(intervalRef.current);
            intervalRef.current = null;
        }
        setState((prev) => ({ ...prev, isPolling: false }));
    }, []);

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
        if (!state.jobId || !state.isPolling) return;

        const jobId = state.jobId;

        const poll = async () => {
            try {
                const data = await getTTSJobStatus(jobId);
                const status =
                    data.status || (data as any).db?.status || (data as any).celery_status || 'PENDING';
                const outputUrl = data.output_url || null;

                const nextState: TTSJobState = {
                    ...stateRef.current,
                    jobId,
                    status,
                    outputUrl,
                };

                stateRef.current = nextState;
                setState(nextState);

                if (['SUCCEEDED', 'FAILED'].includes(status)) {
                    cancel();
                    if (status === 'SUCCEEDED') {
                        onSuccess?.(nextState);
                    } else {
                        onError?.(new Error(data.error || 'TTS job failed'));
                    }
                }
            } catch (e) {
                console.error('Polling error:', e);
            }
        };

        intervalRef.current = window.setInterval(poll, pollingInterval);
        poll();

        return () => {
            if (intervalRef.current) {
                clearInterval(intervalRef.current);
                intervalRef.current = null;
            }
        };
    }, [state.jobId, state.isPolling, pollingInterval, cancel, onSuccess, onError]);

    return { state, submit, cancel, reset };
}
