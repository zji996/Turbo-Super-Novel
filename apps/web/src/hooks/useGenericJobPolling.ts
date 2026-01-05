import { useCallback, useEffect, useRef, useState } from 'react';
import { isTerminalStatus } from '../types';

export interface UseGenericJobPollingOptions<TJob> {
    pollingInterval?: number | ((job: TJob) => number);
    maxPollingDuration?: number;
    getStatus: (jobId: string) => Promise<TJob>;
    getJobStatus: (job: TJob) => string | null | undefined;
    isTerminal?: (status: string | null | undefined, job: TJob) => boolean;
    onStatusUpdate?: (job: TJob) => void;
    onComplete?: (job: TJob) => void;
    onError?: (error: Error, jobId: string) => void;
    onTimeout?: (jobId: string) => void;
    errorRetryInterval?: number | ((attempt: number) => number);
}

export interface UseGenericJobPollingResult {
    startPolling: (jobId: string) => void;
    stopPolling: () => void;
    isPolling: boolean;
    jobId: string | null;
}

const DEFAULT_POLLING_INTERVAL = 2000;
const DEFAULT_MAX_POLLING_DURATION = 30 * 60 * 1000;

export function useGenericJobPolling<TJob>(
    options: UseGenericJobPollingOptions<TJob>
): UseGenericJobPollingResult {
    const optionsRef = useRef(options);
    const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
    const startTimeRef = useRef<number | null>(null);
    const jobIdRef = useRef<string | null>(null);
    const errorAttemptRef = useRef(0);

    const [isPolling, setIsPolling] = useState(false);
    const [jobId, setJobId] = useState<string | null>(null);

    useEffect(() => {
        optionsRef.current = options;
    }, [options]);

    const stopPolling = useCallback(() => {
        if (timeoutRef.current) {
            clearTimeout(timeoutRef.current);
            timeoutRef.current = null;
        }
        startTimeRef.current = null;
        jobIdRef.current = null;
        errorAttemptRef.current = 0;
        setJobId(null);
        setIsPolling(false);
    }, []);

    const pollOnce = useCallback(async () => {
        const currentJobId = jobIdRef.current;
        const {
            pollingInterval = DEFAULT_POLLING_INTERVAL,
            maxPollingDuration = DEFAULT_MAX_POLLING_DURATION,
            getStatus,
            getJobStatus,
            isTerminal,
            onStatusUpdate,
            onComplete,
            onError,
            onTimeout,
            errorRetryInterval,
        } = optionsRef.current;

        if (!currentJobId) return;

        if (
            startTimeRef.current != null &&
            Date.now() - startTimeRef.current > maxPollingDuration
        ) {
            onTimeout?.(currentJobId);
            stopPolling();
            return;
        }

        try {
            const job = await getStatus(currentJobId);
            errorAttemptRef.current = 0;

            onStatusUpdate?.(job);

            const status = getJobStatus(job);
            const terminal = isTerminal ? isTerminal(status, job) : isTerminalStatus(status);
            if (terminal) {
                stopPolling();
                onComplete?.(job);
                return;
            }

            const intervalMs =
                typeof pollingInterval === 'function'
                    ? pollingInterval(job)
                    : pollingInterval;
            timeoutRef.current = setTimeout(pollOnce, intervalMs);
        } catch (error) {
            const err = error instanceof Error ? error : new Error(String(error));
            errorAttemptRef.current += 1;
            onError?.(err, currentJobId);

            const retryMs =
                typeof errorRetryInterval === 'function'
                    ? errorRetryInterval(errorAttemptRef.current)
                    : errorRetryInterval ??
                      (typeof pollingInterval === 'number'
                          ? pollingInterval * 2
                          : DEFAULT_POLLING_INTERVAL * 2);

            timeoutRef.current = setTimeout(pollOnce, retryMs);
        }
    }, [stopPolling]);

    const startPolling = useCallback(
        (nextJobId: string) => {
            stopPolling();
            jobIdRef.current = nextJobId;
            startTimeRef.current = Date.now();
            setJobId(nextJobId);
            setIsPolling(true);
            pollOnce();
        },
        [pollOnce, stopPolling]
    );

    useEffect(() => stopPolling, [stopPolling]);

    return {
        startPolling,
        stopPolling,
        isPolling,
        jobId,
    };
}

