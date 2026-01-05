import { useState, useEffect, useCallback } from 'react';

export interface JobHistoryItem {
    jobId: string;
    createdAt: number;
    status: 'success' | 'failed';
    inputs: Record<string, unknown>;
    outputUrl?: string;
    error?: string;
}

export function useJobHistory(storageKey: string, maxItems = 20) {
    const [history, setHistory] = useState<JobHistoryItem[]>([]);

    useEffect(() => {
        try {
            const saved = localStorage.getItem(storageKey);
            if (saved) {
                setHistory(JSON.parse(saved));
            }
        } catch (e) {
            console.error('Failed to load job history:', e);
        }
    }, [storageKey]);

    const addItem = useCallback(
        (item: JobHistoryItem) => {
            setHistory((prev) => {
                const next = [item, ...prev].slice(0, maxItems);
                try {
                    localStorage.setItem(storageKey, JSON.stringify(next));
                } catch (e) {
                    console.error('Failed to save job history:', e);
                }
                return next;
            });
        },
        [storageKey, maxItems]
    );

    const clearHistory = useCallback(() => {
        try {
            localStorage.removeItem(storageKey);
        } catch (e) {
            console.error('Failed to clear job history:', e);
        }
        setHistory([]);
    }, [storageKey]);

    const removeItem = useCallback(
        (jobId: string) => {
            setHistory((prev) => {
                const next = prev.filter((item) => item.jobId !== jobId);
                try {
                    localStorage.setItem(storageKey, JSON.stringify(next));
                } catch (e) {
                    console.error('Failed to save job history:', e);
                }
                return next;
            });
        },
        [storageKey]
    );

    return { history, addItem, removeItem, clearHistory };
}
