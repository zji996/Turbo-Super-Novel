import { useState, useEffect, useCallback } from 'react';
import type { Job } from '../types';

const STORAGE_KEY = 'i2v-jobs';

/**
 * Hook for persisting jobs in localStorage
 */
export function useJobStorage() {
    const [jobs, setJobs] = useState<Job[]>(() => {
        try {
            const stored = localStorage.getItem(STORAGE_KEY);
            if (stored) {
                return JSON.parse(stored) as Job[];
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

    const addJob = useCallback((job: Job) => {
        setJobs((prev) => [job, ...prev]);
    }, []);

    const updateJob = useCallback((jobId: string, updates: Partial<Job>) => {
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
