import { useState, useCallback, useMemo } from 'react';
import { InputPanel } from '../components/InputPanel';
import { ParamsPanel } from '../components/ParamsPanel';
import { JobPanel } from '../components/JobPanel';
import { ResultPanel } from '../components/ResultPanel';
import { ErrorAlert, StudioHeader, SubmitButton } from '../components';
import { useJobStorage, useJobPolling } from '../hooks';
import { useCapabilityHealth } from '../hooks/useCapabilityHealth';
import { createI2VJob } from '../services/videogen';
import type { I2VJob, I2VParams } from '../types';
import { DEFAULT_I2V_PARAMS } from '../types';

export function I2VStudio() {
    // Form state
    const [imageFile, setImageFile] = useState<File | null>(null);
    const [imagePreview, setImagePreview] = useState<string | null>(null);
    const [prompt, setPrompt] = useState('');
    const [params, setParams] = useState<I2VParams>(DEFAULT_I2V_PARAMS);

    // UI state
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);

    // Job storage
    const { jobs, addJob, updateJob, removeJob } = useJobStorage();

    const { reportFailure, reportSuccess } = useCapabilityHealth();

    // Polling
    useJobPolling(jobs, updateJob, {
        onComplete: (job) => {
            if (job.status === 'SUCCESS') {
                reportSuccess('videogen');
            } else if (job.status === 'FAILURE') {
                reportFailure('videogen', job.error || 'Video generation failed');
            }
        },
    });

    // Selected job
    const selectedJob = useMemo(() => {
        return jobs.find((j) => j.job_id === selectedJobId) || null;
    }, [jobs, selectedJobId]);

    // Validation
    const canSubmit = useMemo(() => {
        return imageFile !== null && prompt.trim().length > 0 && !isSubmitting;
    }, [imageFile, prompt, isSubmitting]);

    // Handlers
    const handleImageChange = useCallback((file: File | null, preview: string | null) => {
        setImageFile(file);
        setImagePreview(preview);
        setError(null);
    }, []);

    const handlePromptChange = useCallback((value: string) => {
        setPrompt(value);
        setError(null);
    }, []);

    const handleSubmit = useCallback(async () => {
        if (!imageFile || !prompt.trim()) return;

        setIsSubmitting(true);
        setError(null);

        try {
            const response = await createI2VJob(imageFile, prompt.trim(), params);

            // Create local job record
            const newJob: I2VJob = {
                job_id: response.job_id,
                job_type: 'i2v',
                status: 'PENDING',
                inputs: {
                    prompt: prompt.trim(),
                    image_preview: imagePreview || undefined,
                },
                params: { ...params },
                created_at: Date.now(),
            };

            addJob(newJob);
            setSelectedJobId(response.job_id);

            // Reset form (keep params for easy iteration)
            setImageFile(null);
            setImagePreview(null);
            setPrompt('');
        } catch (err) {
            console.error('Failed to create job:', err);
            const message = err instanceof Error ? err.message : 'Failed to create job';
            setError(message);
            reportFailure('videogen', message);
        } finally {
            setIsSubmitting(false);
        }
    }, [imageFile, prompt, params, imagePreview, addJob, reportFailure]);

    const handleRetry = useCallback((prevParams: I2VParams) => {
        // Set new random seed but keep other params
        setParams({
            ...DEFAULT_I2V_PARAMS,
            ...prevParams,
            seed: Math.floor(Math.random() * 2147483647),
        });
    }, []);

    return (
        <div className="animate-fade-in">
            {/* Header */}
            <StudioHeader
                title="I2V Studio"
                description="Transform images into stunning videos with AI"
                titleClassName="text-3xl font-bold bg-gradient-to-r from-[var(--color-accent-primary)] to-[var(--color-accent-secondary)] bg-clip-text text-transparent"
            />

            {/* Main Content */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Left Column - Input */}
                <div className="space-y-6">
                    <div className="card">
                        <InputPanel
                            imagePreview={imagePreview}
                            prompt={prompt}
                            onImageChange={handleImageChange}
                            onPromptChange={handlePromptChange}
                            disabled={isSubmitting}
                        />
                    </div>

                    <ParamsPanel
                        params={params}
                        onChange={setParams}
                        disabled={isSubmitting}
                    />

                    {/* Error message */}
                    <ErrorAlert message={error} />

                    {/* Submit button */}
                    <SubmitButton
                        onClick={handleSubmit}
                        disabled={!canSubmit}
                        isLoading={isSubmitting}
                        loadingText="Submitting..."
                        className="w-full py-4 text-lg"
                    >
                        <span className="inline-flex items-center gap-2">
                            <span>🎬</span>
                            Generate Video
                        </span>
                    </SubmitButton>
                </div>

                {/* Right Column - Jobs & Result */}
                <div className="space-y-6">
                    <JobPanel
                        jobs={jobs}
                        selectedJobId={selectedJobId}
                        onSelectJob={setSelectedJobId}
                        onRemoveJob={removeJob}
                    />

                    <ResultPanel
                        job={selectedJob}
                        onRetry={handleRetry}
                    />
                </div>
            </div>
        </div>
    );
}
