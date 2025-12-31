import { useState, useCallback, useMemo } from 'react';
import { InputPanel } from '../components/InputPanel';
import { ParamsPanel } from '../components/ParamsPanel';
import { JobPanel } from '../components/JobPanel';
import { ResultPanel } from '../components/ResultPanel';
import { useJobStorage, useJobPolling } from '../hooks';
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

    // Polling
    useJobPolling(jobs, updateJob, {
        onComplete: (job) => {
            console.log('Job completed:', job.job_id, job.status);
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
            setError(err instanceof Error ? err.message : 'Failed to create job');
        } finally {
            setIsSubmitting(false);
        }
    }, [imageFile, prompt, params, imagePreview, addJob]);

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
            <div className="mb-8">
                <h1 className="text-3xl font-bold bg-gradient-to-r from-[var(--color-accent-primary)] to-[var(--color-accent-secondary)] bg-clip-text text-transparent">
                    I2V Studio
                </h1>
                <p className="text-[var(--color-text-secondary)] mt-2">
                    Transform images into stunning videos with AI
                </p>
            </div>

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
                    {error && (
                        <div className="p-4 rounded-lg bg-[var(--color-error)]/10 border border-[var(--color-error)]/20">
                            <p className="text-sm text-[var(--color-error)]">{error}</p>
                        </div>
                    )}

                    {/* Submit button */}
                    <button
                        onClick={handleSubmit}
                        disabled={!canSubmit}
                        className="btn-primary w-full py-4 text-lg"
                    >
                        {isSubmitting ? (
                            <span className="inline-flex items-center gap-2">
                                <span className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                                Submitting...
                            </span>
                        ) : (
                            <span className="inline-flex items-center gap-2">
                                <span>🎬</span>
                                Generate Video
                            </span>
                        )}
                    </button>
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
