import type { Job, I2VParams } from '../types';

interface ResultPanelProps {
    job: Job | null;
    onRetry?: (params: I2VParams) => void;
}

export function ResultPanel({ job, onRetry }: ResultPanelProps) {
    if (!job) {
        return (
            <div className="card h-full flex items-center justify-center min-h-80">
                <div className="text-center">
                    <div className="text-5xl mb-4">🎬</div>
                    <p className="text-[var(--color-text-secondary)]">
                        Select a job to view result
                    </p>
                </div>
            </div>
        );
    }

    const isRunning = job.status === 'PENDING' || job.status === 'STARTED';
    const isSuccess = job.status === 'SUCCESS' && job.output_url;
    const isFailure = job.status === 'FAILURE' || job.error;
    const durationSeconds = Number.isFinite(job.params.duration_seconds) ? job.params.duration_seconds : 5;

    return (
        <div className="card h-full">
            {/* Running State */}
            {isRunning && (
                <div className="flex flex-col items-center justify-center min-h-80 py-12">
                    <div className="relative w-20 h-20 mb-6">
                        <div className="absolute inset-0 rounded-full border-4 border-[var(--color-border)]" />
                        <div className="absolute inset-0 rounded-full border-4 border-t-[var(--color-accent-primary)] animate-spin" />
                    </div>
                    <p className="text-lg font-medium text-[var(--color-text-primary)]">
                        Generating video...
                    </p>
                    <p className="text-sm text-[var(--color-text-muted)] mt-2">
                        Status: {job.db_status || job.status}
                    </p>
                    <p className="text-xs text-[var(--color-text-muted)] mt-1">
                        Target duration: {durationSeconds}s
                    </p>
                    <p className="text-xs text-[var(--color-text-muted)] mt-4">
                        This may take a few minutes
                    </p>
                </div>
            )}

            {/* Success State */}
            {isSuccess && (
                <div className="space-y-4">
                    {/* Video Player */}
                    <div className="rounded-xl overflow-hidden bg-black">
                        <video
                            src={job.output_url}
                            controls
                            autoPlay
                            loop
                            className="w-full"
                        />
                    </div>

                    {/* Info & Actions */}
                    <div className="flex items-center justify-between">
                        <div className="text-sm text-[var(--color-text-muted)]">
                            <span>Seed: {job.params.seed}</span>
                            <span className="mx-2">•</span>
                            <span>Steps: {job.params.num_steps}</span>
                            <span className="mx-2">•</span>
                            <span>Duration: {durationSeconds}s</span>
                        </div>

                        <a
                            href={job.output_url}
                            download
                            target="_blank"
                            rel="noopener noreferrer"
                            className="btn-primary inline-flex items-center gap-2"
                        >
                            <span>⬇️</span>
                            Download
                        </a>
                    </div>

                    {/* Original prompt */}
                    <div className="p-3 rounded-lg bg-[var(--color-bg-secondary)]">
                        <p className="text-xs text-[var(--color-text-muted)] mb-1">Prompt</p>
                        <p className="text-sm text-[var(--color-text-secondary)]">
                            {job.inputs.prompt}
                        </p>
                    </div>
                </div>
            )}

            {/* Failure State */}
            {isFailure && !isRunning && (
                <div className="flex flex-col items-center justify-center min-h-80 py-12">
                    <div className="text-5xl mb-4">❌</div>
                    <p className="text-lg font-medium text-[var(--color-error)]">
                        Generation Failed
                    </p>

                    {/* Error message */}
                    <div className="mt-4 p-4 rounded-lg bg-[var(--color-error)]/10 border border-[var(--color-error)]/20 max-w-md">
                        <pre className="text-sm text-[var(--color-error)] whitespace-pre-wrap font-mono">
                            {job.error || 'Unknown error occurred'}
                        </pre>
                    </div>

                    {/* Retry button */}
                    {onRetry && (
                        <button
                            onClick={() => onRetry(job.params)}
                            className="btn-secondary mt-6 inline-flex items-center gap-2"
                        >
                            <span>🔄</span>
                            Retry with Different Seed
                        </button>
                    )}
                </div>
            )}
        </div>
    );
}
