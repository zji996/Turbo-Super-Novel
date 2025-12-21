import type { Job } from '../types';
import { formatDistanceToNow } from '../utils/time';

interface JobPanelProps {
    jobs: Job[];
    selectedJobId: string | null;
    onSelectJob: (jobId: string) => void;
    onRemoveJob: (jobId: string) => void;
}

export function JobPanel({ jobs, selectedJobId, onSelectJob, onRemoveJob }: JobPanelProps) {
    if (jobs.length === 0) {
        return (
            <div className="card text-center py-12">
                <div className="text-4xl mb-3">📭</div>
                <p className="text-[var(--color-text-secondary)]">No jobs yet</p>
                <p className="text-sm text-[var(--color-text-muted)] mt-1">
                    Upload an image and click Generate to start
                </p>
            </div>
        );
    }

    return (
        <div className="space-y-3">
            <h3 className="text-sm font-semibold text-[var(--color-text-secondary)] uppercase tracking-wider">
                Recent Jobs ({jobs.length})
            </h3>

            <div className="space-y-2 max-h-96 overflow-y-auto pr-2">
                {jobs.map((job) => (
                    <JobCard
                        key={job.job_id}
                        job={job}
                        isSelected={job.job_id === selectedJobId}
                        onSelect={() => onSelectJob(job.job_id)}
                        onRemove={() => onRemoveJob(job.job_id)}
                    />
                ))}
            </div>
        </div>
    );
}

interface JobCardProps {
    job: Job;
    isSelected: boolean;
    onSelect: () => void;
    onRemove: () => void;
}

function JobCard({ job, isSelected, onSelect, onRemove }: JobCardProps) {
    const getStatusBadge = () => {
        const isRunning = job.status === 'STARTED' || job.status === 'PENDING';
        const isSuccess = job.status === 'SUCCESS';
        const isFailure = job.status === 'FAILURE';

        if (isSuccess) {
            return <span className="badge badge-success">✓ Success</span>;
        }
        if (isFailure) {
            return <span className="badge badge-error">✕ Failed</span>;
        }
        if (isRunning) {
            return (
                <span className="badge badge-running">
                    <span className="w-2 h-2 rounded-full bg-current animate-pulse" />
                    {job.db_status || job.status}
                </span>
            );
        }
        return <span className="badge badge-pending">{job.status}</span>;
    };

    return (
        <div
            onClick={onSelect}
            className={`
        card cursor-pointer transition-all duration-200 group
        ${isSelected
                    ? 'ring-2 ring-[var(--color-accent-primary)] shadow-[var(--shadow-glow)]'
                    : 'hover:bg-[var(--color-bg-tertiary)]'
                }
      `}
        >
            <div className="flex items-start gap-3">
                {/* Thumbnail */}
                {job.inputs.image_preview && (
                    <img
                        src={job.inputs.image_preview}
                        alt="Input"
                        className="w-12 h-12 rounded-lg object-cover flex-shrink-0"
                    />
                )}

                <div className="flex-1 min-w-0">
                    {/* Status + Time */}
                    <div className="flex items-center justify-between gap-2 mb-1">
                        {getStatusBadge()}
                        <span className="text-xs text-[var(--color-text-muted)]">
                            {formatDistanceToNow(job.created_at)}
                        </span>
                    </div>

                    {/* Prompt preview */}
                    <p className="text-sm text-[var(--color-text-secondary)] truncate">
                        {job.inputs.prompt}
                    </p>

                    {/* Error message */}
                    {job.error && (
                        <p className="text-xs text-[var(--color-error)] mt-1 truncate">
                            {job.error}
                        </p>
                    )}
                </div>

                {/* Remove button */}
                <button
                    onClick={(e) => {
                        e.stopPropagation();
                        onRemove();
                    }}
                    className="p-1 rounded opacity-0 group-hover:opacity-100 hover:bg-[var(--color-error)]/20 transition-all"
                    title="Remove job"
                >
                    <span className="text-[var(--color-error)]">✕</span>
                </button>
            </div>
        </div>
    );
}
