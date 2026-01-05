import { isPendingStatus, isTerminalStatus } from '../../types';

interface StatusBadgeProps {
    status: string | null | undefined;
}

function normalize(status: string | null | undefined): string {
    return String(status || '').trim().toUpperCase();
}

function statusClass(status: string): string {
    if (status === 'COMPLETED' || status === 'SUCCESS' || status === 'SUCCEEDED') {
        return 'bg-green-500/15 text-green-300 border-green-500/30';
    }
    if (status === 'ERROR' || status === 'FAILURE' || status === 'FAILED') {
        return 'bg-red-500/15 text-red-300 border-red-500/30';
    }
    if (status === 'CANCELLED' || status === 'REVOKED') {
        return 'bg-yellow-500/15 text-yellow-200 border-yellow-500/30';
    }
    if (status === 'STARTED' || status === 'RUNNING' || status === 'PROGRESS') {
        return 'bg-blue-500/15 text-blue-200 border-blue-500/30';
    }
    if (status === 'SUBMITTED' || status === 'PENDING' || status === 'CREATED') {
        return 'bg-slate-500/15 text-slate-200 border-slate-500/30';
    }
    if (isTerminalStatus(status)) {
        return 'bg-slate-500/15 text-slate-200 border-slate-500/30';
    }
    if (isPendingStatus(status)) {
        return 'bg-blue-500/15 text-blue-200 border-blue-500/30';
    }
    return 'bg-slate-500/15 text-slate-200 border-slate-500/30';
}

export function StatusBadge({ status }: StatusBadgeProps) {
    const normalized = normalize(status);
    return (
        <span
            className={`inline-flex items-center px-2 py-0.5 rounded-md text-xs border ${statusClass(
                normalized
            )}`}
        >
            {normalized || 'UNKNOWN'}
        </span>
    );
}
