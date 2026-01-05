import type { CapabilityHealthStatus } from '../hooks/capabilityHealthContext';

interface CapabilityCardProps {
    label: string;
    status: CapabilityHealthStatus;
    provider?: string | null;
    lastError?: string | null;
    lastSuccess?: number | null;
}

function getStatusColor(status: CapabilityHealthStatus): string {
    switch (status) {
        case 'available':
            return 'bg-emerald-500';
        case 'unavailable':
            return 'bg-rose-500';
        default:
            return 'bg-slate-500';
    }
}

function getStatusLabel(status: CapabilityHealthStatus): string {
    switch (status) {
        case 'available':
            return '可用';
        case 'unavailable':
            return '不可用';
        default:
            return '未知';
    }
}

function getStatusBgClass(status: CapabilityHealthStatus): string {
    switch (status) {
        case 'available':
            return 'bg-emerald-500/10 border-emerald-500/30';
        case 'unavailable':
            return 'bg-rose-500/10 border-rose-500/30';
        default:
            return 'bg-slate-500/10 border-slate-500/30';
    }
}

function formatTime(timestamp: number | null): string {
    if (!timestamp) return '';
    const date = new Date(timestamp);
    return date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
}

export function CapabilityCard({
    label,
    status,
    provider,
    lastError,
    lastSuccess,
}: CapabilityCardProps) {
    return (
        <div
            className={`rounded-lg border p-4 transition-all hover:scale-[1.02] ${getStatusBgClass(status)}`}
            title={lastError || undefined}
        >
            <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                    <span className={`h-3 w-3 rounded-full ${getStatusColor(status)} animate-pulse`} />
                    <span className="font-semibold text-[var(--color-text-primary)]">{label}</span>
                </div>
                <span className="text-xs text-[var(--color-text-muted)]">{getStatusLabel(status)}</span>
            </div>
            <div className="text-xs text-[var(--color-text-muted)] space-y-1">
                {provider && (
                    <div className="flex items-center gap-1">
                        <span>Provider:</span>
                        <span className="text-[var(--color-text-secondary)]">{provider}</span>
                    </div>
                )}
                {lastSuccess && (
                    <div className="flex items-center gap-1">
                        <span>✓</span>
                        <span className="text-[var(--color-text-secondary)]">{formatTime(lastSuccess)}</span>
                    </div>
                )}
                {lastError && (
                    <div className="text-rose-400 truncate" title={lastError}>
                        {lastError}
                    </div>
                )}
            </div>
        </div>
    );
}
