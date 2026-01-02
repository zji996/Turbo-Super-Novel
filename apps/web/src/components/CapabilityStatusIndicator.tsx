import { useCallback } from 'react';
import type { CapabilityName } from '../services/capabilities';
import { useCapabilityHealth } from '../hooks/useCapabilityHealth';

const LABELS: Record<CapabilityName, string> = {
    tts: 'TTS',
    imagegen: 'Image',
    videogen: 'Video',
    llm: 'LLM',
};

function dotClass(status: 'unknown' | 'available' | 'unavailable'): string {
    if (status === 'available') return 'bg-emerald-500';
    if (status === 'unavailable') return 'bg-rose-500';
    return 'bg-slate-500';
}

export function CapabilityStatusIndicator() {
    const { health, isProbing, probeError, refreshFromProbe } = useCapabilityHealth();

    const refresh = useCallback(async () => {
        await refreshFromProbe();
    }, [refreshFromProbe]);

    return (
        <div className="mt-3 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-tertiary)] p-3">
            <div className="flex items-center justify-between">
                <div className="text-xs font-semibold text-[var(--color-text-muted)] uppercase tracking-wider">
                    Capabilities
                </div>
                <button
                    onClick={refresh}
                    disabled={isProbing}
                    className="text-xs text-[var(--color-text-muted)] hover:text-[var(--color-text-primary)]"
                    title="Refresh"
                >
                    {isProbing ? '...' : '↻'}
                </button>
            </div>

            {probeError ? (
                <div className="mt-2 text-xs text-[var(--color-error)]">
                    {probeError}
                </div>
            ) : (
                <div className="mt-2 grid grid-cols-2 gap-2">
                    {(Object.keys(LABELS) as CapabilityName[]).map((name) => {
                        const cap = health[name];
                        const provider = cap.provider || 'unknown';
                        return (
                            <div
                                key={name}
                                className="flex items-center gap-2 rounded-md bg-[var(--color-bg-secondary)] px-2 py-1"
                                title={cap.lastError || undefined}
                            >
                                <span className={`h-2.5 w-2.5 rounded-full ${dotClass(cap.status)}`} />
                                <span className="text-xs text-[var(--color-text-primary)]">{LABELS[name]}</span>
                                <span className="ml-auto text-[10px] text-[var(--color-text-muted)]">
                                    {provider}
                                </span>
                            </div>
                        );
                    })}
                </div>
            )}
        </div>
    );
}
