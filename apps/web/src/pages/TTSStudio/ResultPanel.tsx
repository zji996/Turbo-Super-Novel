import { StatusBadge } from '../../components/ui';
import type { TTSJobState } from '../../hooks/useTTSJob';
import { isTerminalStatus } from '../../types';

interface ResultPanelProps {
    jobState: TTSJobState;
    onReset: () => void;
}

export function ResultPanel({ jobState, onReset }: ResultPanelProps) {
    return (
        <div className="card">
            <h3 className="font-semibold mb-4">合成结果</h3>
            {jobState.jobId ? (
                <div className="space-y-4">
                    <div className="flex items-center gap-2">
                        <span>状态:</span>
                        <StatusBadge status={jobState.status} />
                    </div>
                    {jobState.isPolling && !isTerminalStatus(jobState.status) && (
                        <div className="flex items-center gap-2 text-sm text-[var(--color-text-muted)]">
                            <span className="animate-pulse">●</span>
                            处理中...
                        </div>
                    )}
                    {jobState.error && (
                        <p className="text-sm text-red-500">{jobState.error}</p>
                    )}
                    {jobState.outputUrl && (
                        <div>
                            <audio controls src={jobState.outputUrl} className="w-full" />
                            <a
                                href={jobState.outputUrl}
                                download
                                className="btn-secondary w-full mt-2 text-center block"
                            >
                                下载音频
                            </a>
                        </div>
                    )}
                    <button onClick={onReset} className="btn-secondary w-full">
                        重置
                    </button>
                </div>
            ) : (
                <p className="text-[var(--color-text-muted)]">暂无任务</p>
            )}
        </div>
    );
}
