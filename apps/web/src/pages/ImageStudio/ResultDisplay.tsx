import type { ImageGenJob } from '../../types';
import { StatusBadge } from '../../components/ui';

interface ResultDisplayProps {
    job: ImageGenJob | null;
    status: string | null;
    progress: number;
    isPolling: boolean;
    error: string | null;
    imageUrl: string | null;
    statusMessage: string | null;
    onClear: () => void;
}

export function ResultDisplay({
    job,
    status,
    progress,
    isPolling,
    error,
    imageUrl,
    statusMessage,
    onClear,
}: ResultDisplayProps) {
    return (
        <div className="card">
            <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold">结果</h3>
                {job && (
                    <button
                        onClick={onClear}
                        className="text-sm text-[var(--color-text-muted)] hover:text-[var(--color-text-primary)]"
                    >
                        清除
                    </button>
                )}
            </div>

            {job && (
                <div className="mb-4">
                    <div className="flex items-center justify-between text-sm mb-2">
                        <span className="text-[var(--color-text-muted)]">
                            状态: <StatusBadge status={status} />
                        </span>
                        {isPolling && statusMessage && (
                            <span className="text-[var(--color-primary)]">
                                {statusMessage}
                            </span>
                        )}
                    </div>

                    {isPolling && (
                        <div className="w-full h-2 bg-[var(--color-bg-tertiary)] rounded-full overflow-hidden">
                            <div
                                className="h-full bg-[var(--color-primary)] transition-all duration-300"
                                style={{ width: `${progress}%` }}
                            />
                        </div>
                    )}
                </div>
            )}

            {error && (
                <div className="mb-4 p-3 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400">
                    {error}
                </div>
            )}

            {imageUrl ? (
                <div className="space-y-4">
                    <img
                        src={imageUrl}
                        alt="Generated"
                        className="w-full rounded-lg border border-[var(--color-border)]"
                    />
                    <div className="flex gap-2">
                        <a
                            href={imageUrl}
                            download
                            target="_blank"
                            rel="noopener noreferrer"
                            className="btn-secondary flex-1 text-center"
                        >
                            📥 下载
                        </a>
                        <a
                            href={imageUrl}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="btn-secondary flex-1 text-center"
                        >
                            🔗 原图
                        </a>
                    </div>
                </div>
            ) : isPolling ? (
                <div className="flex flex-col items-center justify-center py-16 text-[var(--color-text-muted)]">
                    <div className="animate-spin text-4xl mb-4">🎨</div>
                    <p>正在生成图像...</p>
                </div>
            ) : (
                <div className="flex flex-col items-center justify-center py-16 text-[var(--color-text-muted)]">
                    <div className="text-4xl mb-4">🖼️</div>
                    <p>输入 Prompt 开始生成</p>
                </div>
            )}
        </div>
    );
}
