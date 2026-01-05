import type { JobHistoryItem } from '../hooks/useJobHistory';

interface HistoryItemProps {
    item: JobHistoryItem;
    onClick: () => void;
    onRemove: () => void;
}

function formatTime(timestamp: number): string {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return '刚刚';
    if (diffMins < 60) return `${diffMins}分钟前`;
    if (diffHours < 24) return `${diffHours}小时前`;
    if (diffDays < 7) return `${diffDays}天前`;
    return date.toLocaleDateString('zh-CN');
}

function HistoryItem({ item, onClick, onRemove }: HistoryItemProps) {
    const isSuccess = item.status === 'success';

    // Try to extract a meaningful preview from inputs
    const getPreview = (): string => {
        if (item.inputs.text) return String(item.inputs.text).slice(0, 50);
        if (item.inputs.prompt) return String(item.inputs.prompt).slice(0, 50);
        return item.jobId.slice(0, 12);
    };

    return (
        <div
            className={`group relative p-3 rounded-lg border cursor-pointer transition-all hover:scale-[1.01] ${isSuccess
                    ? 'border-emerald-500/30 bg-emerald-500/5 hover:bg-emerald-500/10'
                    : 'border-rose-500/30 bg-rose-500/5 hover:bg-rose-500/10'
                }`}
            onClick={onClick}
        >
            <div className="flex items-start justify-between gap-2">
                <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                        <span
                            className={`h-2 w-2 rounded-full ${isSuccess ? 'bg-emerald-500' : 'bg-rose-500'}`}
                        />
                        <span className="text-xs text-[var(--color-text-muted)]">
                            {formatTime(item.createdAt)}
                        </span>
                    </div>
                    <p className="text-sm text-[var(--color-text-primary)] truncate">{getPreview()}</p>
                    {item.error && (
                        <p className="text-xs text-rose-400 mt-1 truncate" title={item.error}>
                            {item.error}
                        </p>
                    )}
                </div>
                <button
                    onClick={(e) => {
                        e.stopPropagation();
                        onRemove();
                    }}
                    className="opacity-0 group-hover:opacity-100 p-1 text-[var(--color-text-muted)] hover:text-rose-400 transition-opacity"
                    title="删除"
                >
                    ×
                </button>
            </div>
        </div>
    );
}

interface JobHistoryPanelProps {
    history: JobHistoryItem[];
    onSelect: (item: JobHistoryItem) => void;
    onRemove: (jobId: string) => void;
    onClear: () => void;
    title?: string;
    emptyText?: string;
}

export function JobHistoryPanel({
    history,
    onSelect,
    onRemove,
    onClear,
    title = '历史记录',
    emptyText = '暂无历史记录',
}: JobHistoryPanelProps) {
    if (history.length === 0) {
        return (
            <div className="card">
                <h3 className="font-semibold mb-3">{title}</h3>
                <p className="text-sm text-[var(--color-text-muted)]">{emptyText}</p>
            </div>
        );
    }

    return (
        <div className="card">
            <div className="flex items-center justify-between mb-3">
                <h3 className="font-semibold">{title}</h3>
                <div className="flex items-center gap-2">
                    <span className="text-xs text-[var(--color-text-muted)]">{history.length} 条</span>
                    <button
                        onClick={onClear}
                        className="text-xs text-[var(--color-text-muted)] hover:text-rose-400 transition-colors"
                    >
                        清空
                    </button>
                </div>
            </div>
            <div className="space-y-2 max-h-64 overflow-y-auto">
                {history.map((item) => (
                    <HistoryItem
                        key={item.jobId}
                        item={item}
                        onClick={() => onSelect(item)}
                        onRemove={() => onRemove(item.jobId)}
                    />
                ))}
            </div>
        </div>
    );
}
