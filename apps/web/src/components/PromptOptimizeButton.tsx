import { useCallback } from 'react';
import { usePromptOptimizer } from '../hooks/usePromptOptimizer';

interface PromptOptimizeButtonProps {
    /** 当前文本 */
    text: string;
    /** 优化完成后的回调，传入优化后的文本 */
    onOptimized: (optimized: string) => void;
    /** 是否禁用 */
    disabled?: boolean;
    /** 自定义类名 */
    className?: string;
    /** 按钮大小 */
    size?: 'sm' | 'md';
}

/**
 * 提示词优化按钮组件
 * 
 * 包含：
 * - ✨ AI 优化 按钮：调用 AI 优化文本
 * - ↩ 还原 按钮：撤销到优化前的文本
 */
export function PromptOptimizeButton({
    text,
    onOptimized,
    disabled = false,
    className = '',
    size = 'md',
}: PromptOptimizeButtonProps) {
    const { isOptimizing, canUndo, optimize, undo, error } = usePromptOptimizer();

    const handleOptimize = useCallback(async () => {
        const result = await optimize(text);
        if (result) {
            onOptimized(result);
        }
    }, [text, optimize, onOptimized]);

    const handleUndo = useCallback(() => {
        const original = undo();
        if (original) {
            onOptimized(original);
        }
    }, [undo, onOptimized]);

    const sizeClasses = size === 'sm' ? 'text-sm py-1.5 px-3' : 'py-2 px-4';
    const isDisabled = disabled || isOptimizing || !text.trim();

    return (
        <div className={`flex flex-wrap items-center gap-2 ${className}`}>
            <button
                onClick={handleOptimize}
                disabled={isDisabled}
                className={`btn-secondary ${sizeClasses} inline-flex items-center gap-1.5`}
            >
                {isOptimizing ? (
                    <>
                        <span className="w-4 h-4 border-2 border-current/30 border-t-current rounded-full animate-spin" />
                        优化中...
                    </>
                ) : (
                    <>✨ AI 优化</>
                )}
            </button>

            {canUndo && (
                <button
                    onClick={handleUndo}
                    className={`btn-secondary ${sizeClasses} inline-flex items-center gap-1`}
                >
                    ↩ 还原
                </button>
            )}

            {error && (
                <span className="text-xs text-[var(--color-error)]">{error}</span>
            )}
        </div>
    );
}
