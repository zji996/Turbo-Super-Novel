import type { ChangeEvent } from 'react';

interface PromptInputProps {
    prompt: string;
    isRunning: boolean;
    isOptimizing: boolean;
    enhancePrompt: boolean;
    onChange: (value: string) => void;
    onOptimize: () => Promise<void>;
    onEnhancePromptChange: (value: boolean) => void;
}

export function PromptInput({
    prompt,
    isRunning,
    isOptimizing,
    enhancePrompt,
    onChange,
    onOptimize,
    onEnhancePromptChange,
}: PromptInputProps) {
    return (
        <div className="card">
            <h3 className="font-semibold mb-4">Prompt</h3>
            <textarea
                value={prompt}
                onChange={(e: ChangeEvent<HTMLTextAreaElement>) => onChange(e.target.value)}
                placeholder="描述你想生成的图像..."
                className="w-full h-32 p-3 rounded-lg bg-[var(--color-bg-tertiary)] border border-[var(--color-border)] resize-none mb-4"
                disabled={isRunning}
            />
            <label className="mb-4 flex items-center gap-2 text-sm text-[var(--color-text-secondary)]">
                <input
                    type="checkbox"
                    checked={enhancePrompt}
                    onChange={(e) => onEnhancePromptChange(e.target.checked)}
                    disabled={isRunning}
                />
                提交时 AI 优化提示词
            </label>
            <div className="flex gap-2">
                <button
                    onClick={onOptimize}
                    disabled={isOptimizing || isRunning || !prompt.trim()}
                    className="btn-secondary flex-1"
                >
                    {isOptimizing ? '优化中...' : '✨ AI 优化 Prompt'}
                </button>
            </div>
        </div>
    );
}
