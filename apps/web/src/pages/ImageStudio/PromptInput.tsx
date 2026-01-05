import type { ChangeEvent } from 'react';
import { PromptOptimizeButton } from '../../components/PromptOptimizeButton';

interface PromptInputProps {
    prompt: string;
    onChange: (value: string) => void;
    disabled?: boolean;
}

export function PromptInput({
    prompt,
    onChange,
    disabled = false,
}: PromptInputProps) {
    return (
        <div className="card">
            <h3 className="font-semibold mb-4">Prompt</h3>
            <textarea
                value={prompt}
                onChange={(e: ChangeEvent<HTMLTextAreaElement>) => onChange(e.target.value)}
                placeholder="描述你想生成的图像..."
                className="w-full h-32 p-3 rounded-lg bg-[var(--color-bg-tertiary)] border border-[var(--color-border)] resize-none mb-4"
                disabled={disabled}
            />
            <PromptOptimizeButton
                text={prompt}
                onOptimized={onChange}
                disabled={disabled}
            />
        </div>
    );
}

