import { PromptOptimizeButton } from '../../components/PromptOptimizeButton';
import { SubmitButton } from '../../components';

interface SynthesisPanelProps {
    text: string;
    setText: (text: string) => void;
    onSubmit: () => void;
    isSubmitting: boolean;
    canSubmit: boolean;
}

export function SynthesisPanel({
    text,
    setText,
    onSubmit,
    isSubmitting,
    canSubmit,
}: SynthesisPanelProps) {
    return (
        <>
            <div className="card">
                <h3 className="font-semibold mb-3">合成文本</h3>
                <textarea
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    placeholder="输入要合成的文本..."
                    className="w-full h-28 p-3 rounded-lg bg-[var(--color-bg-tertiary)] border border-[var(--color-border)] resize-none"
                    disabled={isSubmitting}
                />
                <div className="flex items-center justify-between mt-2">
                    <PromptOptimizeButton
                        text={text}
                        onOptimized={setText}
                        disabled={isSubmitting}
                        size="sm"
                    />
                    <span className="text-xs text-[var(--color-text-muted)]">
                        {text.length} / 5000
                    </span>
                </div>
            </div>

            <SubmitButton
                onClick={onSubmit}
                disabled={!canSubmit || isSubmitting}
                isLoading={isSubmitting}
                loadingText="提交中..."
                className="w-full py-4 text-lg"
            >
                🎤 开始合成
            </SubmitButton>
        </>
    );
}
