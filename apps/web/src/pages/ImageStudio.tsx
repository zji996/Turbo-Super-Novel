import { useState, useCallback } from 'react';
import { createImageGenJob } from '../services/imagegen';
import { optimizePrompt } from '../services/llm';

export function ImageStudio() {
    const [prompt, setPrompt] = useState('');
    const [isOptimizing, setIsOptimizing] = useState(false);
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [result, setResult] = useState<{ url?: string; status: string } | null>(null);

    const handleOptimize = useCallback(async () => {
        if (!prompt.trim()) return;
        setIsOptimizing(true);
        try {
            const optimized = await optimizePrompt(prompt);
            setPrompt(optimized);
        } catch (e) {
            console.error(e);
        } finally {
            setIsOptimizing(false);
        }
    }, [prompt]);

    const handleSubmit = useCallback(async () => {
        if (!prompt.trim()) return;
        setIsSubmitting(true);
        try {
            const resp = await createImageGenJob(prompt);
            setResult({ status: 'SUBMITTED', url: resp.output_url });
        } catch (e) {
            console.error(e);
        } finally {
            setIsSubmitting(false);
        }
    }, [prompt]);

    return (
        <div className="animate-fade-in">
            <h1 className="text-3xl font-bold mb-2">🖼️ Image Studio</h1>
            <p className="text-[var(--color-text-secondary)] mb-8">图像生成工具</p>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <div className="card">
                    <h3 className="font-semibold mb-4">Prompt</h3>
                    <textarea
                        value={prompt}
                        onChange={(e) => setPrompt(e.target.value)}
                        placeholder="描述你想生成的图像..."
                        className="w-full h-32 p-3 rounded-lg bg-[var(--color-bg-tertiary)] border border-[var(--color-border)] resize-none mb-4"
                    />
                    <div className="flex gap-2">
                        <button
                            onClick={handleOptimize}
                            disabled={isOptimizing}
                            className="btn-secondary flex-1"
                        >
                            {isOptimizing ? '优化中...' : '✨ AI 优化 Prompt'}
                        </button>
                        <button
                            onClick={handleSubmit}
                            disabled={!prompt.trim() || isSubmitting}
                            className="btn-primary flex-1"
                        >
                            {isSubmitting ? '生成中...' : '🎨 生成图像'}
                        </button>
                    </div>
                </div>

                <div className="card">
                    <h3 className="font-semibold mb-4">结果</h3>
                    {result?.url ? (
                        <img src={result.url} alt="Generated" className="w-full rounded-lg" />
                    ) : result ? (
                        <p className="text-[var(--color-text-muted)]">
                            状态: <span className="font-mono">{result.status}</span>
                        </p>
                    ) : (
                        <p className="text-[var(--color-text-muted)]">暂无结果</p>
                    )}
                </div>
            </div>
        </div>
    );
}

