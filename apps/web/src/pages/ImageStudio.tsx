import { useState, useCallback } from 'react';
import { useImageGenJob } from '../hooks/useImageGenJob';
import { optimizePrompt } from '../services/llm';
import { getStatusMessage, type ImageGenParams } from '../services/imagegen';

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

const DEFAULT_PARAMS: ImageGenParams = {
    width: 1024,
    height: 1024,
    num_inference_steps: 9,
    guidance_scale: 0.0,
};

const SIZE_PRESETS = [
    { label: '1:1 方形', width: 1024, height: 1024 },
    { label: '3:4 竖版', width: 768, height: 1024 },
    { label: '4:3 横版', width: 1024, height: 768 },
    { label: '16:9 宽屏', width: 1024, height: 576 },
    { label: '9:16 竖屏', width: 576, height: 1024 },
];

// ─────────────────────────────────────────────────────────────────────────────
// Component
// ─────────────────────────────────────────────────────────────────────────────

export function ImageStudio() {
    // Prompt state
    const [prompt, setPrompt] = useState('');
    const [isOptimizing, setIsOptimizing] = useState(false);

    // Parameters state
    const [params, setParams] = useState<ImageGenParams>(DEFAULT_PARAMS);
    const [showAdvanced, setShowAdvanced] = useState(false);

    // Image generation hook
    const {
        job,
        isSubmitting,
        isPolling,
        submit,
        cancel,
        clear,
        status,
        progress,
        error,
        imageUrl,
    } = useImageGenJob({
        onSuccess: (job) => {
            console.log('Image generation completed:', job);
        },
        onError: (err) => {
            console.error('Image generation failed:', err);
        },
    });

    // Handlers
    const handleOptimize = useCallback(async () => {
        if (!prompt.trim()) return;
        setIsOptimizing(true);
        try {
            const optimized = await optimizePrompt(prompt);
            setPrompt(optimized);
        } catch (e) {
            console.error('Prompt optimization failed:', e);
        } finally {
            setIsOptimizing(false);
        }
    }, [prompt]);

    const handleSubmit = useCallback(async () => {
        if (!prompt.trim() || isSubmitting || isPolling) return;
        await submit(prompt, params);
    }, [prompt, params, isSubmitting, isPolling, submit]);

    const handleCancel = useCallback(async () => {
        await cancel();
    }, [cancel]);

    const handleClear = useCallback(() => {
        clear();
        setPrompt('');
    }, [clear]);

    const handleSizePreset = useCallback((preset: typeof SIZE_PRESETS[0]) => {
        setParams((prev) => ({ ...prev, width: preset.width, height: preset.height }));
    }, []);

    const isRunning = isSubmitting || isPolling;

    return (
        <div className="animate-fade-in">
            <h1 className="text-3xl font-bold mb-2">🖼️ Image Studio</h1>
            <p className="text-[var(--color-text-secondary)] mb-8">
                图像生成工具 - 基于远程 Z-Image API
            </p>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Left: Input Panel */}
                <div className="space-y-6">
                    {/* Prompt Card */}
                    <div className="card">
                        <h3 className="font-semibold mb-4">Prompt</h3>
                        <textarea
                            value={prompt}
                            onChange={(e) => setPrompt(e.target.value)}
                            placeholder="描述你想生成的图像..."
                            className="w-full h-32 p-3 rounded-lg bg-[var(--color-bg-tertiary)] border border-[var(--color-border)] resize-none mb-4"
                            disabled={isRunning}
                        />
                        <div className="flex gap-2">
                            <button
                                onClick={handleOptimize}
                                disabled={isOptimizing || isRunning || !prompt.trim()}
                                className="btn-secondary flex-1"
                            >
                                {isOptimizing ? '优化中...' : '✨ AI 优化 Prompt'}
                            </button>
                        </div>
                    </div>

                    {/* Size Presets */}
                    <div className="card">
                        <h3 className="font-semibold mb-4">尺寸</h3>
                        <div className="flex flex-wrap gap-2 mb-4">
                            {SIZE_PRESETS.map((preset) => (
                                <button
                                    key={preset.label}
                                    onClick={() => handleSizePreset(preset)}
                                    disabled={isRunning}
                                    className={`px-3 py-1 rounded-lg text-sm transition-colors ${params.width === preset.width && params.height === preset.height
                                            ? 'bg-[var(--color-primary)] text-white'
                                            : 'bg-[var(--color-bg-tertiary)] hover:bg-[var(--color-bg-secondary)]'
                                        }`}
                                >
                                    {preset.label}
                                </button>
                            ))}
                        </div>
                        <div className="text-sm text-[var(--color-text-muted)]">
                            当前: {params.width} × {params.height}
                        </div>
                    </div>

                    {/* Advanced Parameters */}
                    <div className="card">
                        <button
                            onClick={() => setShowAdvanced(!showAdvanced)}
                            className="w-full flex items-center justify-between font-semibold"
                        >
                            <span>高级参数</span>
                            <span className="text-[var(--color-text-muted)]">
                                {showAdvanced ? '▲' : '▼'}
                            </span>
                        </button>

                        {showAdvanced && (
                            <div className="mt-4 space-y-4">
                                {/* Inference Steps */}
                                <div>
                                    <label className="block text-sm mb-1">
                                        推理步数 ({params.num_inference_steps})
                                    </label>
                                    <input
                                        type="range"
                                        min={1}
                                        max={50}
                                        value={params.num_inference_steps}
                                        onChange={(e) =>
                                            setParams((prev) => ({
                                                ...prev,
                                                num_inference_steps: Number(e.target.value),
                                            }))
                                        }
                                        disabled={isRunning}
                                        className="w-full"
                                    />
                                </div>

                                {/* Guidance Scale */}
                                <div>
                                    <label className="block text-sm mb-1">
                                        引导强度 ({params.guidance_scale?.toFixed(1)})
                                    </label>
                                    <input
                                        type="range"
                                        min={0}
                                        max={20}
                                        step={0.5}
                                        value={params.guidance_scale}
                                        onChange={(e) =>
                                            setParams((prev) => ({
                                                ...prev,
                                                guidance_scale: Number(e.target.value),
                                            }))
                                        }
                                        disabled={isRunning}
                                        className="w-full"
                                    />
                                </div>

                                {/* Seed */}
                                <div>
                                    <label className="block text-sm mb-1">随机种子</label>
                                    <div className="flex gap-2">
                                        <input
                                            type="number"
                                            value={params.seed ?? ''}
                                            onChange={(e) =>
                                                setParams((prev) => ({
                                                    ...prev,
                                                    seed: e.target.value ? Number(e.target.value) : undefined,
                                                }))
                                            }
                                            placeholder="留空随机"
                                            disabled={isRunning}
                                            className="flex-1 px-3 py-2 rounded-lg bg-[var(--color-bg-tertiary)] border border-[var(--color-border)]"
                                        />
                                        <button
                                            onClick={() =>
                                                setParams((prev) => ({
                                                    ...prev,
                                                    seed: Math.floor(Math.random() * 2147483647),
                                                }))
                                            }
                                            disabled={isRunning}
                                            className="btn-secondary"
                                        >
                                            🎲
                                        </button>
                                    </div>
                                </div>

                                {/* Negative Prompt */}
                                <div>
                                    <label className="block text-sm mb-1">负面提示词</label>
                                    <textarea
                                        value={params.negative_prompt ?? ''}
                                        onChange={(e) =>
                                            setParams((prev) => ({
                                                ...prev,
                                                negative_prompt: e.target.value || undefined,
                                            }))
                                        }
                                        placeholder="不希望出现的内容..."
                                        disabled={isRunning}
                                        className="w-full h-20 p-3 rounded-lg bg-[var(--color-bg-tertiary)] border border-[var(--color-border)] resize-none"
                                    />
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Action Buttons */}
                    <div className="flex gap-3">
                        {!isRunning ? (
                            <button
                                onClick={handleSubmit}
                                disabled={!prompt.trim()}
                                className="btn-primary flex-1 py-3 text-lg"
                            >
                                🎨 生成图像
                            </button>
                        ) : (
                            <button
                                onClick={handleCancel}
                                className="btn-secondary flex-1 py-3 text-lg"
                            >
                                ✕ 取消生成
                            </button>
                        )}
                    </div>
                </div>

                {/* Right: Result Panel */}
                <div className="card">
                    <div className="flex items-center justify-between mb-4">
                        <h3 className="font-semibold">结果</h3>
                        {job && (
                            <button
                                onClick={handleClear}
                                className="text-sm text-[var(--color-text-muted)] hover:text-[var(--color-text-primary)]"
                            >
                                清除
                            </button>
                        )}
                    </div>

                    {/* Status Display */}
                    {job && (
                        <div className="mb-4">
                            <div className="flex items-center justify-between text-sm mb-2">
                                <span className="text-[var(--color-text-muted)]">
                                    状态: <span className="font-mono">{status}</span>
                                </span>
                                {isPolling && (
                                    <span className="text-[var(--color-primary)]">
                                        {getStatusMessage(job)}
                                    </span>
                                )}
                            </div>

                            {/* Progress Bar */}
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

                    {/* Error Display */}
                    {error && (
                        <div className="mb-4 p-3 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400">
                            {error}
                        </div>
                    )}

                    {/* Image Display */}
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
            </div>
        </div>
    );
}
