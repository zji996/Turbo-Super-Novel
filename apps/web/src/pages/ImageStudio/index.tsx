import { useCallback, useMemo, useState } from 'react';
import { useImageGenJob } from '../../hooks/useImageGenJob';
import { useCapabilityHealth } from '../../hooks/useCapabilityHealth';
import { optimizePrompt } from '../../services/llm';
import { getStatusMessage } from '../../services/imagegen';
import type { ImageGenParams } from '../../types';
import { ParamsPanel } from './ParamsPanel';
import { PromptInput } from './PromptInput';
import { ResultDisplay } from './ResultDisplay';
import type { SizePreset } from './types';

type ImageGenParamsState = Omit<ImageGenParams, 'prompt'>;

const DEFAULT_PARAMS: ImageGenParamsState = {
    width: 1024,
    height: 1024,
    num_inference_steps: 9,
    guidance_scale: 0.0,
};

const SIZE_PRESETS: SizePreset[] = [
    { label: '1:1 方形', width: 1024, height: 1024 },
    { label: '3:4 竖版', width: 768, height: 1024 },
    { label: '4:3 横版', width: 1024, height: 768 },
    { label: '16:9 宽屏', width: 1024, height: 576 },
    { label: '9:16 竖屏', width: 576, height: 1024 },
];

export function ImageStudio() {
    const [prompt, setPrompt] = useState('');
    const [isOptimizing, setIsOptimizing] = useState(false);
    const [enhancePrompt, setEnhancePrompt] = useState(false);
    const [params, setParams] = useState<ImageGenParamsState>(DEFAULT_PARAMS);
    const [showAdvanced, setShowAdvanced] = useState(false);

    const { reportFailure, reportSuccess } = useCapabilityHealth();

    const { job, isSubmitting, isPolling, submit, cancel, clear, status, progress, error, imageUrl } =
        useImageGenJob({
            onSuccess: (job) => {
                console.log('Image generation completed:', job);
                reportSuccess('imagegen');
            },
            onError: (err) => {
                console.error('Image generation failed:', err);
                reportFailure('imagegen', err.message);
            },
        });

    const isRunning = isSubmitting || isPolling;
    const statusMessage = useMemo(() => (job ? getStatusMessage(job) : null), [job]);

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
        if (!prompt.trim() || isRunning) return;
        await submit(prompt, { ...params, enhance_prompt: enhancePrompt });
    }, [prompt, params, enhancePrompt, isRunning, submit]);

    const handleCancel = useCallback(async () => {
        await cancel();
    }, [cancel]);

    const handleClear = useCallback(() => {
        clear();
        setPrompt('');
    }, [clear]);

    const handleSizePreset = useCallback((preset: SizePreset) => {
        setParams((prev) => ({ ...prev, width: preset.width, height: preset.height }));
    }, []);

    const handleRandomSeed = useCallback(() => {
        setParams((prev) => ({ ...prev, seed: Math.floor(Math.random() * 2147483647) }));
    }, []);

    return (
        <div className="animate-fade-in">
            <h1 className="text-3xl font-bold mb-2">🖼️ Image Studio</h1>
            <p className="text-[var(--color-text-secondary)] mb-8">
                图像生成工具 - 基于远程 Z-Image API
            </p>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <div className="space-y-6">
                    <PromptInput
                        prompt={prompt}
                        isRunning={isRunning}
                        isOptimizing={isOptimizing}
                        enhancePrompt={enhancePrompt}
                        onChange={setPrompt}
                        onOptimize={handleOptimize}
                        onEnhancePromptChange={setEnhancePrompt}
                    />

                    <ParamsPanel
                        params={params}
                        isRunning={isRunning}
                        showAdvanced={showAdvanced}
                        onToggleAdvanced={() => setShowAdvanced((prev) => !prev)}
                        onParamsChange={setParams}
                        sizePresets={SIZE_PRESETS}
                        onSizePreset={handleSizePreset}
                        onRandomSeed={handleRandomSeed}
                    />

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

                <ResultDisplay
                    job={job}
                    status={status}
                    progress={progress}
                    isPolling={isPolling}
                    error={error}
                    imageUrl={imageUrl}
                    statusMessage={statusMessage}
                    onClear={handleClear}
                />
            </div>
        </div>
    );
}
