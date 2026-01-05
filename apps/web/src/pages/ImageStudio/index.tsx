import { useCallback, useMemo, useRef, useState } from 'react';
import { useImageGenJob } from '../../hooks/useImageGenJob';
import { useCapabilityHealth } from '../../hooks/useCapabilityHealth';
import { useJobHistory } from '../../hooks/useJobHistory';
import { getStatusMessage } from '../../services/imagegen';
import type { ImageGenParams } from '../../types';
import { ParamsPanel } from './ParamsPanel';
import { PromptInput } from './PromptInput';
import { ResultDisplay } from './ResultDisplay';
import { JobHistoryPanel } from '../../components/JobHistoryPanel';
import { StudioHeader, SubmitButton } from '../../components';
import type { SizePreset } from './types';

type ImageGenParamsState = Omit<ImageGenParams, 'prompt'>;

const HISTORY_KEY = 'tsn_imagegen_history';

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
    const [params, setParams] = useState<ImageGenParamsState>(DEFAULT_PARAMS);
    const [showAdvanced, setShowAdvanced] = useState(false);

    // Keep a ref to the current prompt for use in callback closures
    const promptRef = useRef(prompt);
    promptRef.current = prompt;

    const { reportFailure, reportSuccess } = useCapabilityHealth();
    const { history, addItem, removeItem, clearHistory } = useJobHistory(HISTORY_KEY);

    const { job, isSubmitting, isPolling, submit, cancel, clear, status, progress, error, imageUrl } =
        useImageGenJob({
            onSuccess: (job) => {
                console.log('Image generation completed:', job);
                reportSuccess('imagegen');
                addItem({
                    jobId: job.job_id,
                    createdAt: Date.now(),
                    status: 'success',
                    inputs: { prompt: promptRef.current },
                    outputUrl: job.image_url,
                });
            },
            onError: (err, job) => {
                console.error('Image generation failed:', err);
                reportFailure('imagegen', err.message);
                if (job) {
                    addItem({
                        jobId: job.job_id,
                        createdAt: Date.now(),
                        status: 'failed',
                        inputs: { prompt: promptRef.current },
                        error: err.message,
                    });
                }
            },
        });

    const isRunning = isSubmitting || isPolling;
    const statusMessage = useMemo(() => (job ? getStatusMessage(job) : null), [job]);

    const handleSubmit = useCallback(async () => {
        if (!prompt.trim() || isRunning) return;
        await submit(prompt, params);
    }, [prompt, params, isRunning, submit]);

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

    const handleHistorySelect = useCallback((item: { inputs: Record<string, unknown> }) => {
        if (item.inputs.prompt) {
            setPrompt(String(item.inputs.prompt));
        }
    }, []);

    return (
        <div className="animate-fade-in">
            <StudioHeader
                title="🖼️ Image Studio"
                description="图像生成工具 - 基于远程 Z-Image API"
            />

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div className="lg:col-span-2 space-y-6">
                    <PromptInput
                        prompt={prompt}
                        onChange={setPrompt}
                        disabled={isRunning}
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
                            <SubmitButton
                                onClick={handleSubmit}
                                disabled={!prompt.trim()}
                                isLoading={isSubmitting}
                                loadingText="提交中..."
                                className="flex-1 py-3 text-lg"
                            >
                                🎨 生成图像
                            </SubmitButton>
                        ) : (
                            <SubmitButton
                                onClick={handleCancel}
                                variant="secondary"
                                className="flex-1 py-3 text-lg"
                            >
                                ✕ 取消生成
                            </SubmitButton>
                        )}
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

                <div className="space-y-4">
                    <JobHistoryPanel
                        history={history}
                        onSelect={handleHistorySelect}
                        onRemove={removeItem}
                        onClear={clearHistory}
                        title="生成历史"
                        emptyText="暂无生成记录"
                    />
                </div>
            </div>
        </div>
    );
}

