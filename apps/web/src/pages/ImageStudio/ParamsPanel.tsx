import type { ImageGenParams } from '../../types';
import type { SizePreset } from './types';

interface ParamsPanelProps {
    params: Omit<ImageGenParams, 'prompt'>;
    isRunning: boolean;
    showAdvanced: boolean;
    onToggleAdvanced: () => void;
    onParamsChange: (next: Omit<ImageGenParams, 'prompt'>) => void;
    sizePresets: SizePreset[];
    onSizePreset: (preset: SizePreset) => void;
    onRandomSeed: () => void;
}

export function ParamsPanel({
    params,
    isRunning,
    showAdvanced,
    onToggleAdvanced,
    onParamsChange,
    sizePresets,
    onSizePreset,
    onRandomSeed,
}: ParamsPanelProps) {
    return (
        <>
            <div className="card">
                <h3 className="font-semibold mb-4">尺寸</h3>
                <div className="flex flex-wrap gap-2 mb-4">
                    {sizePresets.map((preset) => (
                        <button
                            key={preset.label}
                            onClick={() => onSizePreset(preset)}
                            disabled={isRunning}
                            className={`px-3 py-1 rounded-lg text-sm transition-colors ${
                                params.width === preset.width && params.height === preset.height
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

            <div className="card">
                <button
                    onClick={onToggleAdvanced}
                    className="w-full flex items-center justify-between font-semibold"
                >
                    <span>高级参数</span>
                    <span className="text-[var(--color-text-muted)]">
                        {showAdvanced ? '▲' : '▼'}
                    </span>
                </button>

                {showAdvanced && (
                    <div className="mt-4 space-y-4">
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
                                    onParamsChange({
                                        ...params,
                                        num_inference_steps: Number(e.target.value),
                                    })
                                }
                                disabled={isRunning}
                                className="w-full"
                            />
                        </div>

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
                                    onParamsChange({
                                        ...params,
                                        guidance_scale: Number(e.target.value),
                                    })
                                }
                                disabled={isRunning}
                                className="w-full"
                            />
                        </div>

                        <div>
                            <label className="block text-sm mb-1">随机种子</label>
                            <div className="flex gap-2">
                                <input
                                    type="number"
                                    value={params.seed ?? ''}
                                    onChange={(e) =>
                                        onParamsChange({
                                            ...params,
                                            seed: e.target.value
                                                ? Number(e.target.value)
                                                : undefined,
                                        })
                                    }
                                    placeholder="留空随机"
                                    disabled={isRunning}
                                    className="flex-1 px-3 py-2 rounded-lg bg-[var(--color-bg-tertiary)] border border-[var(--color-border)]"
                                />
                                <button
                                    onClick={onRandomSeed}
                                    disabled={isRunning}
                                    className="btn-secondary"
                                >
                                    🎲
                                </button>
                            </div>
                        </div>

                        <div>
                            <label className="block text-sm mb-1">负面提示词</label>
                            <textarea
                                value={params.negative_prompt ?? ''}
                                onChange={(e) =>
                                    onParamsChange({
                                        ...params,
                                        negative_prompt: e.target.value || undefined,
                                    })
                                }
                                placeholder="不希望出现的内容..."
                                disabled={isRunning}
                                className="w-full h-20 p-3 rounded-lg bg-[var(--color-bg-tertiary)] border border-[var(--color-border)] resize-none"
                            />
                        </div>
                    </div>
                )}
            </div>
        </>
    );
}
