import { useCallback } from 'react';
import type { I2VParams } from '../types';

interface ParamsPanelProps {
    params: I2VParams;
    onChange: (params: I2VParams) => void;
    disabled?: boolean;
}

const NUM_STEPS_OPTIONS = [1, 2, 3, 4];

export function ParamsPanel({ params, onChange, disabled = false }: ParamsPanelProps) {
    const handleSeedChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
        const value = parseInt(e.target.value, 10);
        onChange({ ...params, seed: isNaN(value) ? 0 : value });
    }, [params, onChange]);

    const handleRandomSeed = useCallback(() => {
        onChange({ ...params, seed: Math.floor(Math.random() * 2147483647) });
    }, [params, onChange]);

    const handleNumStepsChange = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => {
        onChange({ ...params, num_steps: parseInt(e.target.value, 10) });
    }, [params, onChange]);

    const handleQuantizedChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
        onChange({ ...params, quantized: e.target.checked });
    }, [params, onChange]);

    return (
        <div className="card">
            <h3 className="text-sm font-semibold text-[var(--color-text-secondary)] uppercase tracking-wider mb-4">
                Parameters
            </h3>

            <div className="space-y-4">
                {/* Num Steps */}
                <div>
                    <label className="block text-sm text-[var(--color-text-secondary)] mb-1.5">
                        Inference Steps
                    </label>
                    <select
                        value={params.num_steps}
                        onChange={handleNumStepsChange}
                        disabled={disabled}
                        className="input"
                    >
                        {NUM_STEPS_OPTIONS.map((n) => (
                            <option key={n} value={n}>
                                {n} step{n > 1 ? 's' : ''} {n === 4 ? '(recommended)' : ''}
                            </option>
                        ))}
                    </select>
                    <p className="text-xs text-[var(--color-text-muted)] mt-1">
                        More steps = better quality, slower generation
                    </p>
                </div>

                {/* Seed */}
                <div>
                    <label className="block text-sm text-[var(--color-text-secondary)] mb-1.5">
                        Seed
                    </label>
                    <div className="flex gap-2">
                        <input
                            type="number"
                            value={params.seed}
                            onChange={handleSeedChange}
                            min={0}
                            disabled={disabled}
                            className="input flex-1"
                        />
                        <button
                            onClick={handleRandomSeed}
                            disabled={disabled}
                            className="btn-secondary px-3"
                            title="Random seed"
                        >
                            🎲
                        </button>
                    </div>
                    <p className="text-xs text-[var(--color-text-muted)] mt-1">
                        Same seed + prompt = reproducible results
                    </p>
                </div>

                {/* Quantized */}
                <div className="flex items-center justify-between">
                    <div>
                        <label className="text-sm text-[var(--color-text-secondary)]">
                            Quantized Model
                        </label>
                        <p className="text-xs text-[var(--color-text-muted)]">
                            Enable for faster inference (recommended)
                        </p>
                    </div>
                    <label className="relative inline-flex items-center cursor-pointer">
                        <input
                            type="checkbox"
                            checked={params.quantized}
                            onChange={handleQuantizedChange}
                            disabled={disabled}
                            className="sr-only peer"
                        />
                        <div className="w-11 h-6 bg-[var(--color-bg-tertiary)] rounded-full peer peer-checked:bg-[var(--color-accent-primary)] transition-colors">
                            <div className={`
                absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full shadow transition-transform
                ${params.quantized ? 'translate-x-5' : 'translate-x-0'}
              `} />
                        </div>
                    </label>
                </div>
            </div>
        </div>
    );
}
