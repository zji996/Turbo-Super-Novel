import { useTTSStudioContext } from './context';

interface SpeakerProfileDetailProps {
    onDelete: () => void;
}

export function SpeakerProfileDetail({ onDelete }: SpeakerProfileDetailProps) {
    const { selectedProfile } = useTTSStudioContext();

    if (!selectedProfile) {
        return (
            <div className="card text-center py-8">
                <p className="text-[var(--color-text-muted)]">← 请选择一个说话人配置</p>
            </div>
        );
    }

    return (
        <div className="card">
            <div className="flex items-center justify-between mb-3">
                <h3 className="font-semibold">{selectedProfile.name}</h3>
                <button
                    onClick={onDelete}
                    className="text-sm text-red-500 hover:text-red-400"
                >
                    删除
                </button>
            </div>
            {selectedProfile.description && (
                <p className="text-sm text-[var(--color-text-muted)] mb-3">
                    {selectedProfile.description}
                </p>
            )}
            <div className="flex gap-2 mb-3">
                <span className="px-2 py-1 text-xs bg-[var(--color-bg-tertiary)] rounded">
                    {selectedProfile.provider}
                </span>
                <span className="px-2 py-1 text-xs bg-[var(--color-bg-tertiary)] rounded">
                    {selectedProfile.sample_rate / 1000}kHz
                </span>
            </div>
            <div className="mb-3">
                <p className="text-xs text-[var(--color-text-muted)] mb-1">参考文本</p>
                <p className="text-sm p-2 bg-[var(--color-bg-tertiary)] rounded">
                    {selectedProfile.prompt_text}
                </p>
            </div>
            {selectedProfile.prompt_audio_url && (
                <div>
                    <p className="text-xs text-[var(--color-text-muted)] mb-1">参考音频</p>
                    <audio
                        controls
                        src={selectedProfile.prompt_audio_url}
                        className="w-full h-10"
                    />
                </div>
            )}
        </div>
    );
}
