import { useState, useCallback } from 'react';
import { Modal } from '../../components/ui';

interface UploadProfileModalProps {
    isOpen: boolean;
    onClose: () => void;
    onCreate: (data: {
        name: string;
        description?: string;
        promptText: string;
        file: File;
        provider: string;
        sampleRate: number;
    }) => Promise<void>;
    isFirstProfile: boolean;
}

export function UploadProfileModal({
    isOpen,
    onClose,
    onCreate,
    isFirstProfile,
}: UploadProfileModalProps) {
    const [name, setName] = useState('');
    const [description, setDescription] = useState('');
    const [promptText, setPromptText] = useState('');
    const [file, setFile] = useState<File | null>(null);
    const [provider, setProvider] = useState('glm_tts');
    const [sampleRate, setSampleRate] = useState(24000);
    const [isUploading, setIsUploading] = useState(false);

    const resetForm = useCallback(() => {
        setName('');
        setDescription('');
        setPromptText('');
        setFile(null);
        setProvider('glm_tts');
        setSampleRate(24000);
    }, []);

    const handleClose = useCallback(() => {
        resetForm();
        onClose();
    }, [onClose, resetForm]);

    const handleSubmit = useCallback(async () => {
        if (!file || !name || !promptText) return;
        setIsUploading(true);
        try {
            await onCreate({
                name,
                description: description || undefined,
                promptText,
                file,
                provider,
                sampleRate,
            });
            resetForm();
        } catch (e) {
            console.error(e);
        } finally {
            setIsUploading(false);
        }
    }, [file, name, description, promptText, provider, sampleRate, onCreate, resetForm]);

    const canSubmit = name.trim() && promptText.trim() && file && !isUploading;

    return (
        <Modal isOpen={isOpen} onClose={handleClose} title="创建说话人配置">
            <div className="space-y-3">
                <div>
                    <label className="text-sm text-[var(--color-text-muted)]">名称 *</label>
                    <input
                        placeholder="如：温柔女声、激情男声"
                        value={name}
                        onChange={(e) => setName(e.target.value)}
                        className="input w-full mt-1"
                    />
                </div>
                <div>
                    <label className="text-sm text-[var(--color-text-muted)]">描述</label>
                    <input
                        placeholder="可选，简单描述这个配置"
                        value={description}
                        onChange={(e) => setDescription(e.target.value)}
                        className="input w-full mt-1"
                    />
                </div>
                <div>
                    <label className="text-sm text-[var(--color-text-muted)]">参考文本 *</label>
                    <textarea
                        placeholder="参考音频中说的文字内容"
                        value={promptText}
                        onChange={(e) => setPromptText(e.target.value)}
                        className="input w-full mt-1 h-20 resize-none"
                    />
                </div>
                <div>
                    <label className="text-sm text-[var(--color-text-muted)]">参考音频 * (WAV)</label>
                    <input
                        type="file"
                        accept=".wav"
                        onChange={(e) => setFile(e.target.files?.[0] || null)}
                        className="mt-1 w-full"
                    />
                </div>
                <div className="grid grid-cols-2 gap-3">
                    <div>
                        <label className="text-sm text-[var(--color-text-muted)]">Provider</label>
                        <select
                            value={provider}
                            onChange={(e) => setProvider(e.target.value)}
                            className="input w-full mt-1"
                        >
                            <option value="glm_tts">GLM TTS</option>
                        </select>
                    </div>
                    <div>
                        <label className="text-sm text-[var(--color-text-muted)]">采样率</label>
                        <select
                            value={sampleRate}
                            onChange={(e) => setSampleRate(Number(e.target.value))}
                            className="input w-full mt-1"
                        >
                            <option value={24000}>24kHz</option>
                            <option value={32000}>32kHz</option>
                        </select>
                    </div>
                </div>
                {isFirstProfile && (
                    <p className="text-xs text-[var(--color-text-muted)]">
                        这将是您的第一个配置，将自动设为默认
                    </p>
                )}
            </div>
            <div className="flex gap-2 mt-4">
                <button
                    onClick={handleClose}
                    className="btn-secondary flex-1"
                    disabled={isUploading}
                >
                    取消
                </button>
                <button
                    onClick={handleSubmit}
                    className="btn-primary flex-1"
                    disabled={!canSubmit}
                >
                    {isUploading ? '创建中...' : '创建配置'}
                </button>
            </div>
        </Modal>
    );
}
