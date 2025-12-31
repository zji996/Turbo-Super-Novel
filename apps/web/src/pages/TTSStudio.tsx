import { useState, useEffect, useCallback } from 'react';
import { useTTSJob } from '../hooks/useTTSJob';
import { Modal, StatusBadge } from '../components/ui';
import type { SpeakerProfile } from '../types';
import {
    createSpeakerProfile,
    deleteSpeakerProfile,
    listSpeakerProfiles,
} from '../services/tts';

export function TTSStudio() {
    const { state: jobState, submit, reset } = useTTSJob(2000);

    const [profiles, setProfiles] = useState<SpeakerProfile[]>([]);
    const [selectedProfile, setSelectedProfile] = useState<SpeakerProfile | null>(null);
    const [isLoadingProfiles, setIsLoadingProfiles] = useState(true);

    const [text, setText] = useState('');

    const [showUpload, setShowUpload] = useState(false);
    const [uploadName, setUploadName] = useState('');
    const [uploadDescription, setUploadDescription] = useState('');
    const [uploadPromptText, setUploadPromptText] = useState('');
    const [uploadFile, setUploadFile] = useState<File | null>(null);
    const [uploadProvider, setUploadProvider] = useState('glm_tts');
    const [uploadSampleRate, setUploadSampleRate] = useState(24000);
    const [isUploading, setIsUploading] = useState(false);

    useEffect(() => {
        setIsLoadingProfiles(true);
        listSpeakerProfiles()
            .then((data) => {
                setProfiles(data);
                // 自动选择默认配置
                const defaultProfile = data.find((p) => p.is_default);
                if (defaultProfile) {
                    setSelectedProfile(defaultProfile);
                }
            })
            .catch((e) => console.error(e))
            .finally(() => setIsLoadingProfiles(false));
    }, []);

    const handleSubmit = useCallback(async () => {
        if (!selectedProfile || !text.trim()) return;
        await submit({
            text: text.trim(),
            profile_id: selectedProfile.id,
        });
        setText('');
    }, [selectedProfile, text, submit]);

    const handleUpload = useCallback(async () => {
        if (!uploadFile || !uploadName || !uploadPromptText) return;
        setIsUploading(true);
        try {
            const newProfile = await createSpeakerProfile({
                name: uploadName,
                description: uploadDescription || undefined,
                prompt_text: uploadPromptText,
                prompt_audio: uploadFile,
                provider: uploadProvider,
                sample_rate: uploadSampleRate,
                is_default: profiles.length === 0, // 第一个配置设为默认
            });
            setProfiles((prev) => [newProfile, ...prev]);
            setSelectedProfile(newProfile);
            setShowUpload(false);
            setUploadName('');
            setUploadDescription('');
            setUploadPromptText('');
            setUploadFile(null);
            setUploadProvider('glm_tts');
            setUploadSampleRate(24000);
        } catch (e) {
            console.error(e);
        } finally {
            setIsUploading(false);
        }
    }, [uploadFile, uploadName, uploadDescription, uploadPromptText, uploadProvider, uploadSampleRate, profiles.length]);

    const handleDeleteSelected = useCallback(async () => {
        if (!selectedProfile) return;
        if (!window.confirm(`确定删除配置 "${selectedProfile.name}" 吗？`)) return;
        try {
            await deleteSpeakerProfile(selectedProfile.id);
            setProfiles((prev) => prev.filter((p) => p.id !== selectedProfile.id));
            setSelectedProfile(null);
        } catch (e) {
            console.error(e);
        }
    }, [selectedProfile]);

    return (
        <div className="animate-fade-in">
            <h1 className="text-3xl font-bold mb-2">🗣️ TTS Studio</h1>
            <p className="text-[var(--color-text-secondary)] mb-8">语音合成工具 · 选择说话人配置即可快速合成</p>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* 左侧：配置列表 */}
                <div className="space-y-4">
                    <div className="card">
                        <div className="flex items-center justify-between mb-4">
                            <h3 className="font-semibold">说话人配置</h3>
                            <span className="text-sm text-[var(--color-text-muted)]">
                                {profiles.length} 个
                            </span>
                        </div>
                        {isLoadingProfiles ? (
                            <p className="text-[var(--color-text-muted)]">加载中...</p>
                        ) : profiles.length === 0 ? (
                            <p className="text-[var(--color-text-muted)]">暂无配置，请创建一个</p>
                        ) : (
                            <div className="space-y-2 max-h-64 overflow-y-auto">
                                {profiles.map((p) => (
                                    <button
                                        key={p.id}
                                        onClick={() => setSelectedProfile(p)}
                                        className={`w-full text-left p-3 rounded-lg border transition ${selectedProfile?.id === p.id
                                                ? 'border-[var(--color-accent-primary)] bg-[var(--color-accent-primary)]/10'
                                                : 'border-[var(--color-border)] hover:border-[var(--color-text-muted)]'
                                            }`}
                                    >
                                        <div className="flex items-center gap-2">
                                            <span className="font-medium">{p.name}</span>
                                            {p.is_default && (
                                                <span className="px-1.5 py-0.5 text-xs bg-[var(--color-accent-primary)]/20 text-[var(--color-accent-primary)] rounded">
                                                    默认
                                                </span>
                                            )}
                                        </div>
                                        <div className="text-xs text-[var(--color-text-muted)] mt-1">
                                            {p.provider} · {p.sample_rate / 1000}kHz
                                        </div>
                                    </button>
                                ))}
                            </div>
                        )}
                        <button
                            onClick={() => setShowUpload(true)}
                            className="btn-primary mt-4 w-full"
                        >
                            + 创建新配置
                        </button>
                    </div>
                </div>

                {/* 中间：配置详情 + 合成 */}
                <div className="space-y-4">
                    {selectedProfile ? (
                        <div className="card">
                            <div className="flex items-center justify-between mb-3">
                                <h3 className="font-semibold">{selectedProfile.name}</h3>
                                <button
                                    onClick={handleDeleteSelected}
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
                    ) : (
                        <div className="card text-center py-8">
                            <p className="text-[var(--color-text-muted)]">← 请选择一个说话人配置</p>
                        </div>
                    )}

                    <div className="card">
                        <h3 className="font-semibold mb-3">合成文本</h3>
                        <textarea
                            value={text}
                            onChange={(e) => setText(e.target.value)}
                            placeholder="输入要合成的文本..."
                            className="w-full h-28 p-3 rounded-lg bg-[var(--color-bg-tertiary)] border border-[var(--color-border)] resize-none"
                        />
                        <div className="text-xs text-[var(--color-text-muted)] mt-1 text-right">
                            {text.length} / 5000
                        </div>
                    </div>

                    <button
                        onClick={handleSubmit}
                        disabled={!selectedProfile || !text.trim() || jobState.isSubmitting}
                        className="btn-primary w-full py-4 text-lg"
                    >
                        {jobState.isSubmitting ? '提交中...' : '🎤 开始合成'}
                    </button>
                </div>

                {/* 右侧：结果 */}
                <div className="card">
                    <h3 className="font-semibold mb-4">合成结果</h3>
                    {jobState.jobId ? (
                        <div className="space-y-4">
                            <div className="flex items-center gap-2">
                                <span>状态:</span>
                                <StatusBadge status={jobState.status} />
                            </div>
                            {jobState.isPolling && jobState.status !== 'SUCCEEDED' && (
                                <div className="flex items-center gap-2 text-sm text-[var(--color-text-muted)]">
                                    <span className="animate-pulse">●</span>
                                    处理中...
                                </div>
                            )}
                            {jobState.error && (
                                <p className="text-sm text-red-500">{jobState.error}</p>
                            )}
                            {jobState.outputUrl && (
                                <div>
                                    <audio controls src={jobState.outputUrl} className="w-full" />
                                    <a
                                        href={jobState.outputUrl}
                                        download
                                        className="btn-secondary w-full mt-2 text-center block"
                                    >
                                        下载音频
                                    </a>
                                </div>
                            )}
                            <button onClick={reset} className="btn-secondary w-full">
                                重置
                            </button>
                        </div>
                    ) : (
                        <p className="text-[var(--color-text-muted)]">暂无任务</p>
                    )}
                </div>
            </div>

            <Modal
                isOpen={showUpload}
                onClose={() => setShowUpload(false)}
                title="创建说话人配置"
            >
                <div className="space-y-3">
                    <div>
                        <label className="text-sm text-[var(--color-text-muted)]">名称 *</label>
                        <input
                            placeholder="如：温柔女声、激情男声"
                            value={uploadName}
                            onChange={(e) => setUploadName(e.target.value)}
                            className="input w-full mt-1"
                        />
                    </div>
                    <div>
                        <label className="text-sm text-[var(--color-text-muted)]">描述</label>
                        <input
                            placeholder="可选，简单描述这个配置"
                            value={uploadDescription}
                            onChange={(e) => setUploadDescription(e.target.value)}
                            className="input w-full mt-1"
                        />
                    </div>
                    <div>
                        <label className="text-sm text-[var(--color-text-muted)]">参考文本 *</label>
                        <textarea
                            placeholder="参考音频中说的文字内容"
                            value={uploadPromptText}
                            onChange={(e) => setUploadPromptText(e.target.value)}
                            className="input w-full mt-1 h-20 resize-none"
                        />
                    </div>
                    <div>
                        <label className="text-sm text-[var(--color-text-muted)]">参考音频 * (WAV)</label>
                        <input
                            type="file"
                            accept=".wav"
                            onChange={(e) => setUploadFile(e.target.files?.[0] || null)}
                            className="mt-1 w-full"
                        />
                    </div>
                    <div className="grid grid-cols-2 gap-3">
                        <div>
                            <label className="text-sm text-[var(--color-text-muted)]">Provider</label>
                            <select
                                value={uploadProvider}
                                onChange={(e) => setUploadProvider(e.target.value)}
                                className="input w-full mt-1"
                            >
                                <option value="glm_tts">GLM TTS</option>
                            </select>
                        </div>
                        <div>
                            <label className="text-sm text-[var(--color-text-muted)]">采样率</label>
                            <select
                                value={uploadSampleRate}
                                onChange={(e) => setUploadSampleRate(Number(e.target.value))}
                                className="input w-full mt-1"
                            >
                                <option value={24000}>24kHz</option>
                                <option value={32000}>32kHz</option>
                            </select>
                        </div>
                    </div>
                </div>
                <div className="flex gap-2 mt-4">
                    <button
                        onClick={() => setShowUpload(false)}
                        className="btn-secondary flex-1"
                        disabled={isUploading}
                    >
                        取消
                    </button>
                    <button
                        onClick={handleUpload}
                        className="btn-primary flex-1"
                        disabled={!uploadName || !uploadPromptText || !uploadFile || isUploading}
                    >
                        {isUploading ? '创建中...' : '创建配置'}
                    </button>
                </div>
            </Modal>
        </div>
    );
}

