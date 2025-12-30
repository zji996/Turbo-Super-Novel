import { useState, useEffect, useCallback } from 'react';
import {
    listPromptAudios,
    uploadPromptAudio,
    createTTSJob,
    getTTSJobStatus,
    type PromptAudio,
} from '../services/tts';

export function TTSStudio() {
    const [prompts, setPrompts] = useState<PromptAudio[]>([]);
    const [selectedPrompt, setSelectedPrompt] = useState<PromptAudio | null>(null);

    const [text, setText] = useState('');
    const [isSubmitting, setIsSubmitting] = useState(false);

    const [currentJob, setCurrentJob] = useState<{
        id: string;
        status: string;
        output_url?: string;
    } | null>(null);

    const [showUpload, setShowUpload] = useState(false);
    const [uploadName, setUploadName] = useState('');
    const [uploadText, setUploadText] = useState('');
    const [uploadFile, setUploadFile] = useState<File | null>(null);

    useEffect(() => {
        listPromptAudios()
            .then(setPrompts)
            .catch((e) => console.error(e));
    }, []);

    useEffect(() => {
        if (!currentJob || ['SUCCEEDED', 'FAILED'].includes(currentJob.status)) return;

        const interval = setInterval(async () => {
            try {
                const data = await getTTSJobStatus(currentJob.id);
                setCurrentJob({
                    id: currentJob.id,
                    status: data.db?.status || data.celery_status || 'PENDING',
                    output_url: data.output_url,
                });
            } catch (e) {
                console.error(e);
            }
        }, 2000);
        return () => clearInterval(interval);
    }, [currentJob]);

    const handleSubmit = useCallback(async () => {
        if (!selectedPrompt || !text.trim()) return;
        setIsSubmitting(true);
        try {
            const resp = await createTTSJob({
                text: text.trim(),
                prompt_text: selectedPrompt.text,
                prompt_audio_id: selectedPrompt.id,
            });
            setCurrentJob({ id: resp.job_id, status: 'SUBMITTED' });
            setText('');
        } catch (e) {
            console.error(e);
        } finally {
            setIsSubmitting(false);
        }
    }, [selectedPrompt, text]);

    const handleUpload = useCallback(async () => {
        if (!uploadFile || !uploadName || !uploadText) return;
        try {
            const newPrompt = await uploadPromptAudio(uploadName, uploadText, uploadFile);
            setPrompts((prev) => [newPrompt, ...prev]);
            setSelectedPrompt(newPrompt);
            setShowUpload(false);
            setUploadName('');
            setUploadText('');
            setUploadFile(null);
        } catch (e) {
            console.error(e);
        }
    }, [uploadFile, uploadName, uploadText]);

    return (
        <div className="animate-fade-in">
            <h1 className="text-3xl font-bold mb-2">🗣️ TTS Studio</h1>
            <p className="text-[var(--color-text-secondary)] mb-8">语音合成工具</p>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <div className="space-y-6">
                    <div className="card">
                        <h3 className="font-semibold mb-4">参考音频</h3>
                        <div className="space-y-2 max-h-48 overflow-y-auto">
                            {prompts.map((p) => (
                                <button
                                    key={p.id}
                                    onClick={() => setSelectedPrompt(p)}
                                    className={`w-full text-left p-3 rounded-lg border transition ${
                                        selectedPrompt?.id === p.id
                                            ? 'border-[var(--color-accent-primary)] bg-[var(--color-accent-primary)]/10'
                                            : 'border-[var(--color-border)] hover:border-[var(--color-text-muted)]'
                                    }`}
                                >
                                    <div className="font-medium">{p.name}</div>
                                    <div className="text-sm text-[var(--color-text-muted)] truncate">
                                        {p.text}
                                    </div>
                                </button>
                            ))}
                        </div>
                        <button
                            onClick={() => setShowUpload(true)}
                            className="btn-secondary mt-4 w-full"
                        >
                            + 上传新参考音频
                        </button>
                    </div>

                    <div className="card">
                        <h3 className="font-semibold mb-4">合成文本</h3>
                        <textarea
                            value={text}
                            onChange={(e) => setText(e.target.value)}
                            placeholder="输入要合成的文本..."
                            className="w-full h-32 p-3 rounded-lg bg-[var(--color-bg-tertiary)] border border-[var(--color-border)] resize-none"
                        />
                    </div>

                    <button
                        onClick={handleSubmit}
                        disabled={!selectedPrompt || !text.trim() || isSubmitting}
                        className="btn-primary w-full py-4"
                    >
                        {isSubmitting ? '提交中...' : '🎤 开始合成'}
                    </button>
                </div>

                <div className="card">
                    <h3 className="font-semibold mb-4">合成结果</h3>
                    {currentJob ? (
                        <div className="space-y-4">
                            <div>
                                状态: <span className="font-mono">{currentJob.status}</span>
                            </div>
                            {currentJob.output_url && (
                                <audio controls src={currentJob.output_url} className="w-full" />
                            )}
                        </div>
                    ) : (
                        <p className="text-[var(--color-text-muted)]">暂无任务</p>
                    )}
                </div>
            </div>

            {showUpload && (
                <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
                    <div className="card w-full max-w-md">
                        <h3 className="font-semibold mb-4">上传参考音频</h3>
                        <input
                            placeholder="名称"
                            value={uploadName}
                            onChange={(e) => setUploadName(e.target.value)}
                            className="input mb-2 w-full"
                        />
                        <input
                            placeholder="参考文本"
                            value={uploadText}
                            onChange={(e) => setUploadText(e.target.value)}
                            className="input mb-2 w-full"
                        />
                        <input
                            type="file"
                            accept=".wav"
                            onChange={(e) => setUploadFile(e.target.files?.[0] || null)}
                            className="mb-4"
                        />
                        <div className="flex gap-2">
                            <button
                                onClick={() => setShowUpload(false)}
                                className="btn-secondary flex-1"
                            >
                                取消
                            </button>
                            <button onClick={handleUpload} className="btn-primary flex-1">
                                上传
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}

