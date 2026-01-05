import { useState, useEffect, useCallback, useMemo } from 'react';
import { useTTSJob } from '../../hooks/useTTSJob';
import { useCapabilityHealth } from '../../hooks/useCapabilityHealth';
import { useJobHistory } from '../../hooks/useJobHistory';
import type { SpeakerProfile } from '../../types';
import {
    createSpeakerProfile,
    deleteSpeakerProfile,
    listSpeakerProfiles,
} from '../../services/tts';
import { TTSStudioContext } from './context';
import { SpeakerProfileList } from './SpeakerProfileList';
import { SpeakerProfileDetail } from './SpeakerProfileDetail';
import { SynthesisPanel } from './SynthesisPanel';
import { ResultPanel } from './ResultPanel';
import { UploadProfileModal } from './UploadProfileModal';
import { JobHistoryPanel } from '../../components/JobHistoryPanel';
import { StudioHeader } from '../../components';

const HISTORY_KEY = 'tsn_tts_history';

export function TTSStudio() {
    const { reportFailure, reportSuccess } = useCapabilityHealth();
    const { history, addItem, removeItem, clearHistory } = useJobHistory(HISTORY_KEY);

    const { state: jobState, submit, reset } = useTTSJob(
        2000,
        (state) => {
            reportSuccess('tts');
            addItem({
                jobId: state.jobId!,
                createdAt: Date.now(),
                status: 'success',
                inputs: { text },
                outputUrl: state.outputUrl || undefined,
            });
        },
        (err) => {
            reportFailure('tts', err.message);
            if (jobState.jobId) {
                addItem({
                    jobId: jobState.jobId,
                    createdAt: Date.now(),
                    status: 'failed',
                    inputs: { text },
                    error: err.message,
                });
            }
        }
    );

    const [profiles, setProfiles] = useState<SpeakerProfile[]>([]);
    const [selectedProfile, setSelectedProfile] = useState<SpeakerProfile | null>(null);
    const [isLoadingProfiles, setIsLoadingProfiles] = useState(true);

    const [text, setText] = useState('');

    const [showUpload, setShowUpload] = useState(false);

    const refreshProfiles = useCallback(async () => {
        setIsLoadingProfiles(true);
        try {
            const data = await listSpeakerProfiles();
            setProfiles(data);
            const defaultProfile = data.find((p) => p.is_default);
            if (defaultProfile && !selectedProfile) {
                setSelectedProfile(defaultProfile);
            }
        } catch (e) {
            console.error(e);
        } finally {
            setIsLoadingProfiles(false);
        }
    }, [selectedProfile]);

    useEffect(() => {
        refreshProfiles();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    const handleSubmit = useCallback(async () => {
        if (!selectedProfile || !text.trim()) return;
        await submit({
            text: text.trim(),
            profile_id: selectedProfile.id,
        });
    }, [selectedProfile, text, submit]);

    const handleCreateProfile = useCallback(
        async (data: {
            name: string;
            description?: string;
            promptText: string;
            file: File;
            provider: string;
            sampleRate: number;
        }) => {
            const newProfile = await createSpeakerProfile({
                name: data.name,
                description: data.description,
                prompt_text: data.promptText,
                prompt_audio: data.file,
                provider: data.provider,
                sample_rate: data.sampleRate,
                is_default: profiles.length === 0,
            });
            setProfiles((prev) => [newProfile, ...prev]);
            setSelectedProfile(newProfile);
            setShowUpload(false);
        },
        [profiles.length]
    );

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

    const handleHistorySelect = useCallback((item: { inputs: Record<string, unknown> }) => {
        if (item.inputs.text) {
            setText(String(item.inputs.text));
        }
    }, []);

    const contextValue = useMemo(
        () => ({
            profiles,
            selectedProfile,
            setSelectedProfile,
            refreshProfiles,
            isLoadingProfiles,
        }),
        [profiles, selectedProfile, refreshProfiles, isLoadingProfiles]
    );

    const canSubmit = Boolean(selectedProfile && text.trim());

    return (
        <TTSStudioContext.Provider value={contextValue}>
            <div className="animate-fade-in">
                <StudioHeader
                    title="🗣️ TTS Studio"
                    description="语音合成工具 · 选择说话人配置即可快速合成"
                />

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    {/* 左侧：配置列表 */}
                    <div className="space-y-4">
                        <SpeakerProfileList onCreateClick={() => setShowUpload(true)} />
                        <JobHistoryPanel
                            history={history}
                            onSelect={handleHistorySelect}
                            onRemove={removeItem}
                            onClear={clearHistory}
                            title="合成历史"
                            emptyText="暂无合成记录"
                        />
                    </div>

                    {/* 中间：配置详情 + 合成 */}
                    <div className="space-y-4">
                        <SpeakerProfileDetail onDelete={handleDeleteSelected} />
                        <SynthesisPanel
                            text={text}
                            setText={setText}
                            onSubmit={handleSubmit}
                            isSubmitting={jobState.isSubmitting}
                            canSubmit={canSubmit}
                        />
                    </div>

                    {/* 右侧：结果 */}
                    <ResultPanel jobState={jobState} onReset={reset} />
                </div>

                <UploadProfileModal
                    isOpen={showUpload}
                    onClose={() => setShowUpload(false)}
                    onCreate={handleCreateProfile}
                    isFirstProfile={profiles.length === 0}
                />
            </div>
        </TTSStudioContext.Provider>
    );
}
