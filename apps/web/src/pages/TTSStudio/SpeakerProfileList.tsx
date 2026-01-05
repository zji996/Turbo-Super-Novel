import type { SpeakerProfile } from '../../types';
import { useTTSStudioContext } from './context';

export function SpeakerProfileList({ onCreateClick }: { onCreateClick: () => void }) {
    const { profiles, selectedProfile, setSelectedProfile, isLoadingProfiles } = useTTSStudioContext();

    return (
        <div className="card">
            <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold">说话人配置</h3>
                <span className="text-sm text-[var(--color-text-muted)]">{profiles.length} 个</span>
            </div>
            {isLoadingProfiles ? (
                <p className="text-[var(--color-text-muted)]">加载中...</p>
            ) : profiles.length === 0 ? (
                <p className="text-[var(--color-text-muted)]">暂无配置，请创建一个</p>
            ) : (
                <div className="space-y-2 max-h-64 overflow-y-auto">
                    {profiles.map((p) => (
                        <ProfileItem
                            key={p.id}
                            profile={p}
                            isSelected={selectedProfile?.id === p.id}
                            onSelect={() => setSelectedProfile(p)}
                        />
                    ))}
                </div>
            )}
            <button onClick={onCreateClick} className="btn-primary mt-4 w-full">
                + 创建新配置
            </button>
        </div>
    );
}

function ProfileItem({
    profile,
    isSelected,
    onSelect,
}: {
    profile: SpeakerProfile;
    isSelected: boolean;
    onSelect: () => void;
}) {
    return (
        <button
            onClick={onSelect}
            className={`w-full text-left p-3 rounded-lg border transition ${isSelected
                    ? 'border-[var(--color-accent-primary)] bg-[var(--color-accent-primary)]/10'
                    : 'border-[var(--color-border)] hover:border-[var(--color-text-muted)]'
                }`}
        >
            <div className="flex items-center gap-2">
                <span className="font-medium">{profile.name}</span>
                {profile.is_default && (
                    <span className="px-1.5 py-0.5 text-xs bg-[var(--color-accent-primary)]/20 text-[var(--color-accent-primary)] rounded">
                        默认
                    </span>
                )}
            </div>
            <div className="text-xs text-[var(--color-text-muted)] mt-1">
                {profile.provider} · {profile.sample_rate / 1000}kHz
            </div>
        </button>
    );
}
