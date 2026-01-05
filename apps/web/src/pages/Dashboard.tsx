import { Link } from 'react-router-dom';
import { useCapabilityHealth } from '../hooks/useCapabilityHealth';
import { CapabilityCard } from '../components/CapabilityCard';
import type { CapabilityName } from '../services/capabilities';

const tools = [
    { path: '/tools/tts', icon: '🗣️', name: 'TTS 语音合成', desc: '将文本转换为语音' },
    { path: '/tools/imagegen', icon: '🖼️', name: '图像生成', desc: 'AI 生成图像' },
    { path: '/tools/i2v', icon: '🎬', name: '视频生成', desc: '图像转视频' },
    { path: '/tools/llm', icon: '💬', name: 'LLM Studio', desc: '对话测试与提示词优化' },
];

const CAPABILITY_LABELS: Record<CapabilityName, string> = {
    tts: 'TTS 语音合成',
    imagegen: '图像生成',
    videogen: '视频生成',
    llm: 'LLM 对话',
};

export function Dashboard() {
    const { health, isProbing, refreshFromProbe } = useCapabilityHealth();

    return (
        <div className="animate-fade-in">
            <h1 className="text-3xl font-bold mb-2">仪表板</h1>
            <p className="text-[var(--color-text-secondary)] mb-8">
                欢迎使用 Turbo Super Novel
            </p>

            <section className="mb-8">
                <h2 className="text-xl font-semibold mb-4">🛠️ 工具</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                    {tools.map((t) => (
                        <Link
                            key={t.path}
                            to={t.path}
                            className="card hover:border-[var(--color-accent-primary)] hover:scale-[1.02] transition-all"
                        >
                            <div className="flex items-center gap-4">
                                <span className="text-3xl">{t.icon}</span>
                                <div>
                                    <h3 className="font-semibold">{t.name}</h3>
                                    <p className="text-sm text-[var(--color-text-muted)]">
                                        {t.desc}
                                    </p>
                                </div>
                            </div>
                        </Link>
                    ))}
                </div>
            </section>

            <section className="mb-8">
                <div className="flex items-center justify-between mb-4">
                    <h2 className="text-xl font-semibold">⚡ 能力状态</h2>
                    <button
                        onClick={() => refreshFromProbe()}
                        disabled={isProbing}
                        className="text-sm text-[var(--color-text-muted)] hover:text-[var(--color-accent-primary)] transition-colors"
                    >
                        {isProbing ? '刷新中...' : '↻ 刷新'}
                    </button>
                </div>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    {(Object.keys(CAPABILITY_LABELS) as CapabilityName[]).map((cap) => (
                        <CapabilityCard
                            key={cap}
                            label={CAPABILITY_LABELS[cap]}
                            status={health[cap].status}
                            provider={health[cap].provider}
                            lastError={health[cap].lastError}
                            lastSuccess={health[cap].lastSuccess}
                        />
                    ))}
                </div>
            </section>

            <section>
                <h2 className="text-xl font-semibold mb-4">📁 项目</h2>
                <Link to="/projects" className="btn-secondary">
                    查看所有项目 →
                </Link>
            </section>
        </div>
    );
}

