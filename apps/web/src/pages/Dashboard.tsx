import { Link } from 'react-router-dom';

const tools = [
    { path: '/tools/tts', icon: '🗣️', name: 'TTS 语音合成', desc: '将文本转换为语音' },
    { path: '/tools/imagegen', icon: '🖼️', name: '图像生成', desc: 'AI 生成图像' },
    { path: '/tools/i2v', icon: '🎬', name: '视频生成', desc: '图像转视频' },
];

export function Dashboard() {
    return (
        <div className="animate-fade-in">
            <h1 className="text-3xl font-bold mb-2">仪表板</h1>
            <p className="text-[var(--color-text-secondary)] mb-8">
                欢迎使用 Turbo Super Novel
            </p>

            <section className="mb-8">
                <h2 className="text-xl font-semibold mb-4">🛠️ 工具</h2>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    {tools.map((t) => (
                        <Link
                            key={t.path}
                            to={t.path}
                            className="card hover:border-[var(--color-accent-primary)] transition-colors"
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

            <section>
                <h2 className="text-xl font-semibold mb-4">📁 项目</h2>
                <Link to="/projects" className="btn-secondary">
                    查看所有项目 →
                </Link>
            </section>
        </div>
    );
}

