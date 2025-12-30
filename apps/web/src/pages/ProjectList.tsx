import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';

const API_BASE = import.meta.env.VITE_API_BASE_URL || '';

interface Project {
    id: string;
    name: string;
    status: string;
    scene_count: number;
    created_at: string;
}

export function ProjectList() {
    const [projects, setProjects] = useState<Project[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetch(`${API_BASE}/v1/novel/projects`)
            .then((r) => r.json())
            .then((d) => setProjects(d.projects || []))
            .finally(() => setLoading(false));
    }, []);

    return (
        <div className="animate-fade-in">
            <div className="flex justify-between items-center mb-8">
                <h1 className="text-3xl font-bold">📁 项目列表</h1>
                <Link to="/projects/new" className="btn-primary">
                    + 新建项目
                </Link>
            </div>

            {loading ? (
                <p>加载中...</p>
            ) : projects.length === 0 ? (
                <p className="text-[var(--color-text-muted)]">暂无项目</p>
            ) : (
                <div className="space-y-4">
                    {projects.map((p) => (
                        <Link
                            key={p.id}
                            to={`/projects/${p.id}`}
                            className="card block hover:border-[var(--color-accent-primary)]"
                        >
                            <div className="flex justify-between items-center">
                                <div>
                                    <h3 className="font-semibold">{p.name}</h3>
                                    <p className="text-sm text-[var(--color-text-muted)]">
                                        {p.scene_count} 场景
                                    </p>
                                </div>
                                <span className="text-sm">{p.status}</span>
                            </div>
                        </Link>
                    ))}
                </div>
            )}
        </div>
    );
}

