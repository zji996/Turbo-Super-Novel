import { NavLink } from 'react-router-dom';

interface NavItem {
    path: string;
    label: string;
    icon: string;
    disabled?: boolean;
}

const navItems: NavItem[] = [
    { path: '/tools/i2v', label: 'I2V Studio', icon: '🎬' },
    { path: '/projects', label: 'Projects', icon: '📁', disabled: true },
    { path: '/assets', label: 'Assets', icon: '🖼️', disabled: true },
];

export function Sidebar() {
    return (
        <aside className="fixed left-0 top-0 h-screen w-64 bg-[var(--color-bg-secondary)] border-r border-[var(--color-border)] flex flex-col">
            {/* Logo */}
            <div className="p-6 border-b border-[var(--color-border)]">
                <h1 className="text-xl font-bold bg-gradient-to-r from-[var(--color-accent-primary)] to-[var(--color-accent-secondary)] bg-clip-text text-transparent">
                    Turbo Novel
                </h1>
                <p className="text-sm text-[var(--color-text-muted)] mt-1">
                    Video Generation Platform
                </p>
            </div>

            {/* Navigation */}
            <nav className="flex-1 p-4">
                <div className="mb-4">
                    <h2 className="text-xs font-semibold text-[var(--color-text-muted)] uppercase tracking-wider px-3 mb-2">
                        Tools
                    </h2>
                    <ul className="space-y-1">
                        {navItems.map((item) => (
                            <li key={item.path}>
                                <NavLink
                                    to={item.disabled ? '#' : item.path}
                                    onClick={(e) => item.disabled && e.preventDefault()}
                                    className={({ isActive }) =>
                                        `flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all duration-200 ${item.disabled
                                            ? 'opacity-40 cursor-not-allowed'
                                            : isActive
                                                ? 'bg-[var(--color-accent-primary)]/20 text-[var(--color-accent-primary)] shadow-[inset_0_0_20px_rgba(168,85,247,0.1)]'
                                                : 'hover:bg-[var(--color-bg-tertiary)] text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)]'
                                        }`
                                    }
                                >
                                    <span className="text-lg">{item.icon}</span>
                                    <span className="font-medium">{item.label}</span>
                                    {item.disabled && (
                                        <span className="ml-auto text-xs px-2 py-0.5 rounded-full bg-[var(--color-border)] text-[var(--color-text-muted)]">
                                            Soon
                                        </span>
                                    )}
                                </NavLink>
                            </li>
                        ))}
                    </ul>
                </div>
            </nav>

            {/* Footer */}
            <div className="p-4 border-t border-[var(--color-border)]">
                <div className="text-xs text-[var(--color-text-muted)]">
                    <p>MVP v0.1.0</p>
                </div>
            </div>
        </aside>
    );
}
