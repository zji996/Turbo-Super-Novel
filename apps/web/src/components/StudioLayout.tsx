import type { ReactNode } from 'react';

interface StudioLayoutProps {
    title: string;
    icon: string;
    description: string;
    sidebar?: ReactNode;
    main: ReactNode;
    result?: ReactNode;
    headerExtra?: ReactNode;
}

export function StudioLayout({
    title,
    icon,
    description,
    sidebar,
    main,
    result,
    headerExtra,
}: StudioLayoutProps) {
    // Determine grid layout based on which slots are populated
    const hasSidebar = Boolean(sidebar);
    const hasResult = Boolean(result);

    let gridClass = 'grid grid-cols-1 gap-6';
    if (hasSidebar && hasResult) {
        gridClass = 'grid grid-cols-1 lg:grid-cols-3 gap-6';
    } else if (hasSidebar || hasResult) {
        gridClass = 'grid grid-cols-1 lg:grid-cols-2 gap-6';
    }

    return (
        <div className="animate-fade-in">
            {/* Header */}
            <div className="flex items-center justify-between mb-6">
                <div>
                    <h1 className="text-3xl font-bold mb-1">
                        {icon} {title}
                    </h1>
                    <p className="text-[var(--color-text-secondary)]">{description}</p>
                </div>
                {headerExtra && <div>{headerExtra}</div>}
            </div>

            {/* Content Grid */}
            <div className={gridClass}>
                {sidebar && <div className="space-y-4">{sidebar}</div>}
                <div className={`space-y-4 ${!hasSidebar && hasResult ? 'lg:col-span-1' : hasSidebar && hasResult ? '' : 'lg:col-span-1'}`}>
                    {main}
                </div>
                {result && <div className="space-y-4">{result}</div>}
            </div>
        </div>
    );
}
