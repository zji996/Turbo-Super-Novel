import type { ReactNode } from 'react';

export interface StudioHeaderProps {
    title: ReactNode;
    description?: ReactNode;
    action?: ReactNode;
    className?: string;
    titleClassName?: string;
}

export function StudioHeader({
    title,
    description,
    action,
    className,
    titleClassName,
}: StudioHeaderProps) {
    return (
        <div className={className ?? 'mb-8'}>
            <div className={action ? 'flex items-center justify-between gap-4' : undefined}>
                <div>
                    <h1 className={titleClassName ?? 'text-3xl font-bold'}>{title}</h1>
                    {description && (
                        <p className="text-[var(--color-text-secondary)] mt-2">
                            {description}
                        </p>
                    )}
                </div>
                {action}
            </div>
        </div>
    );
}

