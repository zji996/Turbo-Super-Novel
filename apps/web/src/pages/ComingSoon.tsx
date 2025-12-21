interface ComingSoonProps {
    title: string;
}

export function ComingSoon({ title }: ComingSoonProps) {
    return (
        <div className="animate-fade-in flex items-center justify-center min-h-[60vh]">
            <div className="text-center">
                <div className="text-6xl mb-6">🚧</div>
                <h1 className="text-3xl font-bold text-[var(--color-text-primary)] mb-4">
                    {title}
                </h1>
                <p className="text-lg text-[var(--color-text-secondary)] max-w-md">
                    This feature is coming soon. Stay tuned for updates!
                </p>
                <div className="mt-8 inline-flex items-center gap-2 px-4 py-2 rounded-full bg-[var(--color-bg-tertiary)] text-[var(--color-text-muted)]">
                    <span className="w-2 h-2 rounded-full bg-[var(--color-warning)] animate-pulse" />
                    Under Development
                </div>
            </div>
        </div>
    );
}
