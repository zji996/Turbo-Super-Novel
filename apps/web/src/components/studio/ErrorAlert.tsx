export interface ErrorAlertProps {
    message: string | null | undefined;
    className?: string;
}

export function ErrorAlert({ message, className }: ErrorAlertProps) {
    if (!message) return null;
    return (
        <div
            className={
                className ??
                'p-4 rounded-lg bg-[var(--color-error)]/10 border border-[var(--color-error)]/20'
            }
        >
            <p className="text-sm text-[var(--color-error)]">{message}</p>
        </div>
    );
}

