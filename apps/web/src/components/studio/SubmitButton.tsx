import type { ButtonHTMLAttributes, ReactNode } from 'react';

export interface SubmitButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
    isLoading?: boolean;
    loadingText?: ReactNode;
    variant?: 'primary' | 'secondary';
}

export function SubmitButton({
    isLoading = false,
    loadingText = '提交中...',
    variant = 'primary',
    className,
    disabled,
    children,
    type,
    ...rest
}: SubmitButtonProps) {
    const baseClass = variant === 'secondary' ? 'btn-secondary' : 'btn-primary';
    const spinnerClass =
        variant === 'secondary'
            ? 'w-5 h-5 border-2 border-[var(--color-text-muted)]/30 border-t-[var(--color-text-muted)] rounded-full animate-spin'
            : 'w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin';

    return (
        <button
            type={type ?? 'button'}
            {...rest}
            disabled={disabled || isLoading}
            className={`${baseClass} ${className ?? ''}`}
        >
            {isLoading ? (
                <span className="inline-flex items-center gap-2">
                    <span className={spinnerClass} />
                    {loadingText}
                </span>
            ) : (
                children
            )}
        </button>
    );
}

