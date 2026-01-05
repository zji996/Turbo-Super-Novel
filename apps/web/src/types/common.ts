export type BaseJobStatus =
    | 'CREATED'
    | 'SUBMITTED'
    | 'PENDING'
    | 'STARTED'
    | 'PROGRESS'
    | 'DOWNLOADED'
    | 'RUNNING'
    | 'UPLOADED'
    | 'SUCCESS'
    | 'FAILURE'
    | 'SUCCEEDED'
    | 'FAILED'
    | 'REVOKED'
    | 'CANCELLED';

const TERMINAL_STATUSES: ReadonlySet<string> = new Set([
    'SUCCESS',
    'FAILURE',
    'SUCCEEDED',
    'FAILED',
    'REVOKED',
    'CANCELLED',
]);

export function isTerminalStatus(status: string | null | undefined): boolean {
    return status != null && TERMINAL_STATUSES.has(status);
}

export function isPendingStatus(status: string | null | undefined): boolean {
    return status != null && !isTerminalStatus(status);
}

