const API_BASE = import.meta.env.VITE_API_BASE_URL || '';

export class ApiError extends Error {
    constructor(
        public status: number,
        message: string
    ) {
        super(message);
        this.name = 'ApiError';
    }
}

async function parseResponse<T>(resp: Response): Promise<T> {
    if (resp.status === 204) return undefined as T;

    const contentType = resp.headers.get('content-type') || '';
    if (contentType.includes('application/json')) {
        return resp.json();
    }

    const text = await resp.text();
    if (!text) return undefined as T;
    return text as unknown as T;
}

export async function apiRequest<T>(
    method: 'GET' | 'POST' | 'PUT' | 'DELETE',
    path: string,
    body?: unknown
): Promise<T> {
    const init: RequestInit = { method };

    if (body !== undefined) {
        if (body instanceof FormData) {
            init.body = body;
        } else {
            init.headers = { 'Content-Type': 'application/json' };
            init.body = JSON.stringify(body);
        }
    }

    const resp = await fetch(`${API_BASE}${path}`, init);

    if (!resp.ok) {
        const error = await resp.text();
        throw new ApiError(resp.status, error || resp.statusText);
    }

    return parseResponse<T>(resp);
}

export async function apiGet<T>(path: string): Promise<T> {
    return apiRequest<T>('GET', path);
}

export async function apiPost<T>(path: string, body?: unknown): Promise<T> {
    return apiRequest<T>('POST', path, body);
}

export async function apiPut<T>(path: string, body?: unknown): Promise<T> {
    return apiRequest<T>('PUT', path, body);
}

export async function apiDelete<T>(path: string): Promise<T> {
    return apiRequest<T>('DELETE', path);
}

