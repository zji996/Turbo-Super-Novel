const API_BASE = import.meta.env.VITE_API_BASE_URL || '';

export type CapabilityName = 'tts' | 'imagegen' | 'videogen' | 'llm';

export interface CapabilityProbe {
    provider?: 'local' | 'remote' | string | null;
    status?: 'available' | 'unavailable' | string;
    detail?: string;
}

export interface CapabilitiesStatusResponse {
    capabilities: Record<CapabilityName, CapabilityProbe>;
}

export async function fetchCapabilitiesStatus(): Promise<CapabilitiesStatusResponse> {
    const resp = await fetch(`${API_BASE}/v1/capabilities/status`);
    if (!resp.ok) {
        const error = await resp.text();
        throw new Error(`Failed to fetch capabilities status: ${error}`);
    }
    return resp.json();
}

