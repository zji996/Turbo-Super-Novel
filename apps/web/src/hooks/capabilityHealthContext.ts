import { createContext } from 'react';
import type { CapabilityName } from '../services/capabilities';

export type CapabilityHealthStatus = 'unknown' | 'available' | 'unavailable';

export interface CapabilityHealth {
    status: CapabilityHealthStatus;
    lastSuccess: number | null;
    lastError: string | null;
    provider: string | null;
}

export interface CapabilityHealthContextValue {
    health: Record<CapabilityName, CapabilityHealth>;
    isProbing: boolean;
    probeError: string | null;
    refreshFromProbe: () => Promise<void>;
    reportSuccess: (capability: CapabilityName) => void;
    reportFailure: (capability: CapabilityName, error: string) => void;
}

export const CAPABILITIES: CapabilityName[] = ['tts', 'imagegen', 'videogen', 'llm'];

export function defaultHealth(): Record<CapabilityName, CapabilityHealth> {
    return CAPABILITIES.reduce(
        (acc, name) => {
            acc[name] = {
                status: 'unknown',
                lastSuccess: null,
                lastError: null,
                provider: null,
            };
            return acc;
        },
        {} as Record<CapabilityName, CapabilityHealth>
    );
}

export const CapabilityHealthContext = createContext<CapabilityHealthContextValue | null>(null);

