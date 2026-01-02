import type { ReactNode } from 'react';
import { useCallback, useEffect, useMemo, useState } from 'react';
import type { CapabilityName } from '../services/capabilities';
import { fetchCapabilitiesStatus } from '../services/capabilities';
import type { CapabilityHealthContextValue, CapabilityHealthStatus } from './capabilityHealthContext';
import { CAPABILITIES, CapabilityHealthContext, defaultHealth } from './capabilityHealthContext';

export function CapabilityHealthProvider({ children }: { children: ReactNode }) {
    const [health, setHealth] = useState(() => defaultHealth());
    const [isProbing, setIsProbing] = useState(false);
    const [probeError, setProbeError] = useState<string | null>(null);

    const refreshFromProbe = useCallback(async () => {
        setIsProbing(true);
        setProbeError(null);
        try {
            const resp = await fetchCapabilitiesStatus();
            setHealth((prev) => {
                const next = { ...prev };
                for (const name of CAPABILITIES) {
                    const probe = resp.capabilities?.[name];
                    const status: CapabilityHealthStatus =
                        probe?.status === 'available'
                            ? 'available'
                            : probe?.status === 'unavailable'
                                ? 'unavailable'
                                : 'unknown';

                    next[name] = {
                        ...next[name],
                        status,
                        provider: probe?.provider ? String(probe.provider) : next[name].provider,
                    };
                }
                return next;
            });
        } catch (e) {
            setProbeError(e instanceof Error ? e.message : String(e));
        } finally {
            setIsProbing(false);
        }
    }, []);

    useEffect(() => {
        refreshFromProbe();
    }, [refreshFromProbe]);

    const reportSuccess = useCallback((capability: CapabilityName) => {
        setHealth((prev) => ({
            ...prev,
            [capability]: {
                ...prev[capability],
                status: 'available',
                lastSuccess: Date.now(),
                lastError: null,
            },
        }));
    }, []);

    const reportFailure = useCallback((capability: CapabilityName, error: string) => {
        setHealth((prev) => ({
            ...prev,
            [capability]: {
                ...prev[capability],
                status: 'unavailable',
                lastError: error,
            },
        }));
    }, []);

    const value = useMemo<CapabilityHealthContextValue>(
        () => ({
            health,
            isProbing,
            probeError,
            refreshFromProbe,
            reportSuccess,
            reportFailure,
        }),
        [health, isProbing, probeError, refreshFromProbe, reportSuccess, reportFailure]
    );

    return <CapabilityHealthContext.Provider value={value}>{children}</CapabilityHealthContext.Provider>;
}

