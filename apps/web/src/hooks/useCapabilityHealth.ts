import { useContext } from 'react';
import type { CapabilityHealthContextValue } from './capabilityHealthContext';
import { CapabilityHealthContext } from './capabilityHealthContext';

export function useCapabilityHealth(): CapabilityHealthContextValue {
    const ctx = useContext(CapabilityHealthContext);
    if (!ctx) {
        throw new Error('useCapabilityHealth must be used within CapabilityHealthProvider');
    }
    return ctx;
}

