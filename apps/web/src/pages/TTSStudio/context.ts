import { createContext, useContext } from 'react';
import type { SpeakerProfile } from '../../types';

export interface TTSStudioContextValue {
    profiles: SpeakerProfile[];
    selectedProfile: SpeakerProfile | null;
    setSelectedProfile: (p: SpeakerProfile | null) => void;
    refreshProfiles: () => Promise<void>;
    isLoadingProfiles: boolean;
}

export const TTSStudioContext = createContext<TTSStudioContextValue | null>(null);

export function useTTSStudioContext(): TTSStudioContextValue {
    const ctx = useContext(TTSStudioContext);
    if (!ctx) {
        throw new Error('useTTSStudioContext must be used within TTSStudio');
    }
    return ctx;
}
