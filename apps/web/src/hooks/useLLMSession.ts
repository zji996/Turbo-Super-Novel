import { useCallback, useEffect, useMemo, useState } from 'react';
import type { ChatMessage } from '../services/llm';

const DEFAULT_STORAGE_KEY = 'tsn_llm_studio_v1';

interface StoredState {
    systemPrompt: string;
    model: string;
    temperature: number;
    maxTokens: number;
    messages: ChatMessage[];
}

function loadState(storageKey: string): StoredState | null {
    try {
        const raw = sessionStorage.getItem(storageKey);
        if (!raw) return null;
        return JSON.parse(raw) as StoredState;
    } catch {
        return null;
    }
}

export function useLLMSession(storageKey: string = DEFAULT_STORAGE_KEY) {
    const stored = useMemo(() => loadState(storageKey), [storageKey]);

    const [systemPrompt, setSystemPrompt] = useState(stored?.systemPrompt || '');
    const [model, setModel] = useState(stored?.model || '');
    const [temperature, setTemperature] = useState(stored?.temperature ?? 0.7);
    const [maxTokens, setMaxTokens] = useState(stored?.maxTokens ?? 1024);
    const [messages, setMessages] = useState<ChatMessage[]>(stored?.messages || []);

    useEffect(() => {
        const next: StoredState = { systemPrompt, model, temperature, maxTokens, messages };
        try {
            sessionStorage.setItem(storageKey, JSON.stringify(next));
        } catch {
            // ignore
        }
    }, [storageKey, systemPrompt, model, temperature, maxTokens, messages]);

    const clearMessages = useCallback(() => {
        setMessages([]);
    }, []);

    return {
        systemPrompt,
        setSystemPrompt,
        model,
        setModel,
        temperature,
        setTemperature,
        maxTokens,
        setMaxTokens,
        messages,
        setMessages,
        clearMessages,
    };
}

