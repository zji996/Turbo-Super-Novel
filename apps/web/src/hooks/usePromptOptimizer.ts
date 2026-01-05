import { useState, useCallback } from 'react';
import { optimizePrompt } from '../services/llm';

export interface UsePromptOptimizerResult {
    /** 是否正在优化 */
    isOptimizing: boolean;
    /** 优化前的原始文本（用于撤销） */
    originalText: string | null;
    /** 是否可以撤销 */
    canUndo: boolean;
    /** 执行优化，返回优化后的文本 */
    optimize: (text: string) => Promise<string | null>;
    /** 撤销优化，返回原始文本 */
    undo: () => string | null;
    /** 清除历史（撤销记录） */
    clear: () => void;
    /** 错误信息 */
    error: string | null;
}

/**
 * 提示词优化 Hook
 * 
 * 提供用户可控的 AI 优化能力：
 * - optimize: 调用 AI 优化文本
 * - undo: 撤销到优化前的文本
 * - clear: 清除撤销记录
 */
export function usePromptOptimizer(): UsePromptOptimizerResult {
    const [isOptimizing, setIsOptimizing] = useState(false);
    const [originalText, setOriginalText] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);

    const optimize = useCallback(async (text: string): Promise<string | null> => {
        if (!text.trim()) return null;

        setIsOptimizing(true);
        setError(null);

        try {
            // 保存原始文本用于撤销
            setOriginalText(text);
            const optimized = await optimizePrompt(text);
            return optimized;
        } catch (e) {
            const message = e instanceof Error ? e.message : 'AI 优化失败';
            setError(message);
            setOriginalText(null); // 失败时清除撤销记录
            return null;
        } finally {
            setIsOptimizing(false);
        }
    }, []);

    const undo = useCallback((): string | null => {
        const original = originalText;
        setOriginalText(null);
        setError(null);
        return original;
    }, [originalText]);

    const clear = useCallback(() => {
        setOriginalText(null);
        setError(null);
    }, []);

    return {
        isOptimizing,
        originalText,
        canUndo: originalText !== null,
        optimize,
        undo,
        clear,
        error,
    };
}
