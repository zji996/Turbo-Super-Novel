import type { ImageGenParams } from '../../types';

export interface SizePreset {
    label: string;
    width: number;
    height: number;
}

export interface ImageStudioParamsState {
    params: ImageGenParams;
    showAdvanced: boolean;
}
