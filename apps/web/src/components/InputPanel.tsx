import { useCallback, useRef, useState } from 'react';

interface InputPanelProps {
    onImageChange: (file: File | null, preview: string | null) => void;
    onPromptChange: (prompt: string) => void;
    imagePreview: string | null;
    prompt: string;
    disabled?: boolean;
}

export function InputPanel({
    onImageChange,
    onPromptChange,
    imagePreview,
    prompt,
    disabled = false,
}: InputPanelProps) {
    const fileInputRef = useRef<HTMLInputElement>(null);
    const [isDragging, setIsDragging] = useState(false);

    const handleFileSelect = useCallback((file: File | null) => {
        if (!file) {
            onImageChange(null, null);
            return;
        }

        // Validate file type
        if (!file.type.startsWith('image/')) {
            alert('Please select an image file (jpg, png, etc.)');
            return;
        }

        // Create preview
        const reader = new FileReader();
        reader.onload = (e) => {
            onImageChange(file, e.target?.result as string);
        };
        reader.readAsDataURL(file);
    }, [onImageChange]);

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);

        const file = e.dataTransfer.files[0];
        if (file) {
            handleFileSelect(file);
        }
    }, [handleFileSelect]);

    const handleDragOver = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(true);
    }, []);

    const handleDragLeave = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);
    }, []);

    const handleClick = useCallback(() => {
        fileInputRef.current?.click();
    }, []);

    const handleInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
        handleFileSelect(e.target.files?.[0] || null);
    }, [handleFileSelect]);

    const handleClearImage = useCallback((e: React.MouseEvent) => {
        e.stopPropagation();
        onImageChange(null, null);
        if (fileInputRef.current) {
            fileInputRef.current.value = '';
        }
    }, [onImageChange]);

    return (
        <div className="space-y-6">
            {/* Image Upload */}
            <div>
                <label className="block text-sm font-medium text-[var(--color-text-secondary)] mb-2">
                    Input Image <span className="text-[var(--color-error)]">*</span>
                </label>
                <div
                    onClick={disabled ? undefined : handleClick}
                    onDrop={disabled ? undefined : handleDrop}
                    onDragOver={disabled ? undefined : handleDragOver}
                    onDragLeave={disabled ? undefined : handleDragLeave}
                    className={`
            relative rounded-xl border-2 border-dashed transition-all duration-300
            ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
            ${isDragging
                            ? 'border-[var(--color-accent-primary)] bg-[var(--color-accent-primary)]/10'
                            : 'border-[var(--color-border)] hover:border-[var(--color-border-hover)]'
                        }
            ${imagePreview ? 'p-2' : 'p-8'}
          `}
                >
                    <input
                        ref={fileInputRef}
                        type="file"
                        accept="image/jpeg,image/png,image/webp"
                        onChange={handleInputChange}
                        className="hidden"
                        disabled={disabled}
                    />

                    {imagePreview ? (
                        <div className="relative group">
                            <img
                                src={imagePreview}
                                alt="Preview"
                                className="w-full rounded-lg object-contain max-h-64"
                            />
                            {!disabled && (
                                <button
                                    onClick={handleClearImage}
                                    className="absolute top-2 right-2 w-8 h-8 rounded-full bg-[var(--color-bg-primary)]/80 hover:bg-[var(--color-error)] flex items-center justify-center transition-colors opacity-0 group-hover:opacity-100"
                                >
                                    <span className="text-lg">✕</span>
                                </button>
                            )}
                        </div>
                    ) : (
                        <div className="text-center">
                            <div className="text-4xl mb-3">📸</div>
                            <p className="text-[var(--color-text-primary)] font-medium">
                                Drop image here or click to upload
                            </p>
                            <p className="text-sm text-[var(--color-text-muted)] mt-1">
                                Supports JPG, PNG, WebP
                            </p>
                        </div>
                    )}
                </div>
            </div>

            {/* Prompt */}
            <div>
                <label className="block text-sm font-medium text-[var(--color-text-secondary)] mb-2">
                    Prompt <span className="text-[var(--color-error)]">*</span>
                </label>
                <textarea
                    value={prompt}
                    onChange={(e) => onPromptChange(e.target.value)}
                    placeholder="Describe the video you want to generate..."
                    disabled={disabled}
                    rows={4}
                    className="input resize-none"
                />
                <p className="text-xs text-[var(--color-text-muted)] mt-1">
                    Be specific about motion and camera movement
                </p>
            </div>
        </div>
    );
}
