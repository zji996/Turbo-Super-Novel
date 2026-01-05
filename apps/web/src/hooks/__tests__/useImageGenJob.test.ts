import { act, renderHook, waitFor } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { useImageGenJob } from '../useImageGenJob';
import { createImageGenJob, getImageGenJobStatus } from '../../services/imagegen';

vi.mock('../../services/imagegen', () => ({
    createImageGenJob: vi.fn(),
    getImageGenJobStatus: vi.fn(),
    cancelImageGenJob: vi.fn(),
}));

describe('useImageGenJob', () => {
    it('submits and calls onSuccess when job reaches SUCCESS', async () => {
        const onSuccess = vi.fn();

        vi.mocked(createImageGenJob).mockResolvedValueOnce({
            job_id: 'job-1',
            status: 'PENDING',
        });

        vi.mocked(getImageGenJobStatus).mockResolvedValueOnce({
            job_id: 'job-1',
            status: 'SUCCESS',
            image_url: 'http://example.com/image.png',
        });

        const { result } = renderHook(() => useImageGenJob({ onSuccess }));

        await act(async () => {
            await result.current.submit('hello');
        });

        await waitFor(() => {
            expect(result.current.status).toBe('SUCCESS');
        });

        expect(onSuccess).toHaveBeenCalledWith(
            expect.objectContaining({ job_id: 'job-1', status: 'SUCCESS' })
        );
    });
});

