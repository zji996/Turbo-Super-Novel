import { renderHook, waitFor } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import type { I2VJob } from '../../types';
import { useJobPolling } from '../useJobPolling';

import { getJobStatus } from '../../services/videogen';

vi.mock('../../services/videogen', () => ({
    getJobStatus: vi.fn(),
}));

describe('useJobPolling', () => {
    it('polls job and calls onComplete when SUCCESS', async () => {
        const updateJob = vi.fn();
        const onComplete = vi.fn();

        const job: I2VJob = {
            job_id: 'job-1',
            job_type: 'i2v',
            status: 'PENDING',
            inputs: { prompt: 'hello' },
            params: { seed: 0, num_steps: 4, quantized: true, duration_seconds: 5 },
            created_at: Date.now(),
        };

        vi.mocked(getJobStatus).mockResolvedValueOnce({
            job_id: job.job_id,
            status: 'SUCCESS',
            video_url: 'http://example.com/video.mp4',
        });

        renderHook(() => useJobPolling([job], updateJob, { onComplete }));

        await waitFor(() => {
            expect(updateJob).toHaveBeenCalledWith(
                job.job_id,
                expect.objectContaining({ status: 'SUCCESS' })
            );
        });

        expect(onComplete).toHaveBeenCalledWith(
            expect.objectContaining({ job_id: job.job_id, status: 'SUCCESS' })
        );
    });
});

