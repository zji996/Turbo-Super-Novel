import { act, renderHook, waitFor } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { useTTSJob } from '../useTTSJob';
import { createTTSJobWithProfile, getTTSJobStatus } from '../../services/tts';

vi.mock('../../services/tts', () => ({
    createTTSJobWithProfile: vi.fn(),
    getTTSJobStatus: vi.fn(),
    listSpeakerProfiles: vi.fn(),
    createSpeakerProfile: vi.fn(),
    deleteSpeakerProfile: vi.fn(),
}));

describe('useTTSJob', () => {
    it('submits and calls onSuccess when job reaches SUCCEEDED', async () => {
        const onSuccess = vi.fn();

        vi.mocked(createTTSJobWithProfile).mockResolvedValueOnce({
            job_id: 'job-1',
            status: 'SUBMITTED',
        });

        vi.mocked(getTTSJobStatus).mockResolvedValueOnce({
            job_id: 'job-1',
            status: 'SUCCEEDED',
            output_url: 'http://example.com/audio.wav',
        });

        const { result } = renderHook(() => useTTSJob(2000, onSuccess));

        await act(async () => {
            await result.current.submit({ text: 'hello', profile_id: 'p-1' });
        });

        await waitFor(() => {
            expect(result.current.state.status).toBe('SUCCEEDED');
        });

        expect(onSuccess).toHaveBeenCalledWith(
            expect.objectContaining({ jobId: 'job-1', status: 'SUCCEEDED' })
        );
    });
});

