"""Video composition utilities.

Handles merging audio tracks with video, and concatenating
multiple video segments into a final output.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Sequence


class CompositionError(RuntimeError):
    """Raised when video composition fails."""
    pass


def check_ffmpeg() -> bool:
    """Check if ffmpeg is available."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def merge_audio_video(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    *,
    overwrite: bool = True,
) -> Path:
    """Merge an audio track with a video file.
    
    Args:
        video_path: Path to the input video file.
        audio_path: Path to the audio file.
        output_path: Path for the output video with audio.
        overwrite: If True, overwrite existing output file.
    
    Returns:
        Path to the output video.
    
    Raises:
        CompositionError: If the merge operation fails.
    """
    if not video_path.is_file():
        raise CompositionError(f"Video file not found: {video_path}")
    if not audio_path.is_file():
        raise CompositionError(f"Audio file not found: {audio_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",  # End when the shorter input ends
    ]
    
    if overwrite:
        cmd.append("-y")
    
    cmd.append(str(output_path))
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )
        if result.returncode != 0:
            raise CompositionError(f"FFmpeg merge failed: {result.stderr}")
    except subprocess.TimeoutExpired as e:
        raise CompositionError(f"FFmpeg merge timed out") from e
    except FileNotFoundError as e:
        raise CompositionError("FFmpeg not found. Please install FFmpeg.") from e
    
    if not output_path.is_file():
        raise CompositionError(f"Output file not created: {output_path}")
    
    return output_path


def concatenate_videos(
    video_paths: Sequence[Path],
    output_path: Path,
    *,
    overwrite: bool = True,
) -> Path:
    """Concatenate multiple video files into one.
    
    All videos should have the same codec, resolution, and frame rate
    for seamless concatenation.
    
    Args:
        video_paths: List of video file paths in order.
        output_path: Path for the output video.
        overwrite: If True, overwrite existing output file.
    
    Returns:
        Path to the concatenated video.
    
    Raises:
        CompositionError: If the concatenation fails.
    """
    if not video_paths:
        raise CompositionError("No video paths provided")
    
    for path in video_paths:
        if not path.is_file():
            raise CompositionError(f"Video file not found: {path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create a temporary file list for FFmpeg concat demuxer
    list_file = output_path.with_suffix(".txt")
    try:
        with open(list_file, "w", encoding="utf-8") as f:
            for path in video_paths:
                # Escape single quotes in path
                escaped_path = str(path.absolute()).replace("'", "'\\''")
                f.write(f"file '{escaped_path}'\n")
        
        cmd = [
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",
        ]
        
        if overwrite:
            cmd.append("-y")
        
        cmd.append(str(output_path))
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout for long videos
            )
            if result.returncode != 0:
                raise CompositionError(f"FFmpeg concat failed: {result.stderr}")
        except subprocess.TimeoutExpired as e:
            raise CompositionError("FFmpeg concat timed out") from e
        except FileNotFoundError as e:
            raise CompositionError("FFmpeg not found. Please install FFmpeg.") from e
        
    finally:
        # Clean up list file
        if list_file.is_file():
            list_file.unlink()
    
    if not output_path.is_file():
        raise CompositionError(f"Output file not created: {output_path}")
    
    return output_path


def get_video_duration(video_path: Path) -> float:
    """Get the duration of a video file in seconds.
    
    Args:
        video_path: Path to the video file.
    
    Returns:
        Duration in seconds.
    
    Raises:
        CompositionError: If unable to get duration.
    """
    if not video_path.is_file():
        raise CompositionError(f"Video file not found: {video_path}")
    
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise CompositionError(f"FFprobe failed: {result.stderr}")
        
        return float(result.stdout.strip())
    except (subprocess.TimeoutExpired, ValueError) as e:
        raise CompositionError(f"Failed to get video duration: {e}") from e
    except FileNotFoundError as e:
        raise CompositionError("FFprobe not found. Please install FFmpeg.") from e
