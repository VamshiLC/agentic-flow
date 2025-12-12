"""
Video Processing Utilities

Utilities for extracting frames from GoPro videos and other video formats.
"""
import os
import cv2
import subprocess
import shutil
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from datetime import timedelta


def extract_frames(video_path, output_dir, sample_rate=1, start_time=None, end_time=None):
    """
    Extract frames from a video file.

    Args:
        video_path: Path to the input video file
        output_dir: Directory to save extracted frames
        sample_rate: Extract every Nth frame (1=all frames, 30=1 per second at 30fps)
        start_time: Optional start time in seconds
        end_time: Optional end time in seconds

    Returns:
        int: Number of frames extracted

    Example:
        # Extract 1 frame per second from a 30fps video
        num_frames = extract_frames("gopro_video.mp4", "frames/", sample_rate=30)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"Video info:")
    print(f"  - FPS: {fps:.2f}")
    print(f"  - Total frames: {total_frames}")
    print(f"  - Duration: {duration:.2f}s")
    print(f"  - Sample rate: every {sample_rate} frames")

    # Calculate start and end frames
    start_frame = int(start_time * fps) if start_time else 0
    end_frame = int(end_time * fps) if end_time else total_frames

    frame_count = 0
    extracted = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Check if we're within the time range
        if frame_count < start_frame:
            frame_count += 1
            continue
        if frame_count >= end_frame:
            break

        # Extract frame if it matches the sample rate
        if frame_count % sample_rate == 0:
            frame_path = os.path.join(output_dir, f"frame_{extracted:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            extracted += 1

            if extracted % 10 == 0:
                print(f"  Extracted {extracted} frames...")

        frame_count += 1

    cap.release()

    print(f"\nExtracted {extracted} frames to {output_dir}")
    return extracted


def get_video_info(video_path):
    """
    Get information about a video file.

    Args:
        video_path: Path to the video file

    Returns:
        dict: Video information (fps, width, height, frame_count, duration)
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0

    cap.release()

    return {
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "duration": duration,
        "duration_formatted": str(timedelta(seconds=int(duration)))
    }


def extract_frame_at_timestamp(video_path, timestamp_seconds, output_path=None):
    """
    Extract a single frame at a specific timestamp.

    Args:
        video_path: Path to the video file
        timestamp_seconds: Timestamp in seconds
        output_path: Optional path to save the frame (if None, returns the frame array)

    Returns:
        numpy.ndarray or str: Frame array if output_path is None, otherwise path to saved frame
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")

    # Get FPS and calculate frame number
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = int(timestamp_seconds * fps)

    # Seek to frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Failed to extract frame at {timestamp_seconds}s")

    if output_path:
        cv2.imwrite(output_path, frame)
        return output_path
    else:
        return frame


def create_video_from_frames(frames_dir, output_video_path, fps=30, frame_pattern="frame_%06d.jpg"):
    """
    Create a video from a directory of frames.

    Args:
        frames_dir: Directory containing frame images
        output_video_path: Path to save the output video
        fps: Frames per second for the output video
        frame_pattern: Pattern for frame filenames (e.g., "frame_%06d.jpg")

    Returns:
        str: Path to the created video file
    """
    import glob

    # Get list of frames
    frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))

    if not frame_files:
        raise ValueError(f"No frame files found in {frames_dir}")

    # Read first frame to get dimensions
    first_frame = cv2.imread(frame_files[0])
    height, width, _ = first_frame.shape

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print(f"Creating video from {len(frame_files)} frames...")

    for i, frame_file in enumerate(frame_files):
        frame = cv2.imread(frame_file)
        out.write(frame)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(frame_files)} frames...")

    out.release()

    print(f"\nVideo created: {output_video_path}")
    return output_video_path


def _chunk_with_ffmpeg(input_path, output_path, start_time, duration):
    """
    Use ffmpeg for fast chunking (no re-encoding).

    Args:
        input_path: Input video path
        output_path: Output chunk path
        start_time: Start time in seconds
        duration: Duration in seconds
    """
    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-ss", str(start_time),
        "-t", str(duration),
        "-c", "copy",  # Copy codec, no re-encode (fast!)
        "-avoid_negative_ts", "1",
        output_path,
        "-y",  # Overwrite
        "-loglevel", "error"
    ]
    subprocess.run(cmd, check=True)


def _chunk_with_opencv(input_path, output_path, start_time, end_time, video_info):
    """
    Fallback to OpenCV if ffmpeg not available.

    Args:
        input_path: Input video path
        output_path: Output chunk path
        start_time: Start time in seconds
        end_time: End time in seconds
        video_info: Video metadata dict
    """
    cap = cv2.VideoCapture(input_path)
    fps = video_info["fps"]
    width = video_info["width"]
    height = video_info["height"]

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    # Seek to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Create writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    current_frame = start_frame
    while current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        current_frame += 1

    cap.release()
    out.release()


def chunk_video(
    video_path: str,
    output_dir: str,
    chunk_duration: float = 600.0,  # 10 minutes
    overlap_seconds: float = 1.0,   # 1 second overlap
    format: str = "mp4"
) -> List[Dict[str, Any]]:
    """
    Split video into smaller chunks for memory-efficient processing.

    Args:
        video_path: Path to input video
        output_dir: Directory to save chunks
        chunk_duration: Duration of each chunk in seconds (default: 600 = 10 min)
        overlap_seconds: Overlap between chunks to avoid missing defects at boundaries
        format: Output video format (default: mp4)

    Returns:
        List of dicts with chunk info:
        [
            {
                "chunk_path": "/path/to/chunk_001.mp4",
                "chunk_index": 0,
                "start_time": 0.0,
                "end_time": 600.0,
                "duration": 600.0,
                "original_video": "/path/to/video.mp4"
            },
            ...
        ]

    Example:
        # Split 60-minute video into 10-minute chunks
        chunks = chunk_video("long_video.mp4", "chunks/", chunk_duration=600)

        # Process each chunk
        for chunk_info in chunks:
            process_video(chunk_info["chunk_path"], ...)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get video info
    video_info = get_video_info(video_path)
    duration = video_info["duration"]

    # If video is shorter than chunk duration, return single "chunk" (original video)
    if duration <= chunk_duration:
        print(f"Video duration ({duration:.1f}s) is less than chunk duration ({chunk_duration:.1f}s)")
        print("Processing as single chunk (no splitting needed)")
        return [{
            "chunk_path": video_path,  # Use original video
            "chunk_index": 0,
            "start_time": 0.0,
            "end_time": duration,
            "duration": duration,
            "original_video": video_path
        }]

    # Calculate chunks
    num_chunks = int(np.ceil(duration / chunk_duration))
    chunks_info = []

    print(f"Splitting video into {num_chunks} chunks...")
    print(f"  Chunk duration: {chunk_duration}s ({chunk_duration/60:.1f} min)")
    print(f"  Overlap: {overlap_seconds}s")

    video_name = Path(video_path).stem

    # Check if ffmpeg is available
    has_ffmpeg = shutil.which("ffmpeg") is not None
    if not has_ffmpeg:
        print("WARNING: ffmpeg not found. Using OpenCV fallback (slower).")
        print("For faster chunking, install ffmpeg: sudo apt install ffmpeg")

    for i in range(num_chunks):
        # Calculate time range
        start_time = max(0, i * chunk_duration - overlap_seconds if i > 0 else 0)
        end_time = min(duration, (i + 1) * chunk_duration + overlap_seconds)
        chunk_duration_actual = end_time - start_time

        # Output path
        chunk_filename = f"{video_name}_chunk_{i:03d}.{format}"
        chunk_path = os.path.join(output_dir, chunk_filename)

        # Try ffmpeg first (faster)
        try:
            if has_ffmpeg:
                _chunk_with_ffmpeg(video_path, chunk_path, start_time, chunk_duration_actual)
            else:
                # Fallback to OpenCV
                _chunk_with_opencv(video_path, chunk_path, start_time, end_time, video_info)
        except Exception as e:
            print(f"ERROR creating chunk {i}: {e}")
            # Try OpenCV fallback if ffmpeg fails
            if has_ffmpeg:
                print("Retrying with OpenCV...")
                _chunk_with_opencv(video_path, chunk_path, start_time, end_time, video_info)

        chunks_info.append({
            "chunk_path": chunk_path,
            "chunk_index": i,
            "start_time": start_time,
            "end_time": end_time,
            "duration": chunk_duration_actual,
            "original_video": video_path
        })

        print(f"  Created chunk {i+1}/{num_chunks}: {chunk_filename} ({start_time:.1f}s - {end_time:.1f}s)")

    return chunks_info
