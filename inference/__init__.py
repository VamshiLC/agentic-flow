"""
Inference module for single frame and video processing
"""
from .single_frame import process_single_frame, process_single_frame_simple
from .video_processor import process_video, process_video_simple

__all__ = [
    "process_single_frame",
    "process_single_frame_simple",
    "process_video",
    "process_video_simple"
]
