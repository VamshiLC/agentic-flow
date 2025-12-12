"""
Utility functions for video processing and output formatting
"""
from .video_utils import (
    extract_frames,
    get_video_info,
    extract_frame_at_timestamp,
    create_video_from_frames
)
from .output_formatter import (
    format_detection_output,
    format_single_detection,
    save_detection_json,
    create_detection_summary,
    CATEGORY_MAPPINGS
)

__all__ = [
    # Video utils
    "extract_frames",
    "get_video_info",
    "extract_frame_at_timestamp",
    "create_video_from_frames",
    # Output formatting
    "format_detection_output",
    "format_single_detection",
    "save_detection_json",
    "create_detection_summary",
    "CATEGORY_MAPPINGS"
]
