"""
OCR Module - License Plate Detection and Text Extraction

Uses Qwen3-VL for detection and OCR.
Includes plate tracking across video frames with voting.
Focused on North American license plate formats.
"""

from .license_plate_agent import LicensePlateOCR
from .processor import process_image, process_video
from .tracker import PlateTracker

__all__ = [
    "LicensePlateOCR",
    "process_image",
    "process_video",
    "PlateTracker"
]
