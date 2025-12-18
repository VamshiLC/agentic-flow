"""
OCR Module - License Plate Detection and Text Extraction

Architecture:
- SAM3: Detection + Tracking (text prompt: "license plate")
- Qwen3-VL: OCR (read text from cropped plates)
- IoU Tracker: Fallback when SAM3 native tracking unavailable

Focused on North American license plate formats.
"""

from .license_plate_agent import LicensePlateOCR
from .processor import process_image, process_video
from .tracker import PlateTracker
from .sam3_tracker import SAM3PlateTracker

__all__ = [
    "LicensePlateOCR",
    "process_image",
    "process_video",
    "PlateTracker",
    "SAM3PlateTracker"
]
