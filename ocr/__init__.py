"""
OCR Module - License Plate Detection and Text Extraction

Uses Qwen3-VL for detection and OCR, SAM3 for segmentation.
Focused on North American license plate formats.
"""

from .license_plate_agent import LicensePlateOCR
from .processor import process_image, process_video

__all__ = [
    "LicensePlateOCR",
    "process_image",
    "process_video"
]
