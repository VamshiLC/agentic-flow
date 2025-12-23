"""
OCR Module - License Plate Detection and Text Extraction

Architecture:
- SAM3: Detection + Tracking (text prompt: "license plate")
- Qwen3-VL: OCR (read text from cropped plates)
- IoU Tracker: Fallback when SAM3 native tracking unavailable

Features:
- Image preprocessing (CLAHE, denoise, upscaling) for better OCR
- Multi-frame voting for video tracking
- Post-processing text correction

Focused on North American license plate formats.
"""

from .license_plate_agent import LicensePlateOCR
from .processor import process_image, process_video
from .tracker import PlateTracker
from .sam3_tracker import SAM3PlateTracker
from .utils import (
    preprocess_plate_for_ocr,
    upscale_plate,
    correct_ocr_text,
    MultiFrameOCRVoter,
    vote_ocr_results,
    vote_ocr_by_character
)

__all__ = [
    "LicensePlateOCR",
    "process_image",
    "process_video",
    "PlateTracker",
    "SAM3PlateTracker",
    # Preprocessing utilities
    "preprocess_plate_for_ocr",
    "upscale_plate",
    "correct_ocr_text",
    "MultiFrameOCRVoter",
    "vote_ocr_results",
    "vote_ocr_by_character"
]
