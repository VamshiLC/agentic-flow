"""
License Plate Detection Agent with OCR

Two-stage approach for better accuracy:
1. Detect license plates using Qwen3-VL
2. Segment plates using SAM3 (optional)
3. Crop plate region and run OCR using Qwen3-VL

Reuses existing model loaders from the project.
"""

import re
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Optional, Union, Tuple
import logging

# Import existing Qwen model loader (no modifications needed)
from models.qwen_direct_loader import Qwen3VLDirectDetector

from .prompts import (
    build_plate_detection_prompt,
    build_ocr_prompt,
    build_combined_detection_ocr_prompt
)

logger = logging.getLogger(__name__)


class LicensePlateOCR:
    """
    License plate detection and OCR agent.

    Architecture:
    - Qwen3-VL: Detects plates + performs OCR (reused from existing code)
    - SAM3: Segments plate regions for precise masks (reused from existing code)

    Two modes:
    1. two_stage=True (default): Detect -> Crop -> OCR (better accuracy)
    2. two_stage=False: Combined detection + OCR in single pass (faster)
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-4B-Instruct",
        device: Optional[str] = None,
        use_quantization: bool = False,
        low_memory: bool = False,
        two_stage: bool = True,
        plate_padding: float = 0.1
    ):
        """
        Initialize license plate OCR agent.

        Args:
            model_name: Qwen model ID (default: Qwen3-VL-4B-Instruct)
            device: cuda/cpu (auto-detect if None)
            use_quantization: Use 8-bit quantization for lower memory
            low_memory: Enable additional memory optimizations
            two_stage: Use two-stage detection+OCR (recommended for accuracy)
            plate_padding: Padding ratio when cropping plate (0.1 = 10%)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.two_stage = two_stage
        self.plate_padding = plate_padding
        self.use_quantization = use_quantization

        print(f"\n{'='*60}")
        print("LICENSE PLATE OCR - Initializing")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Mode: {'Two-stage (accurate)' if two_stage else 'Single-pass (fast)'}")
        print(f"Quantization: {'Enabled' if use_quantization else 'Disabled'}")
        print(f"{'='*60}\n")

        # Load Qwen3-VL (shared for detection and OCR)
        print("Loading Qwen3-VL model...")
        self.qwen_detector = Qwen3VLDirectDetector(
            model_name=model_name,
            device=device,
            use_quantization=use_quantization,
            low_memory=low_memory
        )

        # SAM3 not needed for OCR - just detection + text reading
        self.sam3_processor = None
        self.sam3_available = False

        print(f"\n{'='*60}")
        print("LICENSE PLATE OCR - Ready")
        print(f"{'='*60}\n")

    def detect_and_read(
        self,
        image: Union[Image.Image, str, np.ndarray]
    ) -> Dict:
        """
        Detect license plates and extract text.

        Args:
            image: PIL Image, numpy array, or path to image

        Returns:
            dict: {
                'plates': [
                    {
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float,
                        'plate_text': str,
                        'ocr_confidence': float,
                        'state': str,
                        'format': str
                    }
                ],
                'num_plates': int
            }
        """
        # Convert to PIL Image
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')

        if self.two_stage:
            return self._detect_two_stage(image)
        else:
            return self._detect_single_stage(image)

    def _detect_two_stage(
        self,
        image: Image.Image
    ) -> Dict:
        """
        Two-stage detection: Detect plates -> Crop -> OCR

        More accurate OCR because the model focuses only on the plate region.
        """
        # Stage 1: Detect plate locations
        logger.info("Stage 1: Detecting license plates...")
        detection_prompt = build_plate_detection_prompt()
        detection_result = self.qwen_detector.detect(image, detection_prompt)

        if not detection_result.get('success', False):
            logger.error(f"Detection failed: {detection_result.get('error', 'Unknown error')}")
            return {'plates': [], 'num_plates': 0}

        # Parse detection response
        plates = self._parse_plate_detections(
            detection_result.get('text', ''),
            image.size
        )

        if len(plates) == 0:
            logger.info("No plates detected")
            return {'plates': [], 'num_plates': 0}

        logger.info(f"Detected {len(plates)} plate(s)")

        # Stage 2: For each plate, crop and run OCR
        enhanced_plates = []
        for i, plate in enumerate(plates):
            bbox = plate['bbox']
            logger.info(f"Stage 2: Reading plate {i+1}/{len(plates)}...")

            # Crop plate region with padding
            cropped_plate = self._crop_plate_region(image, bbox)

            # Run OCR on cropped plate
            ocr_result = self._read_plate_text(cropped_plate)
            plate['plate_text'] = ocr_result.get('text', 'UNREADABLE')
            plate['ocr_confidence'] = ocr_result.get('confidence', 0.0)
            plate['state'] = ocr_result.get('state', 'Unknown')
            plate['format'] = ocr_result.get('format', 'Unknown')

            enhanced_plates.append(plate)
            logger.info(f"  Plate text: {plate['plate_text']} (conf: {plate['ocr_confidence']:.2f})")

        return {
            'plates': enhanced_plates,
            'num_plates': len(enhanced_plates)
        }

    def _detect_single_stage(
        self,
        image: Image.Image
    ) -> Dict:
        """
        Single-stage: Combined detection + OCR in one pass.
        Faster but potentially less accurate OCR.
        """
        logger.info("Single-stage: Detecting plates and reading text...")
        prompt = build_combined_detection_ocr_prompt()
        result = self.qwen_detector.detect(image, prompt)

        if not result.get('success', False):
            return {'plates': [], 'num_plates': 0}

        plates = self._parse_combined_response(
            result.get('text', ''),
            image.size
        )

        return {
            'plates': plates,
            'num_plates': len(plates)
        }

    def _crop_plate_region(
        self,
        image: Image.Image,
        bbox: List[int]
    ) -> Image.Image:
        """
        Crop plate region with padding for better OCR.

        Args:
            image: Full image
            bbox: [x1, y1, x2, y2] bounding box

        Returns:
            Cropped plate image with padding
        """
        x1, y1, x2, y2 = bbox
        width, height = image.size

        # Add padding for better OCR
        pad_w = int((x2 - x1) * self.plate_padding)
        pad_h = int((y2 - y1) * self.plate_padding)

        # Expand bbox with padding, clamped to image bounds
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(width, x2 + pad_w)
        y2 = min(height, y2 + pad_h)

        return image.crop((x1, y1, x2, y2))

    def _read_plate_text(self, plate_image: Image.Image) -> Dict:
        """
        Run OCR on cropped plate image.

        Args:
            plate_image: Cropped plate image

        Returns:
            dict with text, confidence, state, format
        """
        ocr_prompt = build_ocr_prompt()
        ocr_result = self.qwen_detector.detect(plate_image, ocr_prompt)

        if not ocr_result.get('success', False):
            return {
                'text': 'UNREADABLE',
                'confidence': 0.0,
                'state': 'Unknown',
                'format': 'Unknown'
            }

        return self._parse_ocr_response(ocr_result.get('text', ''))

    def _parse_plate_detections(
        self,
        response: str,
        image_size: Tuple[int, int]
    ) -> List[Dict]:
        """
        Parse plate detection response from Qwen.

        Args:
            response: Raw text response from Qwen
            image_size: (width, height) of image

        Returns:
            List of plate detection dicts
        """
        plates = []
        width, height = image_size

        # Check for no detections
        response_lower = response.lower()
        if "no defects detected" in response_lower or "no plates" in response_lower:
            return plates

        # Pattern: Defect: license_plate, Box: [x1, y1, x2, y2], Confidence: <score>
        pattern = r'Defect:\s*license_plate,\s*Box:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\],\s*Confidence:\s*([\d.]+)'
        matches = re.findall(pattern, response, re.IGNORECASE)

        for match in matches:
            try:
                # Convert normalized 0-1000 coords to pixel coords
                x1 = int(float(match[0]) * width / 1000)
                y1 = int(float(match[1]) * height / 1000)
                x2 = int(float(match[2]) * width / 1000)
                y2 = int(float(match[3]) * height / 1000)
                confidence = float(match[4])

                # Clamp to image bounds
                x1 = max(0, min(x1, width))
                y1 = max(0, min(y1, height))
                x2 = max(0, min(x2, width))
                y2 = max(0, min(y2, height))

                # Validate bbox
                if x2 > x1 and y2 > y1 and confidence >= 0.5:
                    plates.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'color': (255, 165, 0)  # Orange for plates
                    })

            except (ValueError, IndexError) as e:
                logger.warning(f"Failed to parse plate detection: {match} ({e})")
                continue

        return plates

    def _parse_ocr_response(self, response: str) -> Dict:
        """
        Parse OCR response from Qwen.

        Args:
            response: Raw OCR response text

        Returns:
            dict with text, confidence, state, format
        """
        result = {
            'text': 'UNREADABLE',
            'confidence': 0.0,
            'state': 'Unknown',
            'format': 'Unknown'
        }

        # Pattern: PlateText: <text>
        text_match = re.search(
            r'PlateText:\s*(.+?)(?:\n|Confidence:|$)',
            response,
            re.IGNORECASE
        )
        if text_match:
            text = text_match.group(1).strip()
            # Clean up the text
            text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
            result['text'] = text

        # Pattern: Confidence: <score>
        conf_match = re.search(r'Confidence:\s*([\d.]+)', response, re.IGNORECASE)
        if conf_match:
            try:
                result['confidence'] = float(conf_match.group(1))
            except ValueError:
                pass

        # Pattern: State: <state_name>
        state_match = re.search(r'State:\s*(.+?)(?:\n|Format:|$)', response, re.IGNORECASE)
        if state_match:
            result['state'] = state_match.group(1).strip()

        # Pattern: Format: <description>
        format_match = re.search(r'Format:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        if format_match:
            result['format'] = format_match.group(1).strip()

        return result

    def _parse_combined_response(
        self,
        response: str,
        image_size: Tuple[int, int]
    ) -> List[Dict]:
        """
        Parse combined detection+OCR response.

        Args:
            response: Raw response from combined prompt
            image_size: (width, height) of image

        Returns:
            List of plate dicts with bbox and text
        """
        plates = []
        width, height = image_size

        if "no plates" in response.lower():
            return plates

        # Pattern: Plate: Box: [x1, y1, x2, y2], Text: <text>, Confidence: <score>, State: <state>
        pattern = r'Plate:\s*Box:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\],\s*Text:\s*([^,]+),\s*Confidence:\s*([\d.]+)(?:,\s*State:\s*([^\n]+))?'
        matches = re.findall(pattern, response, re.IGNORECASE)

        for match in matches:
            try:
                x1 = int(float(match[0]) * width / 1000)
                y1 = int(float(match[1]) * height / 1000)
                x2 = int(float(match[2]) * width / 1000)
                y2 = int(float(match[3]) * height / 1000)
                plate_text = match[4].strip()
                confidence = float(match[5])
                state = match[6].strip() if len(match) > 6 and match[6] else 'Unknown'

                # Clamp to image bounds
                x1 = max(0, min(x1, width))
                y1 = max(0, min(y1, height))
                x2 = max(0, min(x2, width))
                y2 = max(0, min(y2, height))

                if x2 > x1 and y2 > y1:
                    plates.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'plate_text': plate_text,
                        'ocr_confidence': confidence,
                        'state': state,
                        'format': 'Unknown',
                        'color': (255, 165, 0)
                    })

            except (ValueError, IndexError) as e:
                logger.warning(f"Failed to parse combined response: {match} ({e})")
                continue

        return plates

    def cleanup(self):
        """Clean up models and free GPU memory."""
        if hasattr(self, 'qwen_detector'):
            self.qwen_detector.cleanup()
        if self.device == "cuda":
            torch.cuda.empty_cache()
            print("GPU memory cleared")
