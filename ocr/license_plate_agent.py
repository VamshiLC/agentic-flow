"""
License Plate Detection Agent with OCR

Architecture:
- SAM3: Detection + Tracking (text prompt: "license plate")
- Qwen3-VL: OCR (read text from cropped plates)

Falls back to Qwen3-VL for detection if SAM3 unavailable.
"""

import re
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Optional, Union, Tuple, Any
import logging

# Import Qwen model loader
from models.qwen_direct_loader import Qwen3VLDirectDetector

from .prompts import (
    build_plate_detection_prompt,
    build_ocr_prompt,
    build_combined_detection_ocr_prompt
)
from .sam3_tracker import SAM3PlateTracker
from .utils import (
    preprocess_plate_for_ocr,
    upscale_plate,
    is_plate_blurry,
    correct_ocr_text,
    MultiFrameOCRVoter,
    identify_state_from_text
)

logger = logging.getLogger(__name__)


class LicensePlateOCR:
    """
    License plate detection and OCR agent.

    Architecture:
    - SAM3: Detection + Tracking (with text prompt "license plate")
    - Qwen3-VL: OCR (read text from cropped plates)

    Falls back to Qwen3-VL for detection if SAM3 is unavailable.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-4B-Instruct",
        device: Optional[str] = None,
        use_quantization: bool = False,
        low_memory: bool = False,
        use_sam3: bool = True,
        plate_padding: float = 0.15,
        enable_preprocessing: bool = True,
        enable_multi_frame_voting: bool = True
    ):
        """
        Initialize license plate OCR agent.

        Args:
            model_name: Qwen model ID (default: Qwen3-VL-4B-Instruct)
            device: cuda/cpu (auto-detect if None)
            use_quantization: Use 8-bit quantization for lower memory
            low_memory: Enable additional memory optimizations
            use_sam3: Use SAM3 for detection/tracking (recommended)
            plate_padding: Padding ratio when cropping plate (0.15 = 15%)
            enable_preprocessing: Apply image preprocessing for better OCR
            enable_multi_frame_voting: Use multi-frame voting for video
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.plate_padding = plate_padding
        self.use_quantization = use_quantization
        self.use_sam3 = use_sam3
        self.enable_preprocessing = enable_preprocessing
        self.enable_multi_frame_voting = enable_multi_frame_voting

        # Multi-frame OCR voting for video tracking
        self.ocr_voter = MultiFrameOCRVoter(min_votes=3, max_history=30) if enable_multi_frame_voting else None

        print(f"\n{'='*60}")
        print("LICENSE PLATE OCR - Initializing")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Quantization: {'Enabled' if use_quantization else 'Disabled'}")
        print(f"{'='*60}\n")

        # Initialize SAM3 for detection + tracking
        self.sam3_tracker = None
        self.sam3_available = False

        if use_sam3:
            print("Loading SAM3 for detection + tracking...")
            try:
                self.sam3_tracker = SAM3PlateTracker(
                    device=self.device,
                    confidence_threshold=0.5,
                    text_prompt="license plate"
                )
                self.sam3_available = self.sam3_tracker.is_available()
                if self.sam3_available:
                    print(f"✓ SAM3 loaded (native tracking: {self.sam3_tracker.has_native_tracking()})")
                else:
                    print("⚠ SAM3 not available, using Qwen3-VL for detection")
            except Exception as e:
                logger.warning(f"Failed to load SAM3: {e}")
                print("⚠ SAM3 failed to load, using Qwen3-VL for detection")

        # Load Qwen3-VL for OCR (and detection fallback)
        print("Loading Qwen3-VL model for OCR...")
        self.qwen_detector = Qwen3VLDirectDetector(
            model_name=model_name,
            device=device,
            use_quantization=use_quantization,
            low_memory=low_memory
        )

        print(f"\n{'='*60}")
        print("LICENSE PLATE OCR - Ready")
        print(f"Detection: {'SAM3' if self.sam3_available else 'Qwen3-VL'}")
        print(f"OCR: Qwen3-VL")
        print(f"{'='*60}\n")

    def detect_and_read(
        self,
        image: Union[Image.Image, str, np.ndarray],
        frame_idx: Optional[int] = None
    ) -> Dict:
        """
        Detect license plates and extract text.

        Args:
            image: PIL Image, numpy array, or path to image
            frame_idx: Frame index for video tracking (optional)

        Returns:
            dict: {
                'plates': [
                    {
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float,
                        'plate_text': str,
                        'ocr_confidence': float,
                        'state': str,
                        'format': str,
                        'track_id': int (if tracking enabled)
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

        # Use SAM3 for detection if available
        if self.sam3_available and self.sam3_tracker:
            return self._detect_with_sam3(image, frame_idx)
        else:
            # Fallback to Qwen3-VL for detection
            return self._detect_with_qwen(image)

    def _detect_with_sam3(
        self,
        image: Image.Image,
        frame_idx: Optional[int] = None
    ) -> Dict:
        """
        Detect plates using SAM3 and read text using Qwen3-VL.

        SAM3 handles detection + tracking, Qwen handles OCR.
        Uses multi-frame voting for better accuracy when tracking.
        """
        logger.info("SAM3: Detecting and tracking plates...")

        # Get detections from SAM3
        detections = self.sam3_tracker.detect_and_track(image, frame_idx)

        if len(detections) == 0:
            logger.info("No plates detected by SAM3")
            return {'plates': [], 'num_plates': 0}

        logger.info(f"SAM3 detected {len(detections)} plate(s)")

        # For each detection, crop and run OCR with Qwen
        enhanced_plates = []
        for i, det in enumerate(detections):
            bbox = det.get('bbox', [])
            if len(bbox) != 4:
                continue

            logger.info(f"Qwen OCR: Reading plate {i+1}/{len(detections)}...")

            # Crop plate region with padding and preprocessing
            cropped_plate = self._crop_plate_region(image, bbox)

            # Run OCR on cropped plate using Qwen
            ocr_result = self._read_plate_text(cropped_plate)

            raw_text = ocr_result.get('text', 'UNREADABLE')
            ocr_confidence = ocr_result.get('confidence', 0.0)
            state = ocr_result.get('state', 'Unknown')

            # Try to identify state from plate format if not detected
            if state == 'Unknown' and raw_text != 'UNREADABLE':
                detected_state = identify_state_from_text(raw_text)
                if detected_state:
                    state = detected_state

            # Apply text correction based on detected format
            corrected_text = correct_ocr_text(raw_text, expected_format=state.lower() if state != 'Unknown' else None)

            # Use multi-frame voting if tracking is enabled
            track_id = det.get('track_id')
            final_text = corrected_text
            final_confidence = ocr_confidence

            if track_id is not None and self.ocr_voter is not None:
                # Add reading to voter and get voted result
                voted_text, voted_confidence = self.ocr_voter.add_reading(track_id, corrected_text)
                if voted_confidence > final_confidence:
                    final_text = voted_text
                    final_confidence = voted_confidence
                    logger.info(f"  Multi-frame voting: {voted_text} (conf: {voted_confidence:.2f})")

            plate = {
                'bbox': bbox,
                'confidence': det.get('confidence', 0.8),
                'plate_text': final_text,
                'ocr_confidence': final_confidence,
                'state': state,
                'format': ocr_result.get('format', 'Unknown'),
                'mask': det.get('mask'),
                'color': (255, 165, 0)
            }

            # Add track_id if available (from SAM3 tracking)
            if track_id is not None:
                plate['track_id'] = track_id

            enhanced_plates.append(plate)
            logger.info(f"  Plate text: {plate['plate_text']} (conf: {plate['ocr_confidence']:.2f})")

        return {
            'plates': enhanced_plates,
            'num_plates': len(enhanced_plates)
        }

    def _detect_with_qwen(
        self,
        image: Image.Image
    ) -> Dict:
        """
        Detect plates using Qwen3-VL (fallback when SAM3 unavailable).

        Two-stage: Detect -> Crop -> OCR
        """
        # Stage 1: Detect plate locations
        logger.info("Qwen: Detecting license plates...")
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
            logger.info(f"Qwen OCR: Reading plate {i+1}/{len(plates)}...")

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
        bbox: List[int],
        apply_preprocessing: bool = True
    ) -> Image.Image:
        """
        Crop plate region with padding and preprocessing for better OCR.

        Args:
            image: Full image
            bbox: [x1, y1, x2, y2] bounding box
            apply_preprocessing: Apply image preprocessing

        Returns:
            Cropped and preprocessed plate image
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

        cropped = image.crop((x1, y1, x2, y2))

        # Apply preprocessing if enabled
        if apply_preprocessing and self.enable_preprocessing:
            # Convert to numpy for preprocessing
            plate_np = np.array(cropped)

            # Check if plate is too blurry
            if is_plate_blurry(plate_np, threshold=50.0):
                logger.warning("Plate image is blurry, OCR may be less accurate")

            # Upscale small plates for better OCR
            plate_np = upscale_plate(plate_np)

            # Apply contrast enhancement (keep color for VLM)
            plate_np = preprocess_plate_for_ocr(
                plate_np,
                apply_grayscale=False,  # Keep color for VLM
                apply_clahe=True,
                apply_denoise=True,
                apply_sharpen=False,
                apply_deskew=False
            )

            # Convert back to PIL
            cropped = Image.fromarray(plate_np)

        return cropped

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
                    # Check aspect ratio - plates are wider than tall (ratio > 1.5)
                    # Wheels are square (ratio ~1.0), so reject those
                    box_width = x2 - x1
                    box_height = y2 - y1
                    aspect_ratio = box_width / box_height if box_height > 0 else 0

                    # License plates typically have aspect ratio 1.5 to 3.0
                    # Reject if too square (likely a wheel) or too narrow
                    if aspect_ratio < 1.3:
                        logger.info(f"Rejecting detection with aspect ratio {aspect_ratio:.2f} (likely wheel)")
                        continue

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

    def reset_tracker(self):
        """Reset tracker and OCR voter for new video."""
        if self.sam3_tracker:
            self.sam3_tracker.reset()
        if self.ocr_voter:
            self.ocr_voter.reset()

    def get_final_ocr_results(self) -> Dict[int, Tuple[str, float]]:
        """Get final voted OCR results for all tracked plates."""
        if self.ocr_voter:
            return self.ocr_voter.get_all_final_results()
        return {}

    def init_video(self, first_frame: Union[Image.Image, np.ndarray]):
        """Initialize video tracking with first frame."""
        if self.sam3_tracker:
            self.sam3_tracker.init_video(first_frame)

    def has_native_tracking(self) -> bool:
        """Check if native SAM3 tracking is available."""
        if self.sam3_tracker:
            return self.sam3_tracker.has_native_tracking()
        return False

    def cleanup(self):
        """Clean up models and free GPU memory."""
        if hasattr(self, 'qwen_detector'):
            self.qwen_detector.cleanup()
        if hasattr(self, 'sam3_tracker') and self.sam3_tracker:
            self.sam3_tracker.cleanup()
        if self.device == "cuda":
            torch.cuda.empty_cache()
            print("GPU memory cleared")
