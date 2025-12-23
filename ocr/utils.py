"""
Utility functions for License Plate OCR

- Image preprocessing for better OCR accuracy
- Drawing functions for visualizing plates
- North American plate format validation
- Post-processing and text correction
- Helper functions
"""

import cv2
import numpy as np
import re
from typing import List, Dict, Optional, Tuple
from PIL import Image
from collections import Counter
import logging

logger = logging.getLogger(__name__)

# Colors for drawing (BGR format for OpenCV)
PLATE_COLOR = (0, 165, 255)  # Orange
PLATE_TEXT_BG = (0, 100, 200)  # Dark orange

# Minimum plate height for OCR (upscale if smaller)
MIN_PLATE_HEIGHT = 80
MIN_PLATE_WIDTH = 200


# =============================================================================
# IMAGE PREPROCESSING FOR BETTER OCR
# =============================================================================

def preprocess_plate_for_ocr(
    image: np.ndarray,
    apply_grayscale: bool = True,
    apply_clahe: bool = True,
    apply_denoise: bool = True,
    apply_sharpen: bool = False,
    apply_deskew: bool = False
) -> np.ndarray:
    """
    Apply preprocessing pipeline to plate image for better OCR accuracy.

    Args:
        image: RGB or BGR numpy array of cropped plate
        apply_grayscale: Convert to grayscale
        apply_clahe: Apply CLAHE contrast enhancement
        apply_denoise: Apply denoising
        apply_sharpen: Apply sharpening filter
        apply_deskew: Attempt to deskew rotated text

    Returns:
        Preprocessed image (grayscale or RGB depending on options)
    """
    processed = image.copy()

    # Convert to grayscale if requested
    if apply_grayscale:
        if len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)

    # Apply CLAHE for contrast enhancement
    if apply_clahe:
        if len(processed.shape) == 2:  # Grayscale
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            processed = clahe.apply(processed)
        else:  # Color image - apply to L channel in LAB
            lab = cv2.cvtColor(processed, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            processed = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Apply denoising
    if apply_denoise:
        if len(processed.shape) == 2:
            processed = cv2.fastNlMeansDenoising(processed, h=10)
        else:
            processed = cv2.fastNlMeansDenoisingColored(processed, h=10, hColor=10)

    # Apply sharpening
    if apply_sharpen:
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        processed = cv2.filter2D(processed, -1, kernel)

    # Attempt deskew
    if apply_deskew:
        processed = deskew_plate(processed)

    return processed


def upscale_plate(
    image: np.ndarray,
    min_height: int = MIN_PLATE_HEIGHT,
    min_width: int = MIN_PLATE_WIDTH
) -> np.ndarray:
    """
    Upscale small plate images for better OCR accuracy.

    Args:
        image: Plate image (numpy array)
        min_height: Minimum height to upscale to
        min_width: Minimum width to upscale to

    Returns:
        Upscaled image if needed, otherwise original
    """
    h, w = image.shape[:2]

    # Calculate scale factor
    scale_h = min_height / h if h < min_height else 1.0
    scale_w = min_width / w if w < min_width else 1.0
    scale = max(scale_h, scale_w)

    if scale > 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        # Use LANCZOS for high-quality upscaling
        upscaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        logger.debug(f"Upscaled plate from {w}x{h} to {new_w}x{new_h}")
        return upscaled

    return image


def deskew_plate(image: np.ndarray) -> np.ndarray:
    """
    Deskew a rotated plate image.

    Args:
        image: Plate image (grayscale or color)

    Returns:
        Deskewed image
    """
    # Convert to grayscale for angle detection
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # Threshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find coordinates of non-zero pixels
    coords = np.column_stack(np.where(binary > 0))

    if len(coords) < 10:
        return image

    # Get the minimum area rectangle
    angle = cv2.minAreaRect(coords)[-1]

    # Adjust angle
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90

    # Only deskew if angle is significant but not too extreme
    if abs(angle) > 1.0 and abs(angle) < 15.0:
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, rotation_matrix, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        logger.debug(f"Deskewed plate by {angle:.1f} degrees")
        return rotated

    return image


def is_plate_blurry(image: np.ndarray, threshold: float = 100.0) -> bool:
    """
    Check if plate image is too blurry for reliable OCR.

    Args:
        image: Plate image
        threshold: Laplacian variance threshold (lower = more blurry)

    Returns:
        True if image is blurry
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold


def enhance_plate_contrast(image: np.ndarray) -> np.ndarray:
    """
    Enhance plate contrast using multiple techniques.

    Args:
        image: Plate image (RGB)

    Returns:
        Contrast-enhanced image
    """
    if len(image.shape) == 2:
        # Grayscale - use histogram equalization
        return cv2.equalizeHist(image)

    # Convert to LAB and enhance L channel
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    # Merge and convert back
    enhanced_lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

    return enhanced


# =============================================================================
# POST-PROCESSING AND TEXT CORRECTION
# =============================================================================

# Common OCR character confusions
OCR_CONFUSIONS = {
    # Letter to number (when in number position)
    'O': '0', 'o': '0',
    'I': '1', 'l': '1', 'i': '1',
    'S': '5', 's': '5',
    'B': '8',
    'Z': '2', 'z': '2',
    'G': '6', 'g': '6',
    'T': '7',
    'A': '4',
    # Number to letter (when in letter position)
    '0': 'O',
    '1': 'I',
    '5': 'S',
    '8': 'B',
    '2': 'Z',
    '6': 'G',
}


def correct_ocr_text(
    text: str,
    expected_format: Optional[str] = None
) -> str:
    """
    Correct common OCR mistakes in plate text.

    Args:
        text: Raw OCR text
        expected_format: Optional format hint (e.g., 'california', 'texas')

    Returns:
        Corrected text
    """
    if not text or text == 'UNREADABLE':
        return text

    # Clean up text first
    cleaned = text.upper().strip()
    cleaned = re.sub(r'[^A-Z0-9\s\-]', '', cleaned)

    # Format-specific corrections
    if expected_format == 'california':
        # Format: 1ABC234 (digit + 3 letters + 3 digits)
        if len(cleaned) == 7:
            corrected = []
            for i, char in enumerate(cleaned):
                if i == 0 or i >= 4:  # Should be digit
                    corrected.append(OCR_CONFUSIONS.get(char, char) if char.isalpha() else char)
                else:  # Should be letter (positions 1-3)
                    corrected.append(OCR_CONFUSIONS.get(char, char) if char.isdigit() else char)
            return ''.join(corrected)

    elif expected_format in ['texas', 'us_standard']:
        # Format: ABC-1234 (3 letters + 4 digits)
        cleaned_nodash = cleaned.replace('-', '').replace(' ', '')
        if len(cleaned_nodash) == 7:
            corrected = []
            for i, char in enumerate(cleaned_nodash):
                if i < 3:  # Should be letter
                    if char.isdigit():
                        corrected.append(OCR_CONFUSIONS.get(char, char))
                    else:
                        corrected.append(char)
                else:  # Should be digit
                    if char.isalpha():
                        corrected.append(OCR_CONFUSIONS.get(char, char))
                    else:
                        corrected.append(char)
            return ''.join(corrected[:3]) + '-' + ''.join(corrected[3:])

    elif expected_format == 'ontario':
        # Format: ABCD 123 (4 letters + 3 digits)
        cleaned_nospace = cleaned.replace(' ', '')
        if len(cleaned_nospace) == 7:
            corrected = []
            for i, char in enumerate(cleaned_nospace):
                if i < 4:  # Should be letter
                    if char.isdigit():
                        corrected.append(OCR_CONFUSIONS.get(char, char))
                    else:
                        corrected.append(char)
                else:  # Should be digit
                    if char.isalpha():
                        corrected.append(OCR_CONFUSIONS.get(char, char))
                    else:
                        corrected.append(char)
            return ''.join(corrected[:4]) + ' ' + ''.join(corrected[4:])

    return cleaned


def vote_ocr_results(ocr_results: List[str]) -> Tuple[str, float]:
    """
    Vote among multiple OCR results to get the best one.

    Args:
        ocr_results: List of OCR text results

    Returns:
        Tuple of (best_text, confidence)
    """
    if not ocr_results:
        return 'UNREADABLE', 0.0

    # Filter out unreadable results
    valid_results = [r for r in ocr_results if r and r != 'UNREADABLE']

    if not valid_results:
        return 'UNREADABLE', 0.0

    if len(valid_results) == 1:
        return valid_results[0], 0.7

    # Count occurrences
    counter = Counter(valid_results)
    most_common = counter.most_common(1)[0]
    best_text = most_common[0]
    count = most_common[1]

    # Confidence based on agreement
    confidence = count / len(valid_results)

    return best_text, confidence


def vote_ocr_by_character(ocr_results: List[str]) -> Tuple[str, float]:
    """
    Character-level voting for better accuracy with noisy OCR.

    Args:
        ocr_results: List of OCR text results

    Returns:
        Tuple of (best_text, confidence)
    """
    if not ocr_results:
        return 'UNREADABLE', 0.0

    # Filter and normalize
    valid_results = [r.upper().strip() for r in ocr_results if r and r != 'UNREADABLE']

    if not valid_results:
        return 'UNREADABLE', 0.0

    if len(valid_results) == 1:
        return valid_results[0], 0.7

    # Find the most common length
    lengths = Counter(len(r) for r in valid_results)
    target_length = lengths.most_common(1)[0][0]

    # Filter to matching length
    same_length = [r for r in valid_results if len(r) == target_length]

    if not same_length:
        return vote_ocr_results(valid_results)

    # Vote per character position
    result_chars = []
    total_confidence = 0.0

    for i in range(target_length):
        chars_at_pos = [r[i] for r in same_length]
        counter = Counter(chars_at_pos)
        best_char, count = counter.most_common(1)[0]
        result_chars.append(best_char)
        total_confidence += count / len(same_length)

    avg_confidence = total_confidence / target_length
    return ''.join(result_chars), avg_confidence


class MultiFrameOCRVoter:
    """
    Accumulate OCR results across multiple frames for tracked plates.
    Uses voting to determine the most likely correct plate text.
    """

    def __init__(self, min_votes: int = 3, max_history: int = 30):
        """
        Args:
            min_votes: Minimum number of readings before returning result
            max_history: Maximum readings to keep per track
        """
        self.min_votes = min_votes
        self.max_history = max_history
        self.track_history: Dict[int, List[str]] = {}
        self.track_results: Dict[int, Tuple[str, float]] = {}

    def add_reading(self, track_id: int, ocr_text: str) -> Tuple[str, float]:
        """
        Add an OCR reading for a tracked plate.

        Args:
            track_id: Tracking ID for the plate
            ocr_text: OCR result for this frame

        Returns:
            Tuple of (voted_text, confidence)
        """
        if track_id not in self.track_history:
            self.track_history[track_id] = []

        # Add new reading
        if ocr_text and ocr_text != 'UNREADABLE':
            self.track_history[track_id].append(ocr_text)

            # Trim history if needed
            if len(self.track_history[track_id]) > self.max_history:
                self.track_history[track_id] = self.track_history[track_id][-self.max_history:]

        # Vote on results
        history = self.track_history[track_id]
        if len(history) >= self.min_votes:
            voted_text, confidence = vote_ocr_by_character(history)
            self.track_results[track_id] = (voted_text, confidence)
            return voted_text, confidence
        elif history:
            # Not enough votes yet, return most recent
            return history[-1], 0.5
        else:
            return 'UNREADABLE', 0.0

    def get_result(self, track_id: int) -> Tuple[str, float]:
        """Get the current voted result for a track."""
        if track_id in self.track_results:
            return self.track_results[track_id]
        elif track_id in self.track_history and self.track_history[track_id]:
            return self.track_history[track_id][-1], 0.5
        return 'UNREADABLE', 0.0

    def reset(self):
        """Reset all tracking history."""
        self.track_history.clear()
        self.track_results.clear()

    def get_all_final_results(self) -> Dict[int, Tuple[str, float]]:
        """Get final voted results for all tracks."""
        results = {}
        for track_id in self.track_history:
            if self.track_history[track_id]:
                text, conf = vote_ocr_by_character(self.track_history[track_id])
                results[track_id] = (text, conf)
        return results


# =============================================================================
# DRAWING FUNCTIONS
# =============================================================================


def draw_plate_detections(
    image: np.ndarray,
    plates: List[Dict],
    show_text: bool = True
) -> np.ndarray:
    """
    Draw plate detections with bounding boxes and OCR text.

    Args:
        image: RGB numpy array
        plates: List of plate detection dicts
        show_text: Show OCR text on image

    Returns:
        Annotated image (RGB)
    """
    annotated = image.copy()

    for plate in plates:
        bbox = plate.get('bbox', [])
        plate_text = plate.get('plate_text', 'UNREADABLE')
        confidence = plate.get('confidence', 0.0)
        ocr_confidence = plate.get('ocr_confidence', 0.0)
        state = plate.get('state', 'Unknown')

        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = map(int, bbox)

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), PLATE_COLOR, 3)

        # Draw text label
        if show_text:
            # Primary label: plate text with track ID
            track_id = plate.get('track_id', None)
            label = f"{plate_text}"
            if track_id is not None:
                label = f"{track_id}.{plate_text}"
            if state != 'Unknown':
                label += f" ({state})"

            # Secondary label: confidence
            conf_label = f"Det: {confidence:.2f} | OCR: {ocr_confidence:.2f}"

            # Draw background for text
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            (conf_w, conf_h), _ = cv2.getTextSize(
                conf_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )

            # Background rectangles
            cv2.rectangle(
                annotated,
                (x1, y1 - text_h - 25),
                (x1 + max(text_w, conf_w) + 10, y1),
                PLATE_TEXT_BG,
                -1
            )

            # Plate text (larger)
            cv2.putText(
                annotated,
                label,
                (x1 + 5, y1 - conf_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            # Confidence text (smaller)
            cv2.putText(
                annotated,
                conf_label,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1
            )

    return annotated


def validate_plate_format(
    plate_text: str,
    expected_format: str = None
) -> Tuple[bool, str]:
    """
    Validate plate text against known North American formats.

    Args:
        plate_text: Extracted plate text
        expected_format: Optional specific format to validate against

    Returns:
        Tuple of (is_valid, matched_format or error)
    """
    if not plate_text or plate_text == 'UNREADABLE':
        return False, "Unreadable plate"

    # Clean up text
    clean_text = plate_text.upper().strip()
    clean_text = re.sub(r'[^A-Z0-9\s\-]', '', clean_text)

    # Format patterns
    patterns = {
        'california': r'^[0-9][A-Z]{3}[0-9]{3}$',
        'texas': r'^[A-Z]{3}-?[0-9]{4}$',
        'us_standard': r'^[A-Z]{3}[\s\-]?[0-9]{4}$',
        'us_standard_alt': r'^[0-9]{3}[\s\-]?[A-Z]{3}$',
        'ontario': r'^[A-Z]{4}[\s]?[0-9]{3}$',
        'quebec': r'^[0-9]{3}[\s]?[A-Z]{3}$',
        'mexico': r'^[A-Z]{3}-[0-9]{2}-[0-9]{2}$',
        'vanity': r'^[A-Z0-9]{1,8}$',
    }

    # Check specific format if provided
    if expected_format and expected_format in patterns:
        if re.match(patterns[expected_format], clean_text):
            return True, expected_format
        return False, f"Does not match {expected_format} format"

    # Check all formats
    for format_name, pattern in patterns.items():
        if re.match(pattern, clean_text.replace(' ', '').replace('-', '')):
            return True, format_name

    # If alphanumeric but no specific match, might still be valid
    if re.match(r'^[A-Z0-9\s\-]{4,10}$', clean_text):
        return True, "generic"

    return False, "Unknown format"


def identify_state_from_text(plate_text: str) -> Optional[str]:
    """
    Try to identify state/province from plate text patterns.

    Args:
        plate_text: Extracted plate text

    Returns:
        State name or None
    """
    if not plate_text:
        return None

    clean_text = plate_text.upper().strip()

    # California pattern: digit + 3 letters + 3 digits
    if re.match(r'^[0-9][A-Z]{3}[0-9]{3}$', clean_text):
        return "California"

    # Ontario pattern: 4 letters + 3 digits
    if re.match(r'^[A-Z]{4}\s?[0-9]{3}$', clean_text):
        return "Ontario"

    # Quebec pattern: 3 digits + 3 letters
    if re.match(r'^[0-9]{3}\s?[A-Z]{3}$', clean_text):
        return "Quebec"

    # Mexico pattern: ABC-12-34
    if re.match(r'^[A-Z]{3}-[0-9]{2}-[0-9]{2}$', clean_text):
        return "Mexico"

    return None


def create_plate_summary(plates: List[Dict]) -> Dict:
    """
    Create summary statistics for detected plates.

    Args:
        plates: List of plate detection dicts

    Returns:
        Summary dict
    """
    if not plates:
        return {
            'total_plates': 0,
            'readable_plates': 0,
            'avg_confidence': 0.0,
            'states': [],
            'plate_texts': []
        }

    readable = [p for p in plates if p.get('plate_text', 'UNREADABLE') != 'UNREADABLE']
    states = list(set(p.get('state', 'Unknown') for p in plates if p.get('state') != 'Unknown'))
    texts = [p.get('plate_text') for p in readable]

    avg_conf = sum(p.get('ocr_confidence', 0) for p in plates) / len(plates)

    return {
        'total_plates': len(plates),
        'readable_plates': len(readable),
        'avg_confidence': round(avg_conf, 2),
        'states': states,
        'plate_texts': texts
    }


def pil_to_cv2(image: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV BGR format."""
    rgb = np.array(image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def cv2_to_pil(image: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR to PIL Image."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)
