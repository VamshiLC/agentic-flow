"""
Utility functions for License Plate OCR

- Drawing functions for visualizing plates
- North American plate format validation
- Helper functions
"""

import cv2
import numpy as np
import re
from typing import List, Dict, Optional, Tuple
from PIL import Image


# Colors for drawing (BGR format for OpenCV)
PLATE_COLOR = (0, 165, 255)  # Orange
PLATE_TEXT_BG = (0, 100, 200)  # Dark orange


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
