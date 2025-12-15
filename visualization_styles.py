"""
Modern Visualization Styles for Infrastructure Detection

Provides professional styling with:
- Severity-based color palette with better contrast
- Rounded rectangles for labels
- Text shadows and outlines for readability
- Visual hierarchy and modern design
"""
import cv2
import numpy as np
from typing import Tuple, Optional


# Modern color palette (BGR format) grouped by severity
MODERN_COLORS = {
    # CRITICAL - Red tones (requires immediate attention)
    "potholes": (40, 50, 240),                    # Bright red
    "alligator_cracks": (60, 90, 255),            # Orange-red

    # HIGH PRIORITY - Orange/Yellow tones
    "transverse_cracks": (0, 140, 255),           # Deep orange
    "longitudinal_cracks": (0, 200, 255),         # Orange
    "damaged_crosswalks": (0, 180, 240),          # Dark orange
    "damaged_paint": (0, 165, 255),               # Medium orange

    # SOCIAL ISSUES - Purple/Magenta tones
    "homeless_encampment": (180, 50, 200),        # Purple
    "homeless_person": (220, 80, 255),            # Magenta
    "abandoned_vehicle": (140, 0, 180),           # Dark purple
    "dumped_trash": (160, 60, 180),               # Purple-gray

    # INFRASTRUCTURE - Blue/Cyan tones
    "manholes": (200, 150, 50),                   # Steel blue
    "street_signs": (255, 200, 0),                # Bright cyan-blue
    "traffic_lights": (220, 180, 0),              # Deep cyan

    # MINOR - Green/Gray tones
    "tyre_marks": (100, 120, 100),                # Muted green-gray
    "graffiti": (180, 100, 220),                  # Pink-purple
}


# Severity levels for visual hierarchy
SEVERITY_LEVELS = {
    "critical": ["potholes", "alligator_cracks"],
    "high": ["transverse_cracks", "longitudinal_cracks", "damaged_crosswalks", "damaged_paint"],
    "social": ["homeless_encampment", "homeless_person", "abandoned_vehicle", "dumped_trash"],
    "infrastructure": ["manholes", "street_signs", "traffic_lights"],
    "minor": ["tyre_marks", "graffiti"]
}


# Styling configuration
STYLE_CONFIG = {
    "bbox_thickness": {
        "critical": 4,
        "high": 3,
        "social": 3,
        "infrastructure": 2,
        "minor": 2
    },
    "label_font_scale": {
        "critical": 0.7,
        "high": 0.65,
        "social": 0.65,
        "infrastructure": 0.6,
        "minor": 0.55
    },
    "label_thickness": 2,
    "label_padding": 10,
    "label_corner_radius": 8,
    "text_shadow_offset": 2,
    "mask_alpha": 0.35,
    "contour_thickness": 2
}


def get_severity_level(category: str) -> str:
    """Get severity level for a category."""
    for level, categories in SEVERITY_LEVELS.items():
        if category in categories:
            return level
    return "minor"


def get_color(category: str) -> Tuple[int, int, int]:
    """Get color for a category (BGR format)."""
    return MODERN_COLORS.get(category, (100, 200, 100))


def get_bbox_thickness(category: str) -> int:
    """Get bounding box thickness based on severity."""
    severity = get_severity_level(category)
    return STYLE_CONFIG["bbox_thickness"][severity]


def get_font_scale(category: str) -> float:
    """Get font scale based on severity."""
    severity = get_severity_level(category)
    return STYLE_CONFIG["label_font_scale"][severity]


def draw_rounded_rectangle(
    image: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = -1,
    radius: int = 10
) -> np.ndarray:
    """
    Draw a rounded rectangle.

    Args:
        image: Image to draw on
        pt1: Top-left corner (x, y)
        pt2: Bottom-right corner (x, y)
        color: Color in BGR format
        thickness: Line thickness (-1 for filled)
        radius: Corner radius in pixels

    Returns:
        Modified image
    """
    x1, y1 = pt1
    x2, y2 = pt2

    # Ensure coordinates are valid
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    # Limit radius to half of smallest dimension
    max_radius = min((x2 - x1) // 2, (y2 - y1) // 2)
    radius = min(radius, max_radius)

    if radius <= 0:
        # Fall back to regular rectangle if radius is too small
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        return image

    # Draw filled rounded rectangle
    if thickness < 0:
        # Top rectangle
        cv2.rectangle(image, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        # Left rectangle
        cv2.rectangle(image, (x1, y1 + radius), (x2, y2 - radius), color, -1)

        # Four corners
        cv2.circle(image, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(image, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(image, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(image, (x2 - radius, y2 - radius), radius, color, -1)
    else:
        # Draw outline only
        # Top and bottom lines
        cv2.line(image, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
        cv2.line(image, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
        # Left and right lines
        cv2.line(image, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
        cv2.line(image, (x2, y1 + radius), (x2, y2 - radius), color, thickness)

        # Four corner arcs
        cv2.ellipse(image, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(image, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(image, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(image, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)

    return image


def draw_text_with_shadow(
    image: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font_scale: float = 0.6,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
    shadow_color: Tuple[int, int, int] = (0, 0, 0),
    shadow_offset: int = 2
) -> np.ndarray:
    """
    Draw text with shadow for better readability.

    Args:
        image: Image to draw on
        text: Text to draw
        position: Text position (x, y)
        font_scale: Font scale
        color: Text color (BGR)
        thickness: Text thickness
        shadow_color: Shadow color (BGR)
        shadow_offset: Shadow offset in pixels

    Returns:
        Modified image
    """
    x, y = position
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Draw shadow (offset)
    cv2.putText(
        image,
        text,
        (x + shadow_offset, y + shadow_offset),
        font,
        font_scale,
        shadow_color,
        thickness + 1,
        cv2.LINE_AA
    )

    # Draw main text
    cv2.putText(
        image,
        text,
        (x, y),
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA
    )

    return image


def draw_stylish_label(
    image: np.ndarray,
    text: str,
    position: Tuple[int, int],
    color: Tuple[int, int, int],
    font_scale: float = 0.6,
    padding: int = 10,
    corner_radius: int = 8
) -> np.ndarray:
    """
    Draw a stylish label with rounded background and shadowed text.

    Args:
        image: Image to draw on
        text: Label text
        position: Label position (x, y) - top-left corner
        color: Background color (BGR)
        font_scale: Font scale
        padding: Padding around text
        corner_radius: Corner radius for rounded rectangle

    Returns:
        Modified image
    """
    x, y = position
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = STYLE_CONFIG["label_thickness"]

    # Get text size
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Ensure label doesn't go out of bounds
    x = max(0, x)
    y = max(text_h + padding, y)

    # Draw rounded background
    bg_x1 = x
    bg_y1 = y - text_h - padding
    bg_x2 = x + text_w + padding * 2
    bg_y2 = y + padding // 2

    # Ensure background is within image bounds
    bg_x2 = min(bg_x2, image.shape[1])
    bg_y2 = min(bg_y2, image.shape[0])

    draw_rounded_rectangle(
        image,
        (bg_x1, bg_y1),
        (bg_x2, bg_y2),
        color,
        -1,
        corner_radius
    )

    # Draw text with shadow
    text_x = x + padding
    text_y = y - padding // 2

    draw_text_with_shadow(
        image,
        text,
        (text_x, text_y),
        font_scale,
        (255, 255, 255),
        thickness,
        (0, 0, 0),
        STYLE_CONFIG["text_shadow_offset"]
    )

    return image


def draw_stylish_detection(
    image: np.ndarray,
    detection: dict,
    draw_mask: bool = True
) -> np.ndarray:
    """
    Draw a single detection with modern styling.

    Args:
        image: Image to draw on (RGB format)
        detection: Detection dict with label, bbox, confidence, mask, etc.
        draw_mask: Whether to draw segmentation mask

    Returns:
        Modified image
    """
    label = detection.get('label', 'unknown')
    bbox = detection.get('bbox', [])
    confidence = detection.get('confidence', 0.0)
    has_mask = detection.get('has_mask', False)
    mask = detection.get('mask', None)

    if len(bbox) != 4:
        return image

    x1, y1, x2, y2 = map(int, bbox)

    # Get color and styling
    color_bgr = get_color(label)
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    bbox_thickness = get_bbox_thickness(label)
    font_scale = get_font_scale(label)

    # Create mask overlay if available
    mask_overlay = None
    if draw_mask and has_mask and mask is not None:
        try:
            # Convert mask to numpy array
            if isinstance(mask, list):
                mask_array = np.array(mask, dtype=np.uint8)
            else:
                mask_array = np.array(mask, dtype=np.uint8)

            # Ensure mask is 2D
            if mask_array.ndim > 2:
                mask_array = mask_array.squeeze()

            # Resize mask if needed
            if mask_array.shape != (image.shape[0], image.shape[1]):
                mask_array = cv2.resize(
                    mask_array,
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )

            # Create colored mask overlay
            mask_overlay = np.zeros_like(image)
            mask_overlay[mask_array > 0] = color_rgb

            # Draw mask contour
            contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, color_rgb, STYLE_CONFIG["contour_thickness"])

        except Exception as e:
            mask_overlay = None

    # Draw bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), color_rgb, bbox_thickness)

    # Create label text
    mask_status = "✓" if has_mask and mask_overlay is not None else ""
    label_text = f"{label} {confidence:.2f} {mask_status}".strip()

    # Draw stylish label
    draw_stylish_label(
        image,
        label_text,
        (x1, y1),
        color_rgb,
        font_scale,
        STYLE_CONFIG["label_padding"],
        STYLE_CONFIG["label_corner_radius"]
    )

    return image, mask_overlay


def draw_stylish_detections(
    image: np.ndarray,
    detections: list,
    draw_masks: bool = True
) -> np.ndarray:
    """
    Draw all detections with modern styling.

    Args:
        image: Image to draw on (RGB format)
        detections: List of detection dicts
        draw_masks: Whether to draw segmentation masks

    Returns:
        Annotated image
    """
    annotated = image.copy()

    # Collect all mask overlays
    mask_overlays = []

    for det in detections:
        annotated, mask_overlay = draw_stylish_detection(annotated, det, draw_masks)
        if mask_overlay is not None:
            mask_overlays.append(mask_overlay)

    # Blend all mask overlays
    if mask_overlays:
        combined_mask = np.zeros_like(annotated)
        for mask_overlay in mask_overlays:
            # Add masks together (will blend colors)
            combined_mask = cv2.addWeighted(combined_mask, 1.0, mask_overlay, 1.0, 0)

        # Blend combined mask with annotated image
        annotated = cv2.addWeighted(
            annotated,
            1.0 - STYLE_CONFIG["mask_alpha"],
            combined_mask,
            STYLE_CONFIG["mask_alpha"],
            0
        )

    return annotated
