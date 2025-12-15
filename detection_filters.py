"""
Detection Post-Processing Filters

Reduces false positives by filtering detections based on:
- Mask properties (area, shape, texture)
- Context (location, other nearby detections)
- Confidence thresholds
"""
import numpy as np
from typing import List, Dict, Optional


class DetectionFilter:
    """Filter false positive detections."""

    def __init__(self, exclude_categories: Optional[List[str]] = None):
        """
        Initialize detection filter.

        Args:
            exclude_categories: List of categories to completely exclude (e.g., ["graffiti", "tyre_marks"])
        """
        # Categories to completely exclude from results
        self.exclude_categories = set(exclude_categories or [])

        # Minimum confidence thresholds per category
        self.min_confidence = {
            "homeless_person": 0.85,        # Higher threshold to reduce false positives
            "homeless_encampment": 0.75,
            "dumped_trash": 0.60,
            "abandoned_vehicle": 0.80,
            "graffiti": 0.90,               # Very high - model hallucinates graffiti often
            "tyre_marks": 0.85,             # Also prone to false positives
        }

        # Area constraints (in pixels)
        self.area_constraints = {
            "homeless_person": {
                "min": 5000,    # Person must be at least this many pixels
                "max": 200000   # Can't be too large
            },
            "graffiti": {
                "min": 2000,    # Graffiti must be visible size
                "max": 500000   # Can't be too large
            }
        }

    def filter_detections(
        self,
        detections: List[Dict],
        image_shape: tuple
    ) -> List[Dict]:
        """
        Filter detections to reduce false positives.

        Args:
            detections: List of detection dicts
            image_shape: (height, width) of image

        Returns:
            Filtered list of detections
        """
        filtered = []

        for det in detections:
            if self._should_keep_detection(det, image_shape):
                filtered.append(det)

        return filtered

    def _should_keep_detection(self, det: Dict, image_shape: tuple) -> bool:
        """Determine if detection should be kept."""
        label = det.get('label', '')
        confidence = det.get('confidence', 0.0)

        # Check if category is excluded
        if label in self.exclude_categories:
            return False

        # Check confidence threshold
        min_conf = self.min_confidence.get(label, 0.5)
        if confidence < min_conf:
            return False

        # Check area constraints
        if label in self.area_constraints:
            bbox = det.get('bbox', [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                area = (x2 - x1) * (y2 - y1)

                constraints = self.area_constraints[label]
                if area < constraints['min'] or area > constraints['max']:
                    return False

        # Special rules for specific categories
        if label == "homeless_person":
            return self._validate_homeless_person(det, image_shape)
        elif label == "graffiti":
            return self._validate_graffiti(det, image_shape)
        elif label == "tyre_marks":
            return self._validate_tyre_marks(det, image_shape)

        return True

    def _validate_homeless_person(self, det: Dict, image_shape: tuple) -> bool:
        """
        Additional validation for homeless_person detections.

        Checks:
        - Aspect ratio (person should be roughly vertical)
        - Mask properties if available
        """
        bbox = det.get('bbox', [])
        if len(bbox) != 4:
            return True

        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        if width <= 0 or height <= 0:
            return False

        # Check aspect ratio - person should be taller than wide
        # Piles of trash tend to be wider than tall
        aspect_ratio = height / width

        # Person should have aspect ratio between 0.8 and 3.5
        # (sitting to standing positions)
        if aspect_ratio < 0.8:
            # Too wide - likely trash pile or other object
            return False

        if aspect_ratio > 4.0:
            # Too tall - likely pole or sign
            return False

        # Check mask if available
        has_mask = det.get('has_mask', False)
        mask = det.get('mask', None)

        if has_mask and mask is not None:
            return self._validate_person_mask(mask)

        return True

    def _validate_person_mask(self, mask) -> bool:
        """
        Validate mask properties for person detection.

        A person's mask should have certain characteristics:
        - Connected regions (not too fragmented)
        - Reasonable compactness
        """
        try:
            # Convert mask to numpy array
            if isinstance(mask, list):
                mask_array = np.array(mask, dtype=np.uint8)
            else:
                mask_array = np.array(mask, dtype=np.uint8)

            # Ensure 2D
            if mask_array.ndim > 2:
                mask_array = mask_array.squeeze()

            # Calculate mask properties
            total_pixels = np.sum(mask_array > 0)

            if total_pixels < 3000:
                # Too small to be a person
                return False

            # Check fragmentation - count number of separate regions
            import cv2
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_array, connectivity=8)

            # Subtract 1 for background
            num_regions = num_labels - 1

            if num_regions > 5:
                # Too fragmented - likely trash pile or debris
                return False

            # Check if largest region is dominant (>60% of total)
            if num_regions > 1:
                # Get areas of all regions (excluding background)
                region_areas = stats[1:, cv2.CC_STAT_AREA]
                largest_area = np.max(region_areas)

                if largest_area / total_pixels < 0.6:
                    # No dominant region - likely trash pile
                    return False

            return True

        except Exception:
            # If validation fails, keep detection
            return True

    def _validate_graffiti(self, det: Dict, image_shape: tuple) -> bool:
        """
        Validate graffiti detection to reduce hallucinations.

        Graffiti should:
        - Have high confidence (model often hallucinates this)
        - Be on vertical surfaces (walls, signs)
        - Have certain aspect ratios (not too thin/wide)
        - Not be tiny or huge
        """
        bbox = det.get('bbox', [])
        confidence = det.get('confidence', 0.0)

        # Very high confidence required due to frequent hallucinations
        if confidence < 0.90:
            return False

        if len(bbox) != 4:
            return True

        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        if width <= 0 or height <= 0:
            return False

        # Check aspect ratio - graffiti shouldn't be extremely thin or wide
        aspect_ratio = height / width

        # Graffiti is usually somewhat horizontal (text/tags) or square
        # Filter out very tall thin detections (likely poles, edges, etc.)
        if aspect_ratio > 3.0 or aspect_ratio < 0.2:
            return False

        # Check position - graffiti is usually not in sky/top portion
        image_height = image_shape[0]
        center_y = (y1 + y2) / 2

        # If detection is in top 20% of image, likely not graffiti
        if center_y < image_height * 0.2:
            return False

        return True

    def _validate_tyre_marks(self, det: Dict, image_shape: tuple) -> bool:
        """
        Validate tyre marks detection.

        Tyre marks should:
        - Be on road surface (bottom portion of image)
        - Be horizontal/elongated
        - Have reasonable confidence
        """
        bbox = det.get('bbox', [])
        confidence = det.get('confidence', 0.0)

        # Higher confidence required
        if confidence < 0.85:
            return False

        if len(bbox) != 4:
            return True

        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        if width <= 0 or height <= 0:
            return False

        # Tyre marks are usually wider than tall (horizontal)
        aspect_ratio = height / width

        # Should be quite horizontal
        if aspect_ratio > 0.5:
            return False

        # Should be in lower portion of image (road surface)
        image_height = image_shape[0]
        center_y = (y1 + y2) / 2

        # If detection is in top 40% of image, likely not tyre marks
        if center_y < image_height * 0.4:
            return False

        return True


class ContextualFilter:
    """Filter detections based on context and nearby detections."""

    def filter_by_context(self, detections: List[Dict]) -> List[Dict]:
        """
        Filter detections based on contextual information.

        For example:
        - If "homeless_person" and "dumped_trash" overlap significantly,
          prefer "dumped_trash"
        """
        filtered = detections.copy()

        # Check for overlapping homeless_person and dumped_trash
        homeless_indices = [i for i, d in enumerate(filtered) if d.get('label') == 'homeless_person']
        trash_indices = [i for i, d in enumerate(filtered) if d.get('label') == 'dumped_trash']

        to_remove = set()

        for h_idx in homeless_indices:
            h_bbox = filtered[h_idx].get('bbox', [])
            if len(h_bbox) != 4:
                continue

            for t_idx in trash_indices:
                t_bbox = filtered[t_idx].get('bbox', [])
                if len(t_bbox) != 4:
                    continue

                # Calculate overlap
                overlap = self._calculate_iou(h_bbox, t_bbox)

                if overlap > 0.5:
                    # Significant overlap - prefer trash detection
                    to_remove.add(h_idx)
                    break

        # Remove filtered detections
        return [d for i, d in enumerate(filtered) if i not in to_remove]

    def _calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union for two bboxes."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        # Calculate intersection
        x_inter_min = max(x1_min, x2_min)
        y_inter_min = max(y1_min, y2_min)
        x_inter_max = min(x1_max, x2_max)
        y_inter_max = min(y1_max, y2_max)

        if x_inter_max <= x_inter_min or y_inter_max <= y_inter_min:
            return 0.0

        inter_area = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)

        # Calculate union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0


# Convenience function
def apply_filters(
    detections: List[Dict],
    image_shape: tuple,
    exclude_categories: Optional[List[str]] = None
) -> List[Dict]:
    """
    Apply all detection filters.

    Args:
        detections: List of raw detections
        image_shape: (height, width) of image
        exclude_categories: Categories to completely exclude (e.g., ["graffiti"])

    Returns:
        Filtered detections
    """
    # Apply detection filter
    det_filter = DetectionFilter(exclude_categories=exclude_categories)
    filtered = det_filter.filter_detections(detections, image_shape)

    # Apply contextual filter
    ctx_filter = ContextualFilter()
    filtered = ctx_filter.filter_by_context(filtered)

    return filtered
