"""
SAM3 Exemplar-Based Segmentation

Uses SAM3's bbox prompting capability for exemplar-based segmentation.
When exemplars with bounding boxes are available, SAM3 can find all
instances matching the exemplar pattern.

Methods:
1. Semantic Predictor Mode: Uses SAM3SemanticPredictor with bbox prompts
2. Feature Matching Mode: Extract features from exemplars and match in target
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image

from .exemplar_manager import ExemplarManager, Exemplar

logger = logging.getLogger(__name__)


class SAM3ExemplarSegmenter:
    """
    SAM3 segmentation using exemplar-based prompting.

    Uses SAM3's native capability to find all instances matching
    exemplar bounding boxes.

    Example:
        segmenter = SAM3ExemplarSegmenter(sam3_processor, exemplar_manager)

        # Segment using exemplars
        results = segmenter.segment_with_exemplars(
            target_image=image,
            category="potholes"
        )
    """

    def __init__(
        self,
        sam3_processor,
        exemplar_manager: ExemplarManager,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize SAM3 exemplar segmenter.

        Args:
            sam3_processor: SAM3 processor instance (Sam3Processor)
            exemplar_manager: ExemplarManager with loaded exemplars
            confidence_threshold: Minimum confidence for masks
        """
        self.sam3_processor = sam3_processor
        self.exemplar_manager = exemplar_manager
        self.confidence_threshold = confidence_threshold

        # Try to import Ultralytics SAM3 for semantic predictor
        self._ultralytics_available = False
        try:
            from ultralytics.models.sam import SAM3SemanticPredictor
            self._ultralytics_available = True
            logger.info("Ultralytics SAM3SemanticPredictor available")
        except ImportError:
            logger.info("Ultralytics not available, using native SAM3 mode")

    def segment_with_exemplars(
        self,
        target_image: Image.Image,
        category: str,
        use_semantic_predictor: bool = True,
        max_exemplars: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Segment target image using exemplar-based prompting.

        Two modes available:
        1. Semantic predictor mode (if Ultralytics available):
           - Uses SAM3SemanticPredictor with bbox prompts
           - Finds ALL instances matching exemplar pattern

        2. Feature matching mode (fallback):
           - Extract features from exemplar bboxes
           - Match features in target image
           - Use point prompts at match locations

        Args:
            target_image: PIL Image to segment
            category: Category to segment
            use_semantic_predictor: Use SAM3SemanticPredictor if available
            max_exemplars: Maximum exemplars to use

        Returns:
            List of detection dicts with masks:
            [
                {
                    "mask": np.ndarray,
                    "bbox": [x1, y1, x2, y2],
                    "score": float,
                    "category": str,
                    "source": "exemplar"
                }
            ]
        """
        # Get exemplars with bboxes
        sam3_data = self.exemplar_manager.prepare_for_sam3(category, max_exemplars)

        if not sam3_data["has_bboxes"]:
            logger.info(f"No exemplars with bboxes for {category}, using text prompt")
            return self._segment_with_text_fallback(target_image, category)

        # Try semantic predictor first
        if use_semantic_predictor and self._ultralytics_available:
            return self._semantic_predictor_segment(
                target_image,
                sam3_data["bboxes"],
                category
            )
        else:
            return self._feature_matching_segment(
                target_image,
                sam3_data["exemplar_images"],
                sam3_data["bboxes"],
                category
            )

    def _semantic_predictor_segment(
        self,
        target_image: Image.Image,
        exemplar_bboxes: List[List[int]],
        category: str
    ) -> List[Dict[str, Any]]:
        """
        Use SAM3SemanticPredictor with bbox prompts.

        This method uses Ultralytics' SAM3SemanticPredictor which can
        take bounding box prompts and find all similar instances.
        """
        try:
            from ultralytics.models.sam import SAM3SemanticPredictor

            # Initialize predictor
            overrides = dict(
                conf=self.confidence_threshold,
                task="segment",
                mode="predict",
                model="sam3.pt",
                verbose=False
            )
            predictor = SAM3SemanticPredictor(overrides=overrides)

            # Convert PIL to path or numpy
            import tempfile
            import os

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                target_image.save(f.name)
                temp_path = f.name

            try:
                # Set image
                predictor.set_image(temp_path)

                # Run with bbox prompts
                results = predictor(bboxes=exemplar_bboxes, save=False)

                # Process results
                detections = []
                if results and len(results) > 0:
                    result = results[0]

                    # Extract masks and boxes
                    if hasattr(result, 'masks') and result.masks is not None:
                        masks = result.masks.data.cpu().numpy()
                        boxes = result.boxes.xyxy.cpu().numpy() if hasattr(result, 'boxes') else []
                        scores = result.boxes.conf.cpu().numpy() if hasattr(result, 'boxes') else []

                        for i, mask in enumerate(masks):
                            bbox = boxes[i].tolist() if i < len(boxes) else self._mask_to_bbox(mask)
                            score = float(scores[i]) if i < len(scores) else 0.8

                            detections.append({
                                "mask": mask,
                                "bbox": bbox,
                                "score": score,
                                "category": category,
                                "source": "exemplar_semantic"
                            })

                logger.info(f"SAM3 semantic predictor found {len(detections)} instances for {category}")
                return detections

            finally:
                os.unlink(temp_path)

        except Exception as e:
            logger.warning(f"Semantic predictor failed: {e}, falling back to feature matching")
            return self._feature_matching_segment(
                target_image, [], exemplar_bboxes, category
            )

    def _feature_matching_segment(
        self,
        target_image: Image.Image,
        exemplar_images: List[Image.Image],
        exemplar_bboxes: List[List[int]],
        category: str
    ) -> List[Dict[str, Any]]:
        """
        Fallback: Feature extraction and matching.

        1. Extract visual features from exemplar regions
        2. Find similar regions in target image using template matching
        3. Use SAM3 point prompts at match locations
        """
        import cv2

        detections = []
        target_np = np.array(target_image)

        # Convert RGB to BGR for OpenCV
        if len(target_np.shape) == 3 and target_np.shape[2] == 3:
            target_bgr = cv2.cvtColor(target_np, cv2.COLOR_RGB2BGR)
        else:
            target_bgr = target_np

        # Process each exemplar
        for i, (ex_img, bbox) in enumerate(zip(exemplar_images, exemplar_bboxes)):
            if ex_img is None:
                continue

            try:
                # Extract exemplar region
                ex_np = np.array(ex_img)
                x1, y1, x2, y2 = bbox

                # Ensure bbox is within image bounds
                h, w = ex_np.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                template = ex_np[y1:y2, x1:x2]

                if template.size == 0:
                    continue

                # Convert template to BGR
                if len(template.shape) == 3 and template.shape[2] == 3:
                    template_bgr = cv2.cvtColor(template, cv2.COLOR_RGB2BGR)
                else:
                    template_bgr = template

                # Resize template if too large
                th, tw = template_bgr.shape[:2]
                target_h, target_w = target_bgr.shape[:2]

                if tw > target_w // 2 or th > target_h // 2:
                    scale = min(target_w // 2 / tw, target_h // 2 / th)
                    template_bgr = cv2.resize(
                        template_bgr,
                        (int(tw * scale), int(th * scale))
                    )
                    th, tw = template_bgr.shape[:2]

                # Template matching
                result = cv2.matchTemplate(
                    target_bgr, template_bgr, cv2.TM_CCOEFF_NORMED
                )

                # Find matches above threshold
                threshold = 0.5
                locations = np.where(result >= threshold)

                for pt_y, pt_x in zip(*locations):
                    # Calculate center point for SAM3
                    center_x = pt_x + tw // 2
                    center_y = pt_y + th // 2

                    # Use SAM3 point prompt
                    mask_result = self._segment_at_point(
                        target_image, center_x, center_y
                    )

                    if mask_result is not None:
                        match_score = float(result[pt_y, pt_x])

                        detections.append({
                            "mask": mask_result["mask"],
                            "bbox": mask_result["bbox"],
                            "score": match_score * mask_result.get("score", 0.8),
                            "category": category,
                            "source": "exemplar_matching",
                            "match_location": [int(pt_x), int(pt_y)]
                        })

            except Exception as e:
                logger.warning(f"Feature matching failed for exemplar {i}: {e}")
                continue

        # Remove duplicates based on IoU
        detections = self._remove_duplicate_detections(detections)

        logger.info(f"Feature matching found {len(detections)} instances for {category}")
        return detections

    def _segment_at_point(
        self,
        image: Image.Image,
        x: int,
        y: int
    ) -> Optional[Dict[str, Any]]:
        """Use SAM3 to segment at a specific point."""
        try:
            # Set image if not already set
            inference_state = self.sam3_processor.set_image(image)

            # Point prompt
            output = self.sam3_processor.set_point_prompt(
                state=inference_state,
                point=[[x, y]],
                labels=[1]  # Foreground
            )

            # Extract mask
            if output is None:
                return None

            masks = None
            if hasattr(output, 'masks'):
                masks = output.masks
            elif isinstance(output, dict) and 'masks' in output:
                masks = output['masks']

            if masks is None or len(masks) == 0:
                return None

            # Get first mask
            mask = masks[0]
            if hasattr(mask, 'cpu'):
                mask = mask.cpu().numpy()
            if hasattr(mask, 'numpy'):
                mask = mask.numpy()

            # Ensure 2D
            while len(mask.shape) > 2:
                mask = mask[0]

            # Get bbox from mask
            bbox = self._mask_to_bbox(mask)

            # Get score
            score = 0.8
            if hasattr(output, 'scores') and len(output.scores) > 0:
                score = float(output.scores[0])
            elif isinstance(output, dict) and 'scores' in output:
                scores = output['scores']
                if len(scores) > 0:
                    score = float(scores[0])

            return {
                "mask": mask,
                "bbox": bbox,
                "score": score
            }

        except Exception as e:
            logger.warning(f"Point segmentation failed at ({x}, {y}): {e}")
            return None

    def _segment_with_text_fallback(
        self,
        target_image: Image.Image,
        category: str
    ) -> List[Dict[str, Any]]:
        """Fallback to text-based segmentation when no exemplar bboxes available."""
        try:
            # Set image
            inference_state = self.sam3_processor.set_image(target_image)

            # Text prompt using category name
            display_names = {
                "potholes": "pothole",
                "alligator_cracks": "cracked pavement",
                "manholes": "manhole cover",
                "damaged_crosswalks": "crosswalk",
            }
            text_prompt = display_names.get(category, category.replace("_", " "))

            output = self.sam3_processor.set_text_prompt(
                state=inference_state,
                prompt=text_prompt
            )

            if output is None:
                return []

            # Extract masks
            detections = []
            masks = None

            if hasattr(output, 'masks'):
                masks = output.masks
            elif isinstance(output, dict) and 'masks' in output:
                masks = output['masks']

            if masks is None:
                return []

            scores = []
            if hasattr(output, 'scores'):
                scores = output.scores
            elif isinstance(output, dict) and 'scores' in output:
                scores = output['scores']

            for i, mask in enumerate(masks):
                if hasattr(mask, 'cpu'):
                    mask = mask.cpu().numpy()
                if hasattr(mask, 'numpy'):
                    mask = mask.numpy()

                while len(mask.shape) > 2:
                    mask = mask[0]

                score = float(scores[i]) if i < len(scores) else 0.7
                bbox = self._mask_to_bbox(mask)

                detections.append({
                    "mask": mask,
                    "bbox": bbox,
                    "score": score,
                    "category": category,
                    "source": "text_fallback"
                })

            return detections

        except Exception as e:
            logger.error(f"Text fallback segmentation failed: {e}")
            return []

    def _mask_to_bbox(self, mask: np.ndarray) -> List[int]:
        """Convert binary mask to bounding box [x1, y1, x2, y2]."""
        if mask.sum() == 0:
            return [0, 0, 0, 0]

        rows = np.any(mask > 0.5, axis=1)
        cols = np.any(mask > 0.5, axis=0)

        if not rows.any() or not cols.any():
            return [0, 0, 0, 0]

        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]

        return [int(x1), int(y1), int(x2), int(y2)]

    def _remove_duplicate_detections(
        self,
        detections: List[Dict[str, Any]],
        iou_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Remove duplicate detections based on mask IoU."""
        if len(detections) <= 1:
            return detections

        # Sort by score descending
        detections = sorted(detections, key=lambda x: x["score"], reverse=True)

        keep = []
        for det in detections:
            is_duplicate = False

            for kept in keep:
                iou = self._calculate_mask_iou(det["mask"], kept["mask"])
                if iou > iou_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                keep.append(det)

        return keep

    def _calculate_mask_iou(
        self,
        mask1: np.ndarray,
        mask2: np.ndarray
    ) -> float:
        """Calculate IoU between two binary masks."""
        # Ensure same shape
        if mask1.shape != mask2.shape:
            return 0.0

        # Binarize
        m1 = mask1 > 0.5
        m2 = mask2 > 0.5

        intersection = np.logical_and(m1, m2).sum()
        union = np.logical_or(m1, m2).sum()

        if union == 0:
            return 0.0

        return intersection / union

    def segment_all_categories(
        self,
        target_image: Image.Image,
        categories: Optional[List[str]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Segment all categories that have exemplars.

        Args:
            target_image: Image to segment
            categories: Optional list of categories (defaults to all with exemplars)

        Returns:
            Dict mapping category to list of detections
        """
        if categories is None:
            categories = self.exemplar_manager.get_all_categories()

        results = {}

        for category in categories:
            if self.exemplar_manager.has_exemplars(category):
                detections = self.segment_with_exemplars(target_image, category)
                if detections:
                    results[category] = detections

        return results
