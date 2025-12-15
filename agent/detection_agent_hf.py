"""
Infrastructure Detection Agent using Qwen3-VL (Hugging Face) + SAM3

This implements the agentic flow where:
1. Qwen3-VL detects and describes infrastructure issues
2. SAM3 segments each detection based on Qwen's description

No vLLM server required - uses Hugging Face Transformers directly.
"""
import re
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Optional, Union, Tuple
import logging

from models.qwen_direct_loader import Qwen3VLDirectDetector
from models.sam3_loader import load_sam3_model
from prompts.category_prompts import build_detailed_prompt

logger = logging.getLogger(__name__)


class InfrastructureDetectionAgentHF:
    """
    Agentic infrastructure detector using Qwen3-VL (HF) + SAM3.

    Architecture:
    - Qwen3-VL: Detects infrastructure issues and provides descriptions
    - SAM3: Segments each detection based on description (acts as a tool)

    This is the agentic pattern without vLLM server dependency.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        sam3_processor=None,
        categories: Optional[List[str]] = None,
        device: Optional[str] = None,
        use_quantization: bool = False,
        low_memory: bool = False,
        sam3_confidence: float = 0.3,  # Lower threshold to accept more masks
        exclude_categories: Optional[List[str]] = None
    ):
        """
        Initialize the agentic detector.

        Args:
            model_name: Hugging Face model ID for Qwen3-VL
            sam3_processor: Pre-loaded SAM3 processor (or None to load)
            categories: List of infrastructure categories to detect
            device: Device to use ("cuda", "cpu", or None for auto)
            use_quantization: Use 8-bit quantization for Qwen
            low_memory: Enable memory optimizations
            sam3_confidence: Confidence threshold for SAM3 segmentation
        """
        self.model_name = model_name
        self.categories = categories
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.exclude_categories = exclude_categories or []
        self.sam3_confidence = sam3_confidence

        print(f"\nInitializing Agentic Detector (Qwen3-VL + SAM3)")
        print(f"  Model: {model_name}")
        print(f"  Device: {self.device}")
        print(f"  SAM3 confidence: {sam3_confidence}")
        if self.exclude_categories:
            print(f"  Excluded categories: {', '.join(self.exclude_categories)}")

        # Load Qwen3-VL detector (the "brain")
        self.qwen_detector = Qwen3VLDirectDetector(
            model_name=model_name,
            device=device,
            use_quantization=use_quantization,
            low_memory=low_memory
        )

        # Load SAM3 processor (the "tool")
        if sam3_processor is None:
            print("\nLoading SAM3 segmentation tool...")
            self.sam3_processor = load_sam3_model(
                confidence_threshold=sam3_confidence,
                device=self.device
            )
        else:
            self.sam3_processor = sam3_processor
            print("Using pre-loaded SAM3 processor")

        print("✓ Agentic detector initialized!")

    def detect_infrastructure(
        self,
        image: Union[Image.Image, str],
        use_sam3: bool = True
    ) -> Dict:
        """
        Detect infrastructure issues with optional SAM3 segmentation.

        Agentic workflow:
        1. Qwen3-VL analyzes image → detections with descriptions
        2. For each detection → SAM3 segments based on description
        3. Return detections with bounding boxes + segmentation masks

        Args:
            image: PIL Image or path to image file
            use_sam3: If True, add SAM3 segmentation masks

        Returns:
            dict: {
                'detections': [...],
                'num_detections': int,
                'has_masks': bool
            }
        """
        # Convert to PIL if needed
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')

        # Step 1: Build detection prompt
        if self.categories is None:
            from detector_unified import INFRASTRUCTURE_CATEGORIES
            categories = list(INFRASTRUCTURE_CATEGORIES.keys())
        else:
            categories = self.categories

        prompt = build_detailed_prompt(categories)

        # Step 2: Qwen detects infrastructure issues
        logger.debug("Step 1: Qwen3-VL detection...")
        qwen_result = self.qwen_detector.detect(image, prompt)

        # Handle None or invalid result
        if qwen_result is None or not qwen_result.get('success', False):
            logger.error("Qwen detection failed")
            return {
                'detections': [],
                'num_detections': 0,
                'has_masks': False
            }

        # Step 3: Parse text response to extract detections
        text_response = qwen_result.get('text', '')
        detections = self._parse_detections(text_response, image.size)

        if not use_sam3 or len(detections) == 0:
            # No SAM3 segmentation needed/possible
            return {
                'detections': detections,
                'num_detections': len(detections),
                'has_masks': False
            }

        # Step 4: For each detection, get SAM3 segmentation mask
        logger.debug(f"Step 2: SAM3 segmentation for {len(detections)} detections...")

        enhanced_detections = []
        for det in detections:
            # Use Qwen's bbox (preferred) and description (fallback) to guide SAM3
            try:
                label = det.get('label', 'unknown')
                description = det.get('description', label)
                bbox = det.get('bbox', None)  # Get bounding box from detection

                # Pass bbox to SAM3 (preferred method for accurate segmentation)
                mask = self._segment_with_sam3(
                    image,
                    query=description,  # Fallback if bbox fails
                    bbox=bbox           # Primary method
                )

                # Convert mask to JSON-serializable format
                if mask is not None:
                    # Convert torch.Tensor to numpy if needed
                    if torch.is_tensor(mask):
                        mask = mask.cpu().numpy()
                    # Convert numpy to list for JSON serialization
                    if isinstance(mask, np.ndarray):
                        det['mask'] = mask.tolist()
                    else:
                        det['mask'] = mask
                    det['has_mask'] = True
                else:
                    det['mask'] = None
                    det['has_mask'] = False
            except Exception as e:
                label = det.get('label', 'unknown')
                logger.warning(f"SAM3 segmentation failed for {label}: {e}")
                det['mask'] = None
                det['has_mask'] = False

            enhanced_detections.append(det)

        # Apply post-processing filters to reduce false positives
        try:
            from detection_filters import apply_filters
            image_array = np.array(image)
            filtered_detections = apply_filters(
                enhanced_detections,
                image_array.shape[:2],
                exclude_categories=self.exclude_categories
            )
            logger.debug(f"Filtered {len(enhanced_detections)} -> {len(filtered_detections)} detections")
        except Exception as e:
            logger.warning(f"Filter failed, using unfiltered detections: {e}")
            filtered_detections = enhanced_detections

        return {
            'detections': filtered_detections,
            'num_detections': len(filtered_detections),
            'has_masks': True
        }

    def _parse_detections(
        self,
        response: str,
        image_size: Tuple[int, int],
        confidence_threshold: float = 0.7
    ) -> List[Dict]:
        """
        Parse detection results from Qwen text response.

        Args:
            response: Text response from Qwen model
            image_size: (width, height) of image
            confidence_threshold: Minimum confidence

        Returns:
            list: List of detection dicts with bbox, label, confidence
        """
        from detector_unified import DEFECT_COLORS

        detections = []
        width, height = image_size

        if "no defects detected" in response.lower():
            return detections

        # Pattern: Defect: <type>, Box: [x1, y1, x2, y2], Confidence: <score>
        # Also support pattern without confidence for backward compatibility
        pattern_with_conf = r'Defect:\s*([^,\n]+),\s*Box:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\],\s*Confidence:\s*([\d.]+)'
        pattern_no_conf = r'Defect:\s*([^,\n]+),\s*Box:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'

        matches_with_conf = re.findall(pattern_with_conf, response, re.IGNORECASE)
        matches_no_conf = re.findall(pattern_no_conf, response, re.IGNORECASE)

        # Process matches with confidence first
        for match in matches_with_conf:
            try:
                label = self._normalize_label(match[0].strip().lower())
                confidence = float(match[5])

                # Filter by confidence threshold
                if confidence < confidence_threshold:
                    logger.debug(f"Filtered out {label} with confidence {confidence:.2f} < {confidence_threshold}")
                    continue

                # Convert normalized 0-1000 coords to pixel coords
                x1 = int(float(match[1]) * width / 1000)
                y1 = int(float(match[2]) * height / 1000)
                x2 = int(float(match[3]) * width / 1000)
                y2 = int(float(match[4]) * height / 1000)

                # Clamp to image bounds
                x1 = max(0, min(x1, width))
                y1 = max(0, min(y1, height))
                x2 = max(0, min(x2, width))
                y2 = max(0, min(y2, height))

                # Validate bbox
                if x2 > x1 and y2 > y1:
                    color = DEFECT_COLORS.get(label, (0, 255, 0))

                    # Create detailed SAM3 prompt
                    sam3_description = self._create_sam3_prompt(label)

                    detection = {
                        "label": label,
                        "category": label,
                        "bbox": [x1, y1, x2, y2],
                        "confidence": confidence,  # Use parsed confidence
                        "color": color,
                        "description": sam3_description  # Better prompt for SAM3
                    }

                    detections.append(detection)

            except (ValueError, IndexError) as e:
                logger.warning(f"Failed to parse detection: {match} ({e})")
                continue

        # Process matches without confidence (backward compatibility)
        for match in matches_no_conf:
            try:
                label = self._normalize_label(match[0].strip().lower())

                # Default confidence for backward compatibility
                confidence = 0.8

                # Filter by confidence threshold
                if confidence < confidence_threshold:
                    logger.debug(f"Filtered out {label} with default confidence {confidence:.2f} < {confidence_threshold}")
                    continue

                # Convert normalized 0-1000 coords to pixel coords
                x1 = int(float(match[1]) * width / 1000)
                y1 = int(float(match[2]) * height / 1000)
                x2 = int(float(match[3]) * width / 1000)
                y2 = int(float(match[4]) * height / 1000)

                # Clamp to image bounds
                x1 = max(0, min(x1, width))
                y1 = max(0, min(y1, height))
                x2 = max(0, min(x2, width))
                y2 = max(0, min(y2, height))

                # Validate bbox
                if x2 > x1 and y2 > y1:
                    color = DEFECT_COLORS.get(label, (0, 255, 0))

                    # Create detailed SAM3 prompt
                    sam3_description = self._create_sam3_prompt(label)

                    detection = {
                        "label": label,
                        "category": label,
                        "bbox": [x1, y1, x2, y2],
                        "confidence": confidence,
                        "color": color,
                        "description": sam3_description  # Better prompt for SAM3
                    }

                    detections.append(detection)

            except (ValueError, IndexError) as e:
                logger.warning(f"Failed to parse detection: {match} ({e})")
                continue

        return detections

    def _normalize_label(self, label: str) -> str:
        """Normalize detected label to match predefined categories."""
        from detector_unified import INFRASTRUCTURE_CATEGORIES

        label = label.lower().strip()

        # Direct match
        if label in INFRASTRUCTURE_CATEGORIES:
            return label

        # Keyword matching
        label_mappings = {
            "pothole": "potholes",
            "hole": "potholes",
            "alligator crack": "alligator_cracks",
            "alligator": "alligator_cracks",
            "longitudinal crack": "longitudinal_cracks",
            "longitudinal": "longitudinal_cracks",
            "transverse crack": "transverse_cracks",
            "transverse": "transverse_cracks",
            "vehicle": "abandoned_vehicle",
            "car": "abandoned_vehicle",
            "encampment": "homeless_encampment",
            "tent": "homeless_encampment",
            "homeless": "homeless_person",
            "person": "homeless_person",
            "manhole": "manholes",
            "paint": "damaged_paint",
            "crosswalk": "damaged_crosswalks",
            "trash": "dumped_trash",
            "sign": "street_signs",
            "light": "traffic_lights",
            "tyre": "tyre_marks",
            "tire": "tyre_marks"
        }

        for keyword, category in label_mappings.items():
            if keyword in label:
                return category

        return label

    def _create_sam3_prompt(self, category: str) -> str:
        """
        Create detailed SAM3 prompt from category.

        SAM3 needs descriptive prompts to segment accurately.
        Simple category names like "potholes" are too vague.
        """
        prompts = {
            "potholes": "a hole or depression in the asphalt or concrete road pavement surface",
            "alligator_cracks": "a network of interconnected cracks forming an alligator-skin pattern on the asphalt pavement",
            "longitudinal_cracks": "a crack running parallel along the direction of the road centerline",
            "transverse_cracks": "a crack running across the width of the road perpendicular to the direction of traffic",
            "road_surface_damage": "deteriorated, broken, or crumbling asphalt or concrete road pavement surface with exposed aggregate or material loss",
            "abandoned_vehicle": "a damaged, rusted, broken down, or abandoned motor vehicle with flat tires or missing parts",
            "homeless_encampment": "a tent, tarp, sleeping bag, or makeshift shelter structure used for temporary living",
            "homeless_person": "a person sitting or lying on the ground with belongings, bedding, or camping gear in a public area",
            "manholes": "a round or square metal manhole cover or utility access hatch embedded in the road surface",
            "damaged_paint": "faded, worn, or deteriorated white or yellow road lane markings, arrows, or text painted directly on the asphalt surface",
            "damaged_crosswalks": "faded, worn, or deteriorated white pedestrian crosswalk stripes or zebra crossing markings painted on the road surface",
            "dumped_trash": "trash bags, garbage, litter, debris, furniture, or illegally discarded items on or beside the road",
            "street_signs": "a traffic regulatory sign, warning sign, or street name sign mounted on a metal or wooden pole",
            "traffic_lights": "a traffic signal light fixture with red, yellow, and green colored lights for controlling vehicle traffic",
            "tyre_marks": "dark black rubber tire marks, skid marks, or burnout marks on the asphalt or concrete pavement surface"
        }
        return prompts.get(category, category)

    def _select_best_mask(
        self,
        masks: List[np.ndarray],
        bbox: Optional[List[int]],
        image_size: Tuple[int, int]
    ) -> Optional[np.ndarray]:
        """
        Select the best mask from SAM3's multiple predictions.

        SAM3 returns multiple masks with different granularities. We want the mask that:
        1. Fits within or close to the bounding box
        2. Isn't oversized (doesn't cover the entire image)
        3. Has reasonable coverage of the bbox area

        Args:
            masks: List of binary masks from SAM3
            bbox: Bounding box [x1, y1, x2, y2] in pixels (or None)
            image_size: (width, height) of image

        Returns:
            Best fitting mask or None
        """
        if masks is None or (hasattr(masks, '__len__') and len(masks) == 0):
            return None

        img_width, img_height = image_size
        total_image_pixels = img_width * img_height

        # If no bbox provided, return the smallest mask (most specific)
        if bbox is None:
            # Sort by mask area (ascending) and return smallest
            masks_with_area = [(mask, np.sum(mask)) for mask in masks]
            masks_with_area.sort(key=lambda x: x[1])
            return masks_with_area[0][0] if masks_with_area else None

        x1, y1, x2, y2 = bbox
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        bbox_area = bbox_width * bbox_height

        # Create bbox mask for IoU calculation
        bbox_mask = np.zeros((img_height, img_width), dtype=bool)
        bbox_mask[y1:y2, x1:x2] = True

        best_mask = None
        best_score = -1

        for mask in masks:
            # Convert to numpy array if needed
            if torch.is_tensor(mask):
                mask = mask.cpu().numpy()

            # Ensure mask is 2D
            if mask.ndim > 2:
                mask = mask.squeeze()

            # Ensure mask is boolean
            mask_binary = mask > 0.5 if mask.dtype != bool else mask

            # Calculate mask area
            mask_area = np.sum(mask_binary)

            # Filter out masks that are too large (>50% of image = likely wrong)
            if mask_area > 0.5 * total_image_pixels:
                logger.debug(f"Rejecting oversized mask: {mask_area}/{total_image_pixels} pixels ({mask_area/total_image_pixels*100:.1f}%)")
                continue

            # Filter out masks that are too small (< 1% of bbox)
            if mask_area < 0.01 * bbox_area:
                logger.debug(f"Rejecting undersized mask: {mask_area}/{bbox_area} pixels ({mask_area/bbox_area*100:.1f}% of bbox)")
                continue

            # Calculate IoU with bounding box
            intersection = np.sum(mask_binary & bbox_mask)
            union = np.sum(mask_binary | bbox_mask)
            iou = intersection / union if union > 0 else 0

            # Calculate what % of the mask is inside the bbox
            mask_coverage_in_bbox = intersection / mask_area if mask_area > 0 else 0

            # Score: prefer masks that are mostly inside bbox with good IoU
            # IoU weight: 0.6, coverage weight: 0.4
            score = 0.6 * iou + 0.4 * mask_coverage_in_bbox

            logger.debug(f"Mask area={mask_area}, IoU={iou:.3f}, coverage={mask_coverage_in_bbox:.3f}, score={score:.3f}")

            if score > best_score:
                best_score = score
                best_mask = mask_binary

        if best_mask is not None:
            logger.debug(f"Selected mask with score={best_score:.3f}")
        else:
            logger.warning("No suitable mask found - all masks were oversized, undersized, or poor fit")

        return best_mask

    def _segment_with_sam3(
        self,
        image: Union[Image.Image, str],
        query: str = None,
        bbox: List[int] = None
    ) -> Optional[np.ndarray]:
        """
        Segment an object using SAM3 with geometric prompt (preferred) or text prompt.

        Args:
            image: PIL Image or path
            query: Text description of what to segment (fallback)
            bbox: Bounding box [x1, y1, x2, y2] in pixel coordinates (preferred)

        Returns:
            np.ndarray: Segmentation mask or None if failed
        """
        try:
            # Convert to PIL if needed
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')

            # Get image dimensions for normalization
            img_width, img_height = image.size

            # Step 1: Set the image and get inference state
            inference_state = self.sam3_processor.set_image(image)

            # PRIORITY 1: Use bounding box if available (more accurate)
            if bbox is not None and len(bbox) == 4:
                x1, y1, x2, y2 = bbox

                # Convert [x1, y1, x2, y2] to [center_x, center_y, width, height] normalized
                center_x = ((x1 + x2) / 2) / img_width
                center_y = ((y1 + y2) / 2) / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height

                # SAM3 expects: [center_x, center_y, width, height] normalized to [0, 1]
                box_normalized = [center_x, center_y, width, height]

                # Use geometric prompt (box-based segmentation)
                logger.info(f"SAM3: Calling add_geometric_prompt with box={box_normalized}")
                result = self.sam3_processor.add_geometric_prompt(
                    box=box_normalized,
                    label=True,  # True = positive prompt (include this region)
                    state=inference_state
                )

                # Debug: Check what was returned
                logger.info(f"SAM3: add_geometric_prompt returned type={type(result)}, value={result if not isinstance(result, dict) else 'dict'}")
                logger.info(f"SAM3: inference_state type={type(inference_state)}, keys={inference_state.keys() if isinstance(inference_state, dict) else 'not a dict'}")

                # Try multiple ways to get masks
                masks = None
                if result is not None:
                    # Option 1: Result is the masks directly
                    if isinstance(result, (list, tuple)):
                        masks = result
                        logger.info(f"SAM3: Got masks from result (list/tuple), count={len(masks)}")
                    # Option 2: Result is a dict with masks
                    elif isinstance(result, dict) and 'masks' in result:
                        masks = result['masks']
                        logger.info(f"SAM3: Got masks from result dict, count={len(masks)}")
                    # Option 3: Result has masks attribute
                    elif hasattr(result, 'masks'):
                        masks = result.masks
                        logger.info(f"SAM3: Got masks from result.masks attribute")

                # Option 4: Masks in inference_state
                if masks is None and isinstance(inference_state, dict):
                    masks = inference_state.get('masks', [])
                    logger.info(f"SAM3: Got masks from inference_state, count={len(masks) if masks else 0}")

                if masks is None or (hasattr(masks, '__len__') and len(masks) == 0):
                    logger.error(f"SAM3: No masks found! result={result}, state_keys={inference_state.keys() if isinstance(inference_state, dict) else 'N/A'}")

            # FALLBACK: Use text prompt
            elif query is not None:
                output = self.sam3_processor.set_text_prompt(
                    state=inference_state,
                    prompt=query
                )
                # Extract masks from output
                if isinstance(output, dict) and 'masks' in output:
                    masks = output['masks']
                else:
                    logger.warning(f"Unexpected SAM3 output format: {type(output)}")
                    return None
            else:
                logger.error("SAM3 called without bbox or query")
                return None

            # Extract and select best mask
            if masks is not None and len(masks) > 0:
                logger.debug(f"SAM3 returned {len(masks)} masks for bbox={bbox}")

                # SAM3 returns multiple masks with different granularities
                # We need to select the mask that best fits the bounding box
                selected_mask = self._select_best_mask(masks, bbox, image.size)

                if selected_mask is not None:
                    logger.debug(f"Selected mask with area: {np.sum(selected_mask) if isinstance(selected_mask, np.ndarray) else 'unknown'}")
                    return selected_mask
                else:
                    logger.warning(f"No suitable mask found for bbox={bbox}")
                    return None
            else:
                logger.warning(f"SAM3 returned 0 masks for bbox={bbox}, query={query}")
                return None

        except Exception as e:
            logger.error(f"SAM3 segmentation error: {e}", exc_info=True)
            return None

    def detect_infrastructure_batch(
        self,
        images: List[Union[Image.Image, str]],
        use_sam3: bool = True
    ) -> List[Dict]:
        """
        Batch detection with agentic SAM3 segmentation.

        Args:
            images: List of PIL Images or image paths
            use_sam3: If True, add SAM3 segmentation

        Returns:
            List of detection dictionaries
        """
        results = []
        for image in images:
            result = self.detect_infrastructure(image, use_sam3=use_sam3)
            results.append(result)
        return results
