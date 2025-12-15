"""
Tool Definitions and Executor for Infrastructure Detection Agent

Implements the 4 tools from SAM3 Agent pattern:
1. segment_phrase - Call SAM3 to segment objects
2. examine_each_mask - Render and validate each mask
3. select_masks_and_return - Select final masks
4. report_no_mask - Report no issues found

Based on: https://github.com/facebookresearch/sam3/blob/main/sam3/agent/agent_core.py
"""
import io
import base64
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result from tool execution."""
    success: bool
    tool_name: str
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    image: Optional[Image.Image] = None  # Rendered result image
    should_exit: bool = False  # True for select_masks_and_return, report_no_mask


@dataclass
class MaskData:
    """Stores mask data from SAM3."""
    mask_id: int
    mask: np.ndarray
    score: float
    bbox: List[int]  # [x1, y1, x2, y2]
    rle: Optional[str] = None
    category: str = "unknown"  # Category/prompt used to find this mask
    text_prompt: str = ""  # Original text prompt used


class ToolExecutor:
    """
    Executes tools for the infrastructure detection agent.

    Manages:
    - SAM3 calls for segmentation
    - Mask storage and rendering
    - Detection result compilation
    """

    # Color palette for mask visualization (RGB)
    COLORS = [
        (255, 0, 0),      # Red
        (0, 255, 0),      # Green
        (0, 0, 255),      # Blue
        (255, 255, 0),    # Yellow
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Cyan
        (255, 128, 0),    # Orange
        (128, 0, 255),    # Purple
        (0, 255, 128),    # Spring Green
        (255, 0, 128),    # Rose
    ]

    def __init__(self, sam3_processor, image: Image.Image):
        """
        Initialize tool executor.

        Args:
            sam3_processor: Loaded SAM3 processor
            image: PIL Image to process
        """
        self.sam3_processor = sam3_processor
        self.original_image = image.convert('RGB')
        self.image_size = image.size  # (width, height)

        # State tracking
        self.masks: List[MaskData] = []
        self.used_prompts: set = set()
        self.current_mask_image: Optional[Image.Image] = None
        self.inference_state = None

        # Initialize SAM3 with image
        self._init_sam3()

    def _init_sam3(self):
        """Initialize SAM3 with the image."""
        try:
            self.inference_state = self.sam3_processor.set_image(self.original_image)
            logger.debug("SAM3 initialized with image")
        except Exception as e:
            logger.error(f"Failed to initialize SAM3: {e}")
            raise

    def get_current_masks(self) -> List[MaskData]:
        """Return all accumulated masks."""
        return self.masks

    def execute(self, tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
        """
        Execute a tool by name.

        Args:
            tool_name: Name of tool to execute
            parameters: Tool parameters

        Returns:
            ToolResult with execution outcome
        """
        tool_map = {
            "segment_phrase": self._segment_phrase,
            "examine_each_mask": self._examine_each_mask,
            "select_masks_and_return": self._select_masks_and_return,
            "report_no_mask": self._report_no_mask,
        }

        if tool_name not in tool_map:
            return ToolResult(
                success=False,
                tool_name=tool_name,
                message=f"Unknown tool: {tool_name}",
            )

        try:
            return tool_map[tool_name](parameters)
        except Exception as e:
            logger.error(f"Tool execution error ({tool_name}): {e}", exc_info=True)
            return ToolResult(
                success=False,
                tool_name=tool_name,
                message=f"Tool execution failed: {str(e)}",
            )

    def _segment_phrase(self, params: Dict[str, Any]) -> ToolResult:
        """
        Execute segment_phrase tool - call SAM3 to segment objects.

        Args:
            params: {"text_prompt": "pothole"}

        Returns:
            ToolResult with masks and rendered image
        """
        text_prompt = params.get("text_prompt", "").strip()

        if not text_prompt:
            return ToolResult(
                success=False,
                tool_name="segment_phrase",
                message="Error: text_prompt is required",
            )

        # Check for duplicate prompts
        if text_prompt.lower() in self.used_prompts:
            return ToolResult(
                success=False,
                tool_name="segment_phrase",
                message=f"Error: text_prompt '{text_prompt}' has already been used. Try a different phrase.",
            )

        self.used_prompts.add(text_prompt.lower())

        try:
            # Call SAM3 with text prompt
            output = self.sam3_processor.set_text_prompt(
                state=self.inference_state,
                prompt=text_prompt
            )

            # DEBUG: Log what SAM3 returns
            logger.debug(f"SAM3 output type: {type(output)}")
            if output is not None:
                if isinstance(output, dict):
                    logger.debug(f"SAM3 output keys: {output.keys()}")
                    if 'masks' in output:
                        logger.debug(f"SAM3 masks count: {len(output['masks'])}")
                        if 'scores' in output:
                            logger.debug(f"SAM3 scores: {output['scores'][:5] if len(output['scores']) > 5 else output['scores']}")

            # Extract masks from output
            masks = self._extract_masks(output)

            if not masks:
                return ToolResult(
                    success=True,
                    tool_name="segment_phrase",
                    message=f"SAM3 found 0 masks for '{text_prompt}'. Try a different phrase.",
                    data={"num_masks": 0, "prompt": text_prompt},
                )

            # Add masks to storage with IDs and category
            # BUT skip masks that heavily overlap with existing masks (avoid duplicates)
            start_id = len(self.masks) + 1
            category = self._prompt_to_category(text_prompt)
            added_count = 0
            for i, mask_data in enumerate(masks):
                # Check if this mask overlaps too much with existing masks
                if self._is_duplicate_mask(mask_data):
                    logger.debug(f"Skipping duplicate mask for '{text_prompt}' - overlaps with existing detection")
                    continue

                mask_data.mask_id = start_id + added_count
                mask_data.category = category
                mask_data.text_prompt = text_prompt
                self.masks.append(mask_data)
                added_count += 1

            # Render masks on image
            rendered_image = self._render_masks(self.masks)
            self.current_mask_image = rendered_image

            skipped = len(masks) - added_count
            skip_msg = f" (skipped {skipped} duplicates)" if skipped > 0 else ""

            return ToolResult(
                success=True,
                tool_name="segment_phrase",
                message=f"SAM3 found {added_count} new mask(s) for '{text_prompt}'{skip_msg}. Total masks: {len(self.masks)}",
                data={
                    "num_masks": len(masks),
                    "total_masks": len(self.masks),
                    "prompt": text_prompt,
                    "mask_ids": [m.mask_id for m in masks],
                },
                image=rendered_image,
            )

        except Exception as e:
            logger.error(f"SAM3 segmentation failed: {e}", exc_info=True)
            return ToolResult(
                success=False,
                tool_name="segment_phrase",
                message=f"SAM3 error: {str(e)}",
            )

    def segment_from_boxes(self, detections: List[Dict]) -> List:
        """
        SMART SEGMENTATION: Use SAM3 to segment specific bounding boxes.

        Two-stage approach:
        1. Try SAM3 text prompt with the label (e.g., "pothole")
        2. Crop result to bounding box from Qwen detection

        This combines Qwen's semantic understanding with SAM3's segmentation.

        Args:
            detections: List of dicts from Qwen3 with:
                - 'label': object type
                - 'bbox': [x1, y1, x2, y2]
                - 'confidence': float

        Returns:
            List of MaskData objects with precise segmentation masks
        """
        if not detections:
            return []

        print(f"  Segmenting {len(detections)} detected objects with SAM3...")

        masks = []
        for i, det in enumerate(detections):
            label = det['label']
            bbox = det['bbox']
            confidence = det.get('confidence', 0.8)

            print(f"    [{i+1}/{len(detections)}] {label} at {bbox}...", end=" ")

            try:
                mask_np = None

                # Method 1: Use SAM3 TEXT PROMPT (BEST quality - semantic understanding)
                # Try multiple label variations for better SAM3 text matching
                label_variants = self._get_sam3_friendly_labels(label)

                for variant in label_variants:
                    mask_np = self._segment_with_text_and_bbox(variant, bbox)
                    if mask_np is not None and mask_np.any():
                        print(f"(text:'{variant}') ", end="")
                        break

                # Method 2: Use SAM3 box/point prompts if text failed
                if mask_np is None or not mask_np.any():
                    mask_np = self._segment_box_with_sam3(bbox)

                if mask_np is not None and mask_np.any():
                    mask_data = MaskData(
                        mask_id=len(self.masks) + 1,
                        mask=mask_np,
                        bbox=bbox,
                        category=label,
                        score=confidence,
                        text_prompt=label,
                    )
                    self.masks.append(mask_data)
                    masks.append(mask_data)
                    print("✓")
                else:
                    print("✗ (no mask)")

            except Exception as e:
                print(f"✗ ({e})")
                logger.error(f"Box segmentation failed for {label}: {e}")

        print(f"  Total: {len(masks)} masks generated")
        return masks

    def _get_sam3_friendly_labels(self, label: str) -> List[str]:
        """
        Get SAM3-friendly label variants for better text prompt matching.

        SAM3 works better with common object names. This maps our detection
        categories to terms SAM3 understands well.

        Args:
            label: Original detection label

        Returns:
            List of label variants to try (in order of preference)
        """
        label_lower = label.lower().strip()

        # Mapping from detection labels to SAM3-friendly terms
        label_map = {
            # Road damage
            'pothole': ['pothole', 'hole in road', 'road damage'],
            'crack': ['crack', 'road crack', 'pavement crack'],

            # Homeless/encampments
            'homeless person': ['person', 'human', 'people'],
            'homeless': ['person', 'human', 'tent'],
            'tent': ['tent', 'camping tent', 'shelter'],
            'encampment': ['tent', 'tarp', 'shelter'],
            'sleeping bag': ['sleeping bag', 'blanket', 'bedding'],
            'belongings': ['bags', 'luggage', 'belongings'],

            # Infrastructure
            'manhole': ['manhole cover', 'manhole', 'metal cover', 'drain cover'],
            'graffiti': ['graffiti', 'spray paint', 'wall art', 'paint'],
            'trash': ['trash', 'garbage', 'litter', 'debris'],
            'illegal dumping': ['furniture', 'mattress', 'debris pile'],

            # Vehicles
            'abandoned vehicle': ['car', 'vehicle', 'automobile'],
            'car': ['car', 'vehicle', 'automobile'],
            'vehicle': ['car', 'vehicle', 'automobile'],

            # Other
            'damaged sign': ['sign', 'street sign', 'road sign'],
            'blocked sidewalk': ['obstruction', 'barrier', 'blockage'],
        }

        # Get variants or use original label
        if label_lower in label_map:
            return label_map[label_lower]

        # Check partial matches
        for key, variants in label_map.items():
            if key in label_lower or label_lower in key:
                return variants

        # Return original label as fallback
        return [label]

    def _segment_with_text_and_bbox(self, label: str, bbox: List[int]) -> np.ndarray:
        """
        Use SAM3 text prompt to segment, then select the best mask overlapping with bbox.

        This is the BEST quality method because:
        1. SAM3 uses semantic understanding of the label
        2. We pick the mask that best fits the bbox from Qwen
        3. Result is a proper segmentation, not a rectangle

        Args:
            label: Object type (SAM3-friendly term)
            bbox: [x1, y1, x2, y2] bounding box from Qwen

        Returns:
            numpy array mask or None if failed
        """
        try:
            x1, y1, x2, y2 = bbox
            h, w = self.original_image.size[1], self.original_image.size[0]

            # Call SAM3 text prompt
            output = self.sam3_processor.set_text_prompt(
                state=self.inference_state,
                prompt=label
            )

            if not output:
                return None

            # Get all masks from output
            raw_masks = []
            scores = []

            if isinstance(output, dict):
                raw_masks = output.get('masks', [])
                scores = output.get('scores', [])
            elif hasattr(output, 'masks'):
                raw_masks = output.masks
                scores = getattr(output, 'scores', [])

            if not raw_masks:
                return None

            # Find the mask with BEST overlap with bbox
            best_mask = None
            best_score = 0

            for i, mask in enumerate(raw_masks):
                # Convert to numpy
                if hasattr(mask, 'cpu'):
                    mask_np = mask.cpu().numpy()
                else:
                    mask_np = np.array(mask)

                if mask_np.ndim > 2:
                    mask_np = mask_np.squeeze()

                # Calculate overlap with bbox
                bbox_mask = (mask_np[y1:y2, x1:x2] > 0.5)
                overlap_pixels = np.sum(bbox_mask)

                # Calculate IoU-like score
                total_mask_pixels = np.sum(mask_np > 0.5)
                bbox_area = (y2 - y1) * (x2 - x1)

                if total_mask_pixels > 0 and overlap_pixels > 0:
                    # Score: overlap / max(mask_size, bbox_size)
                    # This favors masks that fit well within the bbox
                    score = overlap_pixels / max(total_mask_pixels, bbox_area)

                    # Bonus if SAM3 gave high confidence
                    if scores and i < len(scores):
                        sam_score = scores[i].item() if hasattr(scores[i], 'item') else scores[i]
                        score *= (1 + sam_score)

                    if score > best_score:
                        best_score = score
                        best_mask = mask_np

            if best_mask is not None and best_score > 0.01:  # Minimum threshold
                # Keep full mask but ensure it overlaps with bbox
                # Don't crop to bbox - keep SAM3's natural segmentation
                final_mask = (best_mask > 0.5).astype(np.uint8) * 255
                return final_mask

            return None

        except Exception as e:
            logger.debug(f"Text+bbox segmentation failed for '{label}': {e}")
            return None

    def _segment_with_text_and_crop(self, label: str, bbox: List[int]) -> np.ndarray:
        """
        Use SAM3 text prompt to find the object, then crop to bbox.

        This is more accurate because:
        1. SAM3 uses semantic understanding for the label
        2. We only keep the mask portion inside the bbox from Qwen

        Args:
            label: Object type (e.g., "pothole", "manhole")
            bbox: [x1, y1, x2, y2] bounding box

        Returns:
            numpy array mask cropped to bbox
        """
        try:
            x1, y1, x2, y2 = bbox
            h, w = self.original_image.size[1], self.original_image.size[0]

            # Use SAM3's text prompt
            output = self.sam3_processor.set_text_prompt(
                state=self.inference_state,
                prompt=label
            )

            if not output:
                return None

            # Get masks from output
            raw_masks = []
            if isinstance(output, dict):
                raw_masks = output.get('masks', [])
            elif hasattr(output, 'masks'):
                raw_masks = output.masks

            if not raw_masks:
                return None

            # Find the mask that best overlaps with bbox
            best_mask = None
            best_overlap = 0

            for mask in raw_masks:
                # Convert to numpy
                if hasattr(mask, 'cpu'):
                    mask_np = mask.cpu().numpy()
                else:
                    mask_np = np.array(mask)

                if mask_np.ndim > 2:
                    mask_np = mask_np.squeeze()

                # Calculate overlap with bbox
                bbox_region = mask_np[y1:y2, x1:x2]
                overlap = np.sum(bbox_region > 0.5)

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_mask = mask_np

            if best_mask is not None and best_overlap > 0:
                # Create final mask cropped to bbox
                final_mask = np.zeros((h, w), dtype=np.uint8)
                final_mask[y1:y2, x1:x2] = (best_mask[y1:y2, x1:x2] > 0.5).astype(np.uint8) * 255
                return final_mask

            return None

        except Exception as e:
            logger.debug(f"Text+crop segmentation failed: {e}")
            return None

    def _segment_box_with_sam3(self, bbox: List[int]) -> np.ndarray:
        """
        Use SAM3 to segment a specific bounding box region.

        Strategy (in order of quality):
        1. SAM3 box prompt - best for box-guided segmentation
        2. SAM3 multi-point prompt - uses center + corner points
        3. Rectangular fallback - last resort

        Args:
            bbox: [x1, y1, x2, y2] bounding box coordinates

        Returns:
            numpy array mask or None if failed
        """
        try:
            x1, y1, x2, y2 = bbox
            h, w = self.original_image.size[1], self.original_image.size[0]

            # Method 1: Try SAM3 box prompt (BEST for box-guided segmentation)
            try:
                # SAM3 box prompt expects [[x1, y1, x2, y2]]
                box_input = [[x1, y1, x2, y2]]

                if hasattr(self.sam3_processor, 'set_box_prompt'):
                    output = self.sam3_processor.set_box_prompt(
                        state=self.inference_state,
                        box=box_input
                    )
                elif hasattr(self.sam3_processor, 'set_boxes_prompt'):
                    output = self.sam3_processor.set_boxes_prompt(
                        state=self.inference_state,
                        boxes=box_input
                    )
                else:
                    output = None

                if output:
                    mask = self._extract_best_mask_from_output(output, bbox)
                    if mask is not None:
                        print("(box prompt) ", end="")
                        return mask

            except Exception as e:
                logger.debug(f"Box prompt failed: {e}")

            # Method 2: Use multiple point prompts (center + corners for better coverage)
            try:
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # Use 5 points: center + 4 corners (inset slightly)
                inset_x = int((x2 - x1) * 0.2)
                inset_y = int((y2 - y1) * 0.2)

                points = [
                    [center_x, center_y],  # Center
                    [x1 + inset_x, y1 + inset_y],  # Top-left
                    [x2 - inset_x, y1 + inset_y],  # Top-right
                    [x1 + inset_x, y2 - inset_y],  # Bottom-left
                    [x2 - inset_x, y2 - inset_y],  # Bottom-right
                ]
                labels = [1, 1, 1, 1, 1]  # All foreground

                output = self.sam3_processor.set_point_prompt(
                    state=self.inference_state,
                    point=points,
                    labels=labels
                )

                if output:
                    mask = self._extract_best_mask_from_output(output, bbox)
                    if mask is not None:
                        print("(point prompt) ", end="")
                        return mask

            except Exception as e:
                logger.debug(f"Multi-point prompt failed: {e}")

            # Method 3: Single center point (simpler fallback)
            try:
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                output = self.sam3_processor.set_point_prompt(
                    state=self.inference_state,
                    point=[[center_x, center_y]],
                    labels=[1]
                )

                if output:
                    mask = self._extract_best_mask_from_output(output, bbox)
                    if mask is not None:
                        print("(center point) ", end="")
                        return mask

            except Exception as e:
                logger.debug(f"Center point prompt failed: {e}")

            # Method 4: Rectangular mask (LAST RESORT)
            logger.debug(f"Using rectangular bbox mask as fallback")
            print("(rect fallback) ", end="")
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[y1:y2, x1:x2] = 255
            return mask

        except Exception as e:
            logger.error(f"SAM3 box segmentation error: {e}")
            # Fallback: return rectangular mask
            h, w = self.original_image.size[1], self.original_image.size[0]
            mask = np.zeros((h, w), dtype=np.uint8)
            x1, y1, x2, y2 = bbox
            mask[y1:y2, x1:x2] = 255
            return mask

    def _extract_best_mask_from_output(self, output, bbox: List[int]) -> np.ndarray:
        """
        Extract the best mask from SAM3 output that overlaps with bbox.

        Args:
            output: SAM3 output (dict or object)
            bbox: [x1, y1, x2, y2] target bounding box

        Returns:
            numpy array mask cropped to bbox, or None
        """
        x1, y1, x2, y2 = bbox
        h, w = self.original_image.size[1], self.original_image.size[0]

        try:
            # Extract masks from output
            if isinstance(output, dict):
                masks = output.get('masks', [])
                if not masks and 'mask' in output:
                    masks = [output['mask']]
            elif hasattr(output, 'masks'):
                masks = output.masks
            elif isinstance(output, list):
                masks = output
            else:
                masks = [output]

            if not masks:
                return None

            # Find mask with best overlap with bbox
            best_mask = None
            best_score = 0

            for mask in masks:
                # Convert to numpy
                if hasattr(mask, 'cpu'):
                    mask_np = mask.cpu().numpy()
                elif hasattr(mask, 'numpy'):
                    mask_np = mask.numpy()
                else:
                    mask_np = np.array(mask)

                if mask_np.ndim > 2:
                    mask_np = mask_np.squeeze()

                # Ensure correct shape
                if mask_np.shape[0] != h or mask_np.shape[1] != w:
                    continue

                # Calculate overlap score with bbox
                bbox_region = (mask_np[y1:y2, x1:x2] > 0.5).sum()
                total_mask = (mask_np > 0.5).sum()

                # Prefer masks that are mostly inside the bbox
                if total_mask > 0:
                    overlap_ratio = bbox_region / total_mask
                    score = bbox_region * overlap_ratio  # Favor high overlap

                    if score > best_score:
                        best_score = score
                        best_mask = mask_np

            if best_mask is not None and best_score > 0:
                # Create final mask cropped to bbox
                final_mask = np.zeros((h, w), dtype=np.uint8)
                final_mask[y1:y2, x1:x2] = (best_mask[y1:y2, x1:x2] > 0.5).astype(np.uint8) * 255

                # Check if mask has reasonable coverage
                mask_pixels = (final_mask > 0).sum()
                bbox_pixels = (y2 - y1) * (x2 - x1)
                coverage = mask_pixels / bbox_pixels if bbox_pixels > 0 else 0

                # If coverage is too low, might be bad segmentation
                if coverage < 0.05:
                    logger.debug(f"Mask coverage too low: {coverage:.2%}")
                    return None

                return final_mask

            return None

        except Exception as e:
            logger.debug(f"Failed to extract mask from output: {e}")
            return None

    def _examine_each_mask(self, params: Dict[str, Any]) -> ToolResult:
        """
        Execute examine_each_mask tool - render each mask for LLM validation.

        This is the KEY validation step from the official SAM3 agent:
        - Shows each mask overlaid on original image
        - LLM must Accept or Reject each mask
        - Shadows and false positives get rejected

        Returns:
            ToolResult with individual mask images for validation
        """
        if not self.masks:
            return ToolResult(
                success=False,
                tool_name="examine_each_mask",
                message="No masks to examine. Call segment_phrase first.",
            )

        # Create individual mask views for validation
        individual_images = []
        mask_info = []

        for mask_data in self.masks:
            # Render this single mask on original image
            img = self._render_single_mask(mask_data)
            individual_images.append((mask_data.mask_id, img))

            mask_info.append({
                "mask_id": mask_data.mask_id,
                "category": mask_data.category,
                "confidence": mask_data.score,
                "bbox": mask_data.bbox,
            })

        # Create combined grid image
        grid_image = self._create_mask_grid(individual_images)

        # Store masks pending validation
        self.masks_pending_validation = list(self.masks)

        validation_prompt = f"""
Examine each mask carefully. For EACH mask, decide:
- **Accept**: Mask correctly shows the object (real pothole, real crack, etc.)
- **Reject**: Mask shows something wrong (shadow, reflection, false positive)

Masks to validate:
{self._format_mask_list_for_validation(mask_info)}

Respond with your verdict for each mask:
<verdict>
mask_1: Accept/Reject - reason
mask_2: Accept/Reject - reason
...
</verdict>

Then call select_masks_and_return with ONLY the accepted mask IDs.
"""

        return ToolResult(
            success=True,
            tool_name="examine_each_mask",
            message=validation_prompt,
            data={
                "num_masks": len(self.masks),
                "mask_ids": [m.mask_id for m in self.masks],
                "masks_info": mask_info,
                "requires_validation": True,
            },
            image=grid_image,
        )

    def _format_mask_list_for_validation(self, mask_info: List[Dict]) -> str:
        """Format mask info for LLM validation prompt."""
        lines = []
        for info in mask_info:
            lines.append(
                f"- Mask {info['mask_id']}: {info['category']} "
                f"(confidence: {info['confidence']:.2f}, bbox: {info['bbox']})"
            )
        return "\n".join(lines)

    def _select_masks_and_return(self, params: Dict[str, Any]) -> ToolResult:
        """
        Execute select_masks_and_return tool - finalize detection.

        Args:
            params: {
                "final_answer_masks": [1, 3],
                "detections": [{"mask_id": 1, "category": "pothole", ...}]
            }

        Returns:
            ToolResult with final detections (should_exit=True)
        """
        mask_ids = params.get("final_answer_masks", [])
        detections = params.get("detections", [])

        if not mask_ids:
            return ToolResult(
                success=False,
                tool_name="select_masks_and_return",
                message="Error: final_answer_masks is required",
            )

        # Validate mask IDs
        valid_ids = {m.mask_id for m in self.masks}
        invalid_ids = [mid for mid in mask_ids if mid not in valid_ids]
        if invalid_ids:
            return ToolResult(
                success=False,
                tool_name="select_masks_and_return",
                message=f"Error: Invalid mask IDs: {invalid_ids}. Valid IDs: {sorted(valid_ids)}",
            )

        # Build final detections with mask data
        final_detections = []

        # Create a lookup from detections list
        det_lookup = {d.get("mask_id"): d for d in detections if d.get("mask_id")}

        for mask_id in mask_ids:
            mask_data = next((m for m in self.masks if m.mask_id == mask_id), None)
            if mask_data:
                # Get detection info from LLM if provided, otherwise use stored category
                det_info = det_lookup.get(mask_id, {})

                # Use stored category from mask, fallback to LLM provided or "unknown"
                category = mask_data.category if mask_data.category != "unknown" else det_info.get("category", "unknown")

                # Determine severity based on category
                severity = det_info.get("severity") or self._category_to_severity(category)

                final_detections.append({
                    "mask_id": mask_id,
                    "category": category,
                    "severity": severity,
                    "description": det_info.get("description", f"{category} detected"),
                    "confidence": mask_data.score,
                    "bbox": mask_data.bbox,
                    "mask": mask_data.mask,
                })

        # Render final selected masks
        selected_masks = [m for m in self.masks if m.mask_id in mask_ids]
        final_image = self._render_masks(selected_masks, show_labels=True)

        return ToolResult(
            success=True,
            tool_name="select_masks_and_return",
            message=f"Selected {len(mask_ids)} mask(s) as final detection.",
            data={
                "detections": final_detections,
                "num_detections": len(final_detections),
            },
            image=final_image,
            should_exit=True,
        )

    def _report_no_mask(self, params: Dict[str, Any]) -> ToolResult:
        """
        Execute report_no_mask tool - report no issues found.

        Args:
            params: {"reason": "explanation"}

        Returns:
            ToolResult with empty detections (should_exit=True)
        """
        reason = params.get("reason", "No infrastructure issues detected")

        return ToolResult(
            success=True,
            tool_name="report_no_mask",
            message=reason,
            data={
                "detections": [],
                "num_detections": 0,
                "reason": reason,
            },
            image=self.original_image,
            should_exit=True,
        )

    def _extract_masks(self, sam3_output) -> List[MaskData]:
        """Extract mask data from SAM3 output."""
        masks = []

        try:
            # Handle different output formats
            if isinstance(sam3_output, dict):
                raw_masks = sam3_output.get('masks', [])
                scores = sam3_output.get('scores', [])
                boxes = sam3_output.get('boxes', [])
                print(f"      [SAM3] Found {len(raw_masks)} raw masks")
                # Handle tensor/list scores safely
                if scores is not None and (isinstance(scores, list) and len(scores) > 0) or (hasattr(scores, 'numel') and scores.numel() > 0):
                    try:
                        score_vals = [s.item() if hasattr(s, 'item') else s for s in scores[:5]]
                        print(f"      [SAM3] Score samples: {score_vals}")
                    except:
                        pass
            elif hasattr(sam3_output, 'masks'):
                raw_masks = sam3_output.masks
                scores = getattr(sam3_output, 'scores', [])
                boxes = getattr(sam3_output, 'boxes', [])
                print(f"      [SAM3] Found {len(raw_masks)} raw masks (object)")
            else:
                print(f"      [SAM3] Unexpected output type: {type(sam3_output)}")
                logger.warning(f"Unexpected SAM3 output type: {type(sam3_output)}")
                return masks

            # Process each mask
            for i, mask in enumerate(raw_masks):
                # Convert mask to numpy
                if hasattr(mask, 'cpu'):
                    mask_np = mask.cpu().numpy()
                elif isinstance(mask, np.ndarray):
                    mask_np = mask
                else:
                    mask_np = np.array(mask)

                # Ensure 2D
                if mask_np.ndim > 2:
                    mask_np = mask_np.squeeze()

                # Get score
                score = scores[i] if i < len(scores) else 0.5
                if hasattr(score, 'item'):
                    score = score.item()

                # Get bbox
                if i < len(boxes):
                    box = boxes[i]
                    if hasattr(box, 'tolist'):
                        box = box.tolist()
                    bbox = [int(b) for b in box]
                else:
                    # Compute bbox from mask
                    bbox = self._mask_to_bbox(mask_np)

                masks.append(MaskData(
                    mask_id=0,  # Will be assigned later
                    mask=mask_np,
                    score=float(score),
                    bbox=bbox,
                ))

        except Exception as e:
            logger.error(f"Failed to extract masks: {e}", exc_info=True)

        return masks

    def _mask_to_bbox(self, mask: np.ndarray) -> List[int]:
        """Convert binary mask to bounding box [x1, y1, x2, y2]."""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        if not rows.any() or not cols.any():
            return [0, 0, 0, 0]

        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]

        return [int(x1), int(y1), int(x2), int(y2)]

    def _is_duplicate_mask(self, new_mask: MaskData, iou_threshold: float = 0.5) -> bool:
        """
        Check if a new mask significantly overlaps with existing masks.

        This prevents the same object (e.g., manhole) from being detected
        multiple times with different labels (e.g., as both 'pothole' and 'manhole').

        Args:
            new_mask: The new mask to check
            iou_threshold: IoU threshold above which masks are considered duplicates

        Returns:
            True if this mask is a duplicate, False otherwise
        """
        if not self.masks:
            return False

        new_mask_np = new_mask.mask
        if new_mask_np is None:
            return False

        # Ensure mask is boolean
        if new_mask_np.dtype != bool:
            new_mask_np = new_mask_np > 0

        new_area = np.sum(new_mask_np)
        if new_area == 0:
            return True  # Empty mask is duplicate

        for existing in self.masks:
            existing_mask_np = existing.mask
            if existing_mask_np is None:
                continue

            # Ensure same shape
            if existing_mask_np.shape != new_mask_np.shape:
                continue

            # Ensure boolean
            if existing_mask_np.dtype != bool:
                existing_mask_np = existing_mask_np > 0

            # Calculate IoU (Intersection over Union)
            intersection = np.sum(new_mask_np & existing_mask_np)
            existing_area = np.sum(existing_mask_np)
            union = new_area + existing_area - intersection

            if union > 0:
                iou = intersection / union
                if iou >= iou_threshold:
                    logger.debug(
                        f"Duplicate detected: IoU={iou:.2f} between "
                        f"'{new_mask.text_prompt}' and existing '{existing.category}'"
                    )
                    return True

        return False

    def _render_masks(
        self,
        masks: List[MaskData],
        show_labels: bool = True,
        alpha: float = 0.4
    ) -> Image.Image:
        """Render masks overlaid on original image."""
        # Create copy of original
        result = self.original_image.copy()
        overlay = Image.new('RGBA', result.size, (0, 0, 0, 0))

        for i, mask_data in enumerate(masks):
            color = self.COLORS[i % len(self.COLORS)]

            # Create colored mask
            mask_colored = Image.new('RGBA', result.size, (0, 0, 0, 0))
            mask_np = mask_data.mask

            # Resize mask if needed
            if mask_np.shape != (result.size[1], result.size[0]):
                from PIL import Image as PILImage
                mask_pil = PILImage.fromarray((mask_np * 255).astype(np.uint8))
                mask_pil = mask_pil.resize(result.size, PILImage.NEAREST)
                mask_np = np.array(mask_pil) > 127

            # Apply color to mask
            mask_array = np.zeros((*mask_np.shape, 4), dtype=np.uint8)
            mask_array[mask_np > 0] = (*color, int(255 * alpha))
            mask_colored = Image.fromarray(mask_array, 'RGBA')

            overlay = Image.alpha_composite(overlay, mask_colored)

        # Composite overlay onto result
        result = result.convert('RGBA')
        result = Image.alpha_composite(result, overlay)

        # Add labels if requested
        if show_labels:
            result = self._add_mask_labels(result, masks)

        return result.convert('RGB')

    def _render_single_mask(self, mask_data: MaskData) -> Image.Image:
        """Render a single mask with label."""
        return self._render_masks([mask_data], show_labels=True)

    def _create_mask_grid(
        self,
        images: List[Tuple[int, Image.Image]],
        cols: int = 3
    ) -> Image.Image:
        """Create a grid of mask images."""
        if not images:
            return self.original_image

        # Calculate grid size
        n = len(images)
        rows = (n + cols - 1) // cols

        # Get thumbnail size
        thumb_width = min(400, self.original_image.size[0])
        thumb_height = int(thumb_width * self.original_image.size[1] / self.original_image.size[0])

        # Create grid
        grid_width = thumb_width * cols
        grid_height = thumb_height * rows
        grid = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))

        for i, (mask_id, img) in enumerate(images):
            row = i // cols
            col = i % cols

            # Resize image
            thumb = img.resize((thumb_width, thumb_height), Image.LANCZOS)

            # Paste into grid
            x = col * thumb_width
            y = row * thumb_height
            grid.paste(thumb, (x, y))

        return grid

    def _add_mask_labels(
        self,
        image: Image.Image,
        masks: List[MaskData]
    ) -> Image.Image:
        """Add numbered labels to mask image."""
        from PIL import ImageDraw, ImageFont

        draw = ImageDraw.Draw(image)

        # Try to load font, fall back to default - SMALLER SIZE
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
        except:
            font = ImageFont.load_default()

        for mask_data in masks:
            bbox = mask_data.bbox
            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = bbox

                # Short label - just category name
                category_display = mask_data.category.replace("_", " ").title()
                label = category_display  # Removed ID prefix for cleaner look

                # Draw label background
                text_bbox = draw.textbbox((x1, y1 - 15), label, font=font)
                draw.rectangle(text_bbox, fill=(0, 0, 0, 180))

                # Draw label text
                draw.text((x1, y1 - 15), label, fill=(255, 255, 255), font=font)

                # Draw bbox outline
                color = self.COLORS[(mask_data.mask_id - 1) % len(self.COLORS)]
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        return image

    def _category_to_severity(self, category: str) -> str:
        """Map category to severity level."""
        critical = ["potholes", "alligator_cracks"]
        medium = ["abandoned_vehicle", "homeless_encampment", "homeless_person", "road_surface_damage"]

        if category in critical:
            return "critical"
        elif category in medium:
            return "medium"
        else:
            return "low"

    def _prompt_to_category(self, text_prompt: str) -> str:
        """
        Convert a text prompt to a category name.

        Maps prompts like "pothole", "hole in road" -> "potholes"
        """
        prompt_lower = text_prompt.lower().strip()

        # Direct mappings
        category_mappings = {
            # Potholes
            "pothole": "potholes",
            "potholes": "potholes",
            "hole": "potholes",
            "hole in road": "potholes",
            "road hole": "potholes",

            # Cracks
            "alligator crack": "alligator_cracks",
            "alligator cracks": "alligator_cracks",
            "alligator": "alligator_cracks",
            "web crack": "alligator_cracks",

            "longitudinal crack": "longitudinal_cracks",
            "longitudinal cracks": "longitudinal_cracks",
            "long crack": "longitudinal_cracks",

            "transverse crack": "transverse_cracks",
            "transverse cracks": "transverse_cracks",
            "cross crack": "transverse_cracks",

            "crack": "longitudinal_cracks",  # Default crack type
            "cracks": "longitudinal_cracks",

            # Road damage
            "road damage": "road_surface_damage",
            "road surface damage": "road_surface_damage",
            "pavement damage": "road_surface_damage",

            # Vehicles
            "abandoned vehicle": "abandoned_vehicle",
            "abandoned car": "abandoned_vehicle",
            "vehicle": "abandoned_vehicle",
            "car": "abandoned_vehicle",

            # Homeless
            "homeless encampment": "homeless_encampment",
            "encampment": "homeless_encampment",
            "tent": "homeless_encampment",
            "homeless person": "homeless_person",
            "homeless": "homeless_person",

            # Infrastructure
            "manhole": "manholes",
            "manholes": "manholes",
            "manhole cover": "manholes",

            "damaged paint": "damaged_paint",
            "road marking": "damaged_paint",
            "road paint": "damaged_paint",
            "faded paint": "damaged_paint",

            "crosswalk": "damaged_crosswalks",
            "damaged crosswalk": "damaged_crosswalks",
            "crosswalks": "damaged_crosswalks",

            "trash": "dumped_trash",
            "dumped trash": "dumped_trash",
            "garbage": "dumped_trash",
            "debris": "dumped_trash",
            "litter": "dumped_trash",

            "street sign": "street_signs",
            "street signs": "street_signs",
            "sign": "street_signs",
            "traffic sign": "street_signs",

            "traffic light": "traffic_lights",
            "traffic lights": "traffic_lights",
            "signal": "traffic_lights",
            "stoplight": "traffic_lights",

            "tyre mark": "tyre_marks",
            "tyre marks": "tyre_marks",
            "tire mark": "tyre_marks",
            "tire marks": "tyre_marks",
            "skid mark": "tyre_marks",
            "skid marks": "tyre_marks",
        }

        # Check direct match
        if prompt_lower in category_mappings:
            return category_mappings[prompt_lower]

        # Check partial match
        for key, category in category_mappings.items():
            if key in prompt_lower or prompt_lower in key:
                return category

        # Return the prompt itself as category if no match
        return prompt_lower.replace(" ", "_")

    def get_all_masks(self) -> List[MaskData]:
        """Get all stored masks."""
        return self.masks

    def remove_masks(self, mask_ids_to_remove: List[int]):
        """
        Remove rejected masks from storage.

        Args:
            mask_ids_to_remove: List of mask IDs to remove
        """
        before_count = len(self.masks)
        self.masks = [m for m in self.masks if m.mask_id not in mask_ids_to_remove]
        after_count = len(self.masks)
        logger.info(f"Removed {before_count - after_count} rejected masks. Remaining: {after_count}")

    def get_used_prompts(self) -> set:
        """Get set of already-used prompts."""
        return self.used_prompts

    def image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
