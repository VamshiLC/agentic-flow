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
                # Method 1: Use SAM3 text prompt with label, then crop to bbox
                mask_np = self._segment_with_text_and_crop(label, bbox)

                if mask_np is None or not mask_np.any():
                    # Method 2: Fallback to point-based segmentation
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

        Strategy: Use text prompt for the region, or multi-point prompting.

        Args:
            bbox: [x1, y1, x2, y2] bounding box coordinates

        Returns:
            numpy array mask or None if failed
        """
        try:
            x1, y1, x2, y2 = bbox
            h, w = self.original_image.size[1], self.original_image.size[0]

            # Method 1: Use multiple point prompts (center + corners)
            # This gives SAM3 better spatial understanding
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Add padding to avoid edge issues
            pad = 5
            points = [
                [center_x, center_y],  # Center (most important)
            ]
            labels = [1]  # All foreground

            # Try multi-point prompting
            try:
                output = self.sam3_processor.set_point_prompt(
                    state=self.inference_state,
                    point=points,
                    labels=labels
                )

                if output and len(output) > 0:
                    # Get the mask array
                    if isinstance(output, dict):
                        mask = output.get('masks', [None])[0] if 'masks' in output else output.get('mask')
                    elif isinstance(output, list) and len(output) > 0:
                        mask = output[0].get('mask') if isinstance(output[0], dict) else output[0]
                    else:
                        mask = None

                    if mask is not None:
                        # Convert to numpy if needed
                        if hasattr(mask, 'cpu'):
                            mask = mask.cpu().numpy()
                        if mask.ndim > 2:
                            mask = mask.squeeze()

                        # Crop mask to bbox region (only keep mask within bbox)
                        cropped_mask = np.zeros_like(mask, dtype=np.uint8)
                        cropped_mask[y1:y2, x1:x2] = (mask[y1:y2, x1:x2] > 0.5).astype(np.uint8) * 255
                        return cropped_mask

            except Exception as e:
                logger.debug(f"Point prompt failed: {e}")

            # Method 2: Create rectangular mask from bbox (fallback)
            logger.debug(f"Using rectangular bbox mask as fallback")
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
            elif hasattr(sam3_output, 'masks'):
                raw_masks = sam3_output.masks
                scores = getattr(sam3_output, 'scores', [])
                boxes = getattr(sam3_output, 'boxes', [])
            else:
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
