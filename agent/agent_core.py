"""
Agent Core - Main Agentic Loop for Infrastructure Detection

This is the core orchestrator that:
1. Sends messages to the LLM (Qwen3-VL)
2. Parses tool calls from responses
3. Executes tools (SAM3 segmentation)
4. Manages conversation history
5. Iterates until detection is complete

Based on: https://github.com/facebookresearch/sam3/blob/main/sam3/agent/agent_core.py
"""
import os
import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
import time

from .system_prompt import get_system_prompt, get_categories
from .tools import ToolExecutor, ToolResult
from .message_manager import MessageManager
from .utils import (
    parse_tool_call,
    parse_tool_call_flexible,
    validate_tool_call,
    normalize_parameters,
    format_error_message,
    build_retry_prompt,
    DebugLogger,
    parse_verdict,
    get_accepted_mask_ids,
    get_rejected_mask_ids
)

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for the agent."""
    max_turns: int = 10  # Quick but thorough
    max_retries: int = 2  # Retry on parse errors
    categories: Optional[List[str]] = None  # Categories to detect
    debug: bool = False  # Enable debug logging
    debug_dir: str = "debug"  # Debug output directory
    prune_after_turns: int = 20  # Prune messages after N turns
    auto_exit_on_masks: bool = False  # Let LLM decide when done
    force_all_categories: bool = True  # Search ALL categories directly with SAM3
    validate_with_llm: bool = True  # Enable LLM validation to filter false positives
    confidence_threshold: float = 0.25  # Lower threshold to catch more
    optimize_memory: bool = True  # Clear Qwen from GPU before SAM3 segmentation


@dataclass
class AgentResult:
    """Result from agent inference."""
    success: bool
    detections: List[Dict[str, Any]]
    num_detections: int
    final_image: Optional[Image.Image]
    turns_taken: int
    message: str


class InfrastructureDetectionAgentCore:
    """
    Core agentic loop for infrastructure detection.

    This implements the SAM3 agent pattern with:
    - Multi-turn conversation with LLM
    - Tool calling (segment_phrase, examine_each_mask, etc.)
    - Iterative refinement
    - Message pruning
    - Few-shot exemplar learning support
    """

    def __init__(
        self,
        qwen_detector,
        sam3_processor,
        config: Optional[AgentConfig] = None,
        exemplar_manager=None,
        exemplar_prompt_builder=None,
        exemplar_strategy: str = "visual_context",
        max_exemplars: int = 3
    ):
        """
        Initialize the agent core.

        Args:
            qwen_detector: Loaded Qwen3VLDirectDetector instance
            sam3_processor: Loaded SAM3 processor
            config: Agent configuration
            exemplar_manager: ExemplarManager instance for few-shot learning
            exemplar_prompt_builder: ExemplarPromptBuilder for multi-image prompts
            exemplar_strategy: Strategy for exemplar prompting ("visual_context", "contrastive", "description")
            max_exemplars: Maximum exemplars per category to use
        """
        self.qwen_detector = qwen_detector
        self.sam3_processor = sam3_processor
        self.config = config or AgentConfig()

        # Few-shot exemplar support
        self.exemplar_manager = exemplar_manager
        self.exemplar_prompt_builder = exemplar_prompt_builder
        self.exemplar_strategy = exemplar_strategy
        self.max_exemplars = max_exemplars

        # Debug logger
        self.debug_logger = DebugLogger(
            enabled=self.config.debug,
            output_dir=self.config.debug_dir
        )

        if exemplar_manager:
            stats = exemplar_manager.get_stats()
            logger.info(f"Agent initialized with {stats['total_exemplars']} exemplars, strategy={exemplar_strategy}")
        logger.info(f"Agent initialized with max_turns={self.config.max_turns}")

    def _optimize_memory_before_sam3(self):
        """
        MEMORY OPTIMIZATION: Clear Qwen model from GPU before SAM3 segmentation.

        This is crucial because:
        1. Qwen3 detection is DONE at this point
        2. SAM3 needs full GPU for accurate segmentation
        3. Having both loaded causes OOM on 22GB GPUs

        The Qwen detector can be reloaded lazily for the next image.
        """
        try:
            print("  [Memory] Optimizing GPU memory for SAM3...")

            # Check if qwen_detector has model to unload
            if hasattr(self.qwen_detector, 'model') and self.qwen_detector.model is not None:
                # Move model to CPU (don't delete, just offload)
                if hasattr(self.qwen_detector.model, 'to'):
                    self.qwen_detector.model.to('cpu')
                    print("  [Memory] Moved Qwen model to CPU")

            # Clear CUDA cache
            torch.cuda.empty_cache()

            # Log memory stats
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"  [Memory] GPU: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

        except Exception as e:
            logger.warning(f"Memory optimization failed (non-critical): {e}")
            # Continue anyway - this is just an optimization

    def _reload_qwen_to_gpu(self):
        """
        Reload Qwen model back to GPU after SAM3 is done.

        Used for batch processing where we need Qwen for the next image.
        """
        try:
            if hasattr(self.qwen_detector, 'model') and self.qwen_detector.model is not None:
                device = self.qwen_detector.device
                if device == "cuda" and torch.cuda.is_available():
                    self.qwen_detector.model.to(device)
                    print("  [Memory] Reloaded Qwen model to GPU")
        except Exception as e:
            logger.warning(f"Failed to reload Qwen to GPU: {e}")

    def run(
        self,
        image: Image.Image,
        user_query: str = "Analyze this road image and detect all infrastructure issues."
    ) -> AgentResult:
        """
        Run the agentic detection loop.

        Args:
            image: PIL Image to analyze
            user_query: Optional custom query

        Returns:
            AgentResult with detections
        """
        start_time = time.time()

        # If force_all_categories is enabled, use direct search instead of LLM loop
        if self.config.force_all_categories:
            return self._run_direct_category_search(image)

        # Run the smart agentic loop with LLM-driven detection
        return self._run_agentic_loop(image, user_query, start_time)

    def _run_direct_category_search(self, image: Image.Image) -> AgentResult:
        """
        QWEN + SAM3 - Smart detection pipeline.
        1. Qwen detects category by category
        2. SAM3 segments each category for precise masks
        Saves output organized by category folders.
        """
        start_time = time.time()
        from PIL import ImageDraw, ImageFont
        import json

        print(f"\n{'='*60}")
        print(f"SMART DETECTION (Qwen + SAM3)")
        print(f"{'='*60}")

        # Initialize SAM3 tool executor
        tool_executor = ToolExecutor(self.sam3_processor, image)

        # STEP 1: Qwen detects objects category by category
        print(f"\n[STEP 1] QWEN DETECTION (Category by Category)")
        print(f"-" * 40)
        qwen_detections = self._ask_qwen_to_detect_with_boxes(image)

        if not qwen_detections:
            print("Qwen found nothing.")
            return AgentResult(
                success=True, detections=[], num_detections=0,
                final_image=image, turns_taken=1,
                message="No infrastructure issues detected"
            )

        print(f"\nQwen found {len(qwen_detections)} objects")

        # STEP 2: SAM3 segments EACH Qwen detection using bbox center as point prompt
        print(f"\n[STEP 2] SAM3 SEGMENTATION (Guided by Qwen bbox centers)")
        print(f"-" * 40)

        detections = []
        img_height, img_width = self.original_image_size if hasattr(self, 'original_image_size') else (image.size[1], image.size[0])

        for i, qwen_det in enumerate(qwen_detections):
            category = qwen_det['label']
            bbox = qwen_det['bbox']
            x1, y1, x2, y2 = bbox

            # Calculate center point of Qwen's bbox to guide SAM3
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            print(f"  [{i+1}/{len(qwen_detections)}] {category} at [{x1},{y1},{x2},{y2}]...", end=" ", flush=True)

            try:
                # Use SAM3 point prompt at center of Qwen's bbox
                # set_point_prompt expects: state, point (list of [x,y]), labels (list of 1=fg/0=bg)
                output = self.sam3_processor.set_point_prompt(
                    state=tool_executor.inference_state,
                    point=[[center_x, center_y]],
                    labels=[1]  # 1 = foreground
                )

                mask = None
                if output:
                    # Extract mask from SAM3 output
                    if isinstance(output, dict):
                        raw_masks = output.get('masks', [])
                        if raw_masks:
                            mask_data = raw_masks[0]
                            if hasattr(mask_data, 'cpu'):
                                mask = mask_data.cpu().numpy()
                            else:
                                mask = np.array(mask_data)
                            if mask.ndim > 2:
                                mask = mask.squeeze()
                    elif hasattr(output, 'masks') and output.masks:
                        mask_data = output.masks[0]
                        if hasattr(mask_data, 'cpu'):
                            mask = mask_data.cpu().numpy()
                        else:
                            mask = np.array(mask_data)
                        if mask.ndim > 2:
                            mask = mask.squeeze()

                # Crop mask to bbox region only (SAM3 may segment beyond bbox)
                if mask is not None:
                    cropped_mask = np.zeros_like(mask, dtype=np.uint8)
                    cropped_mask[y1:y2, x1:x2] = (mask[y1:y2, x1:x2] > 0.5).astype(np.uint8) * 255
                    mask = cropped_mask

                if mask is not None and np.any(mask):
                    detections.append({
                        'label': category,
                        'bbox': bbox,
                        'confidence': qwen_det.get('confidence', 0.8),
                        'mask': mask,
                    })
                    print(f"✓ mask")
                else:
                    # Fallback: use Qwen bbox without mask
                    detections.append({
                        'label': category,
                        'bbox': bbox,
                        'confidence': qwen_det.get('confidence', 0.8),
                        'mask': None,
                    })
                    print(f"- no mask (using bbox)")

            except Exception as e:
                # Fallback: use Qwen detection without SAM3 mask
                detections.append({
                    'label': category,
                    'bbox': bbox,
                    'confidence': qwen_det.get('confidence', 0.8),
                    'mask': None,
                })
                print(f"- error: {str(e)[:30]}")

        print(f"\nTotal: {len(detections)} detections")

        # Filter by user categories if specified
        if self.config.categories and detections:
            user_cats = [c.lower().strip() for c in self.config.categories]
            filtered = []
            for det in detections:
                label = det['label'].lower().strip()
                for cat in user_cats:
                    if cat in label or label in cat:
                        filtered.append(det)
                        break
            print(f"Filtering: {len(detections)} -> {len(filtered)}")
            detections = filtered

        if not detections:
            print("No infrastructure issues detected.")
            return AgentResult(
                success=True, detections=[], num_detections=0,
                final_image=image, turns_taken=1,
                message="No infrastructure issues detected"
            )

        print(f"\nTotal: {len(detections)} objects")

        # Colors for CATEGORY_GROUPS
        colors = {
            # Road defects - Red shades
            'potholes': (255, 0, 0),
            'pothole': (255, 0, 0),
            'cracks': (255, 100, 0),  # Orange for all cracks
            'alligator_cracks': (255, 50, 0),
            'longitudinal_cracks': (255, 100, 0),
            'transverse_cracks': (255, 150, 0),
            # Social issues - Cyan/Purple
            'abandoned_vehicle': (128, 0, 128),
            'homeless_encampment': (0, 255, 255),
            'homeless_person': (0, 200, 200),
            # Infrastructure - Green/Yellow/Blue
            'manholes': (0, 255, 0),
            'manhole': (0, 255, 0),
            'damaged_paint': (255, 255, 0),
            'damaged_crosswalks': (200, 200, 255),
            'dumped_trash': (139, 69, 19),
            'street_signs': (255, 200, 0),
            'traffic_lights': (0, 255, 100),
            'tyre_marks': (100, 100, 100),
            'graffiti': (255, 0, 255),  # Magenta for graffiti
        }

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            font = ImageFont.load_default()

        # Group detections by category
        by_category = {}
        for det in detections:
            cat = det['label']
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(det)

        # Create output directory structure
        output_base = self.config.debug_dir if self.config.debug else "output"
        os.makedirs(output_base, exist_ok=True)

        # Save per-category images and data
        print(f"\n{'='*60}")
        print(f"SAVING BY CATEGORY:")
        print(f"{'='*60}")

        final_detections = []
        detection_id = 0

        for category, cat_detections in by_category.items():
            # Create category folder
            cat_folder = os.path.join(output_base, category)
            os.makedirs(cat_folder, exist_ok=True)

            # Draw image with only this category's detections
            cat_image = image.copy()
            cat_draw = ImageDraw.Draw(cat_image)

            color = colors.get(category, (255, 255, 0))

            cat_data = []
            for det in cat_detections:
                detection_id += 1
                x1, y1, x2, y2 = det['bbox']
                confidence = det.get('confidence', 0.8)
                mask = det.get('mask')

                # Draw SAM3 mask if available (semi-transparent overlay)
                if mask is not None:
                    try:
                        import numpy as np
                        mask_overlay = Image.new('RGBA', cat_image.size, (0, 0, 0, 0))
                        mask_array = np.array(mask)
                        if mask_array.shape[:2] == (cat_image.height, cat_image.width):
                            rgba_color = color + (100,)  # Semi-transparent
                            mask_rgba = np.zeros((*mask_array.shape[:2], 4), dtype=np.uint8)
                            mask_rgba[mask_array > 0] = rgba_color
                            mask_img = Image.fromarray(mask_rgba, 'RGBA')
                            cat_image = Image.alpha_composite(cat_image.convert('RGBA'), mask_img).convert('RGB')
                            cat_draw = ImageDraw.Draw(cat_image)
                    except Exception as e:
                        pass  # Fallback to bbox only

                # Draw bounding box
                cat_draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                try:
                    text_bbox = cat_draw.textbbox((x1, y1 - 20), category, font=font)
                    cat_draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2], fill=color)
                    cat_draw.text((x1, y1 - 20), category, fill=(0, 0, 0), font=font)
                except:
                    cat_draw.text((x1, y1 - 15), category, fill=color)

                det_info = {
                    "id": detection_id,
                    "category": category,
                    "bbox": det['bbox'],
                    "confidence": confidence,
                    "has_mask": mask is not None,
                }
                cat_data.append(det_info)
                final_detections.append({
                    "mask_id": detection_id, "category": category, "severity": "medium",
                    "confidence": confidence, "bbox": det['bbox'], "mask": mask,
                })

            # Save category image
            cat_image_path = os.path.join(cat_folder, f"{category}.png")
            cat_image.save(cat_image_path)

            # Save category JSON
            cat_json_path = os.path.join(cat_folder, f"{category}.json")
            with open(cat_json_path, 'w') as f:
                json.dump({"category": category, "count": len(cat_data), "detections": cat_data}, f, indent=2)

            print(f"  ✓ {category}: {len(cat_detections)} detections -> {cat_folder}/")

        # Create combined image with all detections (masks + boxes)
        result_image = image.copy().convert('RGBA')
        import numpy as np

        # First draw all SAM3 masks
        for det in detections:
            mask = det.get('mask')
            if mask is not None:
                try:
                    label = det['label']
                    color = colors.get(label, (255, 255, 0))
                    mask_array = np.array(mask)
                    if mask_array.shape[:2] == (image.height, image.width):
                        rgba_color = color + (80,)  # Semi-transparent
                        mask_rgba = np.zeros((*mask_array.shape[:2], 4), dtype=np.uint8)
                        mask_rgba[mask_array > 0] = rgba_color
                        mask_img = Image.fromarray(mask_rgba, 'RGBA')
                        result_image = Image.alpha_composite(result_image, mask_img)
                except:
                    pass

        result_image = result_image.convert('RGB')
        draw = ImageDraw.Draw(result_image)

        # Then draw bounding boxes and labels
        for det in detections:
            label = det['label']
            x1, y1, x2, y2 = det['bbox']
            color = colors.get(label, (255, 255, 0))

            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            try:
                text_bbox = draw.textbbox((x1, y1 - 20), label, font=font)
                draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2], fill=color)
                draw.text((x1, y1 - 20), label, fill=(0, 0, 0), font=font)
            except:
                draw.text((x1, y1 - 15), label, fill=color)

        # Save summary JSON (exclude numpy arrays which can't be serialized)
        json_safe_detections = []
        for det in final_detections:
            safe_det = {
                "mask_id": det["mask_id"],
                "category": det["category"],
                "severity": det["severity"],
                "confidence": float(det["confidence"]) if det["confidence"] else 0.8,
                "bbox": det["bbox"],
                "has_mask": det.get("mask") is not None,
            }
            json_safe_detections.append(safe_det)

        summary = {
            "total_detections": len(final_detections),
            "categories_found": list(by_category.keys()),
            "by_category": {cat: len(dets) for cat, dets in by_category.items()},
            "detections": json_safe_detections,
        }
        summary_path = os.path.join(output_base, "summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Summary: {summary_path}")

        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"RESULTS: {len(final_detections)} objects detected ({elapsed:.2f}s)")
        print(f"{'='*60}")

        return AgentResult(
            success=True, detections=final_detections,
            num_detections=len(final_detections), final_image=result_image,
            turns_taken=1, message=f"Found {len(final_detections)} issues"
        )

    def _ask_qwen_to_detect_with_boxes(self, image: Image.Image) -> List[Dict]:
        """Ask Qwen to detect - category by category for accuracy."""
        img_width, img_height = image.size

        # All categories with DETAILED descriptions for better detection
        CATEGORIES = {
            "road_defects": [
                ("potholes", "holes, depressions, or cavities in road pavement - dark circular or irregular shapes in the road"),
                ("alligator_cracks", "interconnected cracks forming a pattern resembling alligator skin or spider web - multiple small polygonal pieces of cracked pavement"),
                ("longitudinal_cracks", "cracks running parallel to the road direction/centerline - long straight or slightly curved cracks along the road"),
                ("transverse_cracks", "cracks running perpendicular/across the road direction - cracks that go from one side of the lane to the other"),
            ],
            "social_issues": [
                ("homeless_encampment", "tents, tarps, blue tarps, makeshift shelters, camping tents, fabric shelters, cardboard shelters on sidewalk or street - ANY tent or tarp structure"),
                ("homeless_person", "person sleeping on ground, person sitting on sidewalk with belongings, person lying down on street"),
                ("abandoned_vehicle", "old rusted car, damaged vehicle, car with flat tires, car covered in dust/dirt, vehicle that looks unused or broken down"),
            ],
            "infrastructure": [
                ("manholes", "round or square metal covers on road surface, utility access covers, sewer covers"),
                ("damaged_crosswalks", "faded WHITE PAINTED zebra stripes of pedestrian crossing - NOT road cracks or pavement damage"),
                ("dumped_trash", "garbage bags, piles of trash, illegally dumped items, debris, litter piles, discarded furniture or appliances"),
                ("street_signs", "stop signs, speed limit signs, street name signs, warning signs, any road signage"),
                ("traffic_lights", "traffic signal lights, red/yellow/green lights at intersections"),
                ("graffiti", "spray paint, tags, vandalism, painted text or images on walls, buildings, fences, or any surface"),
            ],
        }

        all_detections = []

        # Search each category separately for accuracy
        for group_name, categories in CATEGORIES.items():
            print(f"\n[{group_name.upper()}]")
            for category, description in categories:
                detections = self._detect_single_category(image, category, description, img_width, img_height)
                if detections:
                    # FILTER bad bounding boxes for road defects
                    filtered = self._filter_bad_detections(detections, img_width, img_height)
                    if filtered:
                        print(f"  ✓ {category}: {len(filtered)} found (filtered from {len(detections)})")
                        all_detections.extend(filtered)
                    else:
                        print(f"  - {category}: 0 (all {len(detections)} rejected by filter)")
                else:
                    print(f"  - {category}: 0")

        # Remove duplicate detections (same area detected as multiple crack types)
        if all_detections:
            before_count = len(all_detections)
            all_detections = self._remove_duplicate_detections(all_detections, iou_threshold=0.4)
            if len(all_detections) < before_count:
                print(f"\n[DEDUP] Removed {before_count - len(all_detections)} overlapping detections")

        return all_detections

    def _filter_bad_detections(self, detections: List[Dict], img_width: int, img_height: int) -> List[Dict]:
        """
        MINIMAL filtering - only reject OBVIOUS false positives.

        Be VERY permissive - cracks on roads can appear anywhere in perspective.
        Only reject boxes that are clearly impossible (like entirely in sky).
        """
        # DISABLED - just return all detections
        # The model knows what cracks look like, trust it
        return detections

    def _remove_duplicate_detections(self, all_detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """
        Remove duplicate/overlapping detections across categories.

        When the same area is detected as multiple crack types, keep only ONE.
        Priority: potholes > alligator_cracks > longitudinal_cracks > transverse_cracks
        """
        if len(all_detections) <= 1:
            return all_detections

        # Priority order (lower = higher priority)
        priority = {
            'potholes': 1,
            'alligator_cracks': 2,
            'longitudinal_cracks': 3,
            'transverse_cracks': 4,
        }

        def calc_iou(box1, box2):
            """Calculate Intersection over Union."""
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])

            if x2 <= x1 or y2 <= y1:
                return 0.0

            intersection = (x2 - x1) * (y2 - y1)
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union = area1 + area2 - intersection

            return intersection / union if union > 0 else 0.0

        # Sort by priority
        sorted_dets = sorted(all_detections, key=lambda d: priority.get(d['label'], 99))

        keep = []
        for det in sorted_dets:
            is_duplicate = False
            for kept in keep:
                iou = calc_iou(det['bbox'], kept['bbox'])
                if iou > iou_threshold:
                    is_duplicate = True
                    print(f"      [DEDUP] Removing duplicate {det['label']} (IoU={iou:.2f} with {kept['label']})")
                    break
            if not is_duplicate:
                keep.append(det)

        return keep

    def _detect_single_category(self, image: Image.Image, category: str, description: str, img_width: int, img_height: int) -> List[Dict]:
        """Detect a single category, using MERGED exemplar approach if available.

        Strategy: Merge exemplar image INTO target image, then ask model to find
        similar patterns. This works better than multi-image prompts.
        """
        # Check if we should use exemplar-enhanced detection (MERGED approach)
        use_exemplars = (
            self.exemplar_manager is not None and
            self.exemplar_prompt_builder is not None and
            self.exemplar_manager.has_exemplars(category)
        )

        if use_exemplars:
            return self._detect_with_exemplars(image, category, description, img_width, img_height)

        # Standard text-only detection
        return self._detect_with_text_only(image, category, description, img_width, img_height)

    def _detect_with_text_only(self, image: Image.Image, category: str, description: str, img_width: int, img_height: int) -> List[Dict]:
        """Standard detection using text prompts only (no exemplars)."""

        # Special STRICT prompt for potholes only (to avoid false positives)
        if category == "potholes":
            prompt = f"""Analyze this image for: potholes

A pothole is a HOLE or DEPRESSION in the road surface with visible DEPTH.

CRITICAL - DO NOT DETECT:
- Shadows (they have no depth)
- Dark patches or stains
- Normal road texture
- Wet spots or reflections

ONLY detect if you see an ACTUAL HOLE with visible depth/damage.

If you find REAL potholes, output JSON:
[{{"label": "potholes", "bbox_2d": [x1, y1, x2, y2]}}]

Image size: {img_width} x {img_height} pixels.

If NO potholes exist, output: []"""

        # Special prompts for CRACK detection
        elif category == "longitudinal_cracks":
            prompt = f"""Look carefully at the ROAD SURFACE for longitudinal cracks.

Longitudinal cracks look like:
- Long dark lines running along the direction of the road
- Single crack lines or parallel crack lines
- Usually appear as dark lines on gray asphalt
- Run from near to far in the image (following road direction)
- May branch or have smaller cracks alongside

Scan the entire road surface from bottom to top of the image.

Output JSON array with bounding boxes around each crack:
[{{"label": "longitudinal_cracks", "bbox_2d": [x1, y1, x2, y2]}}]

Image size: {img_width} x {img_height} pixels.
If NO longitudinal cracks visible, output: []"""

        elif category == "transverse_cracks":
            prompt = f"""Look carefully at the ROAD SURFACE for transverse cracks.

Transverse cracks look like:
- Dark lines running ACROSS the road (left to right)
- Perpendicular to the direction of travel
- Single lines or groups of parallel lines crossing the road
- Usually appear as horizontal dark lines on gray asphalt

Scan the road surface for any cracks running across/horizontally.

Output JSON array with bounding boxes around each crack:
[{{"label": "transverse_cracks", "bbox_2d": [x1, y1, x2, y2]}}]

Image size: {img_width} x {img_height} pixels.
If NO transverse cracks visible, output: []"""

        elif category == "alligator_cracks":
            prompt = f"""Look carefully at the ROAD SURFACE in this image for alligator cracks.

Alligator cracks look like:
- Network of interconnected cracks forming a web/mesh pattern
- Resembles alligator skin or dried mud
- Dark lines forming irregular polygon shapes on gray asphalt
- Usually found in areas with heavy traffic or old pavement
- Can be small patches or large areas of cracked surface

Look at the asphalt/road surface. Find ANY areas where you see this cracked pattern.

Output JSON array with bounding boxes around each cracked area:
[{{"label": "alligator_cracks", "bbox_2d": [x1, y1, x2, y2]}}]

Image size: {img_width} x {img_height} pixels.
If NO alligator cracks visible on road, output: []"""

        elif category == "damaged_crosswalks":
            prompt = f"""Find damaged or faded crosswalks in this road image.

A crosswalk is WHITE PAINTED ZEBRA STRIPES for pedestrians to cross.

WHAT TO LOOK FOR:
- Faded white painted stripes that are worn/damaged
- Zebra crossing markings that are barely visible
- White parallel lines painted across the road

DO NOT DETECT:
- Road cracks or pavement damage (those are NOT crosswalks)
- Dark lines/cracks in asphalt (NOT crosswalks)
- Lane markings or road lines (NOT crosswalks)

ONLY output if you see actual WHITE PAINTED crosswalk stripes that are damaged/faded.

Output JSON:
[{{"label": "damaged_crosswalks", "bbox_2d": [x1, y1, x2, y2]}}]

Image size: {img_width} x {img_height} pixels.
If NO damaged crosswalks, output: []"""

        else:
            # Normal prompt for all other categories
            prompt = f"""Look at this image. Find ALL instances of: {category}

What to look for: {description}

IMPORTANT: Draw TIGHT bounding boxes around each object. Do NOT draw boxes larger than the actual object.

Output format - JSON array with bounding boxes:
[{{"label": "{category}", "bbox_2d": [x1, y1, x2, y2]}}]

Image size: {img_width} x {img_height} pixels.
x1,y1 = top-left corner. x2,y2 = bottom-right corner.

If you find {category}, output the JSON. If nothing found, output: []"""

        try:
            result = self.qwen_detector.detect(image, prompt)
            if not result.get("success"):
                return []
            response_text = result.get("text", "")
            detections = self._parse_json_detection_response(response_text, img_width, img_height)
            for d in detections:
                d['label'] = category  # Force correct label

            return detections
        except Exception as e:
            logger.error(f"Detection error for {category}: {e}")
            return []

    def _merge_exemplar_with_target(self, exemplar_img: Image.Image, target_img: Image.Image) -> Image.Image:
        """
        Merge exemplar image with target image side-by-side.

        Layout: [EXEMPLAR | TARGET]
        The exemplar is resized to fit on the left, target on the right.
        """
        # Resize exemplar to be smaller (reference size)
        exemplar_height = target_img.height // 3  # 1/3 of target height
        aspect_ratio = exemplar_img.width / exemplar_img.height
        exemplar_width = int(exemplar_height * aspect_ratio)

        # Resize exemplar
        exemplar_resized = exemplar_img.resize((exemplar_width, exemplar_height), Image.Resampling.LANCZOS)

        # Create combined image (exemplar on top-left corner of target)
        combined = target_img.copy()

        # Add a border around exemplar
        from PIL import ImageDraw
        border = 3

        # Paste exemplar in top-left with border
        paste_x, paste_y = 10, 10

        # Draw border rectangle
        draw = ImageDraw.Draw(combined)
        draw.rectangle(
            [paste_x - border, paste_y - border,
             paste_x + exemplar_width + border, paste_y + exemplar_height + border],
            outline=(255, 0, 0), width=border
        )

        # Paste exemplar
        combined.paste(exemplar_resized, (paste_x, paste_y))

        # Add label "EXAMPLE"
        try:
            from PIL import ImageFont
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        except:
            font = ImageFont.load_default()

        draw.rectangle([paste_x, paste_y + exemplar_height + border, paste_x + 80, paste_y + exemplar_height + border + 20], fill=(255, 0, 0))
        draw.text((paste_x + 5, paste_y + exemplar_height + border + 2), "EXAMPLE", fill=(255, 255, 255), font=font)

        return combined

    def _detect_with_exemplars(self, image: Image.Image, category: str, description: str, img_width: int, img_height: int) -> List[Dict]:
        """
        Enhanced detection using few-shot exemplars.

        STRATEGY: Merge exemplar image INTO the target image, then ask model
        to find similar patterns. This works better than multi-image prompts.

        Args:
            image: Target image to analyze
            category: Category to detect (e.g., "potholes")
            description: Text description of the category
            img_width: Image width in pixels
            img_height: Image height in pixels

        Returns:
            List of detections with bboxes
        """
        try:
            print(f"      [EXEMPLAR] Using MERGED exemplar detection for '{category}'")

            # Get exemplar images
            exemplars = self.exemplar_manager.get_positive_exemplars(category, max_count=1)

            if not exemplars or not exemplars[0].image:
                print(f"      [EXEMPLAR] No exemplars found for '{category}', falling back to text-only")
                return self._detect_with_text_only(image, category, description, img_width, img_height)

            exemplar_img = exemplars[0].image
            print(f"      [EXEMPLAR] Got 1 exemplar image, merging with target...")

            # MERGE exemplar INTO target image
            merged_image = self._merge_exemplar_with_target(exemplar_img, image)

            # Get display name
            display_name = category.replace("_", " ")

            # Calculate where the exemplar box is in the merged image (to exclude from detection)
            exemplar_height = img_height // 3
            aspect_ratio = exemplar_img.width / exemplar_img.height
            exemplar_width = int(exemplar_height * aspect_ratio)
            exemplar_box_right = 10 + exemplar_width + 10
            exemplar_box_bottom = 10 + exemplar_height + 30

            # Prompt that references the merged example
            merged_prompt = f"""Look at this image. In the TOP-LEFT corner (red border, "EXAMPLE" label) is a reference showing {display_name}.

Find ALL INDIVIDUAL {display_name} in the road area of this image.

CRITICAL - BOUNDING BOX RULES:
- Draw a SEPARATE small box around EACH individual {display_name}
- Each box should be TIGHT - only around ONE crack/defect
- Do NOT draw one big box covering the whole road
- If there are 5 cracks, output 5 separate small boxes
- Box width/height should typically be 50-300 pixels, NOT 1000+ pixels

Output JSON array with MULTIPLE small bounding boxes:
[{{"label": "{category}", "bbox_2d": [x1, y1, x2, y2]}}, {{"label": "{category}", "bbox_2d": [x1, y1, x2, y2]}}, ...]

Image size: {img_width} x {img_height} pixels.
Exclude the example area (top-left corner up to x={exemplar_box_right}, y={exemplar_box_bottom}).
If no {display_name} found, output: []"""

            print(f"      [EXEMPLAR] Running detection on merged image...")
            result = self.qwen_detector.detect(merged_image, merged_prompt)

            if not result.get("success"):
                print(f"      [EXEMPLAR] Detection FAILED: {result.get('error', 'unknown')}")
                return self._detect_with_text_only(image, category, description, img_width, img_height)

            response_text = result.get("text", "")
            print(f"      [EXEMPLAR] Model response: {response_text[:300]}...")
            detections = self._parse_json_detection_response(response_text, img_width, img_height)

            # Filter out any detections that overlap with the exemplar box
            filtered_detections = []
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                # Keep if detection is mostly outside the exemplar area
                if x1 > exemplar_box_right or y1 > exemplar_box_bottom:
                    filtered_detections.append(det)
                else:
                    print(f"      [EXEMPLAR] Filtered out detection in exemplar area: {det['bbox']}")

            detections = filtered_detections
            print(f"      [EXEMPLAR] Parsed {len(detections)} detections for '{category}'")

            # Force correct label
            for d in detections:
                d['label'] = category
                d['used_exemplars'] = True

            if detections:
                print(f"      [EXEMPLAR] SUCCESS: Found {len(detections)} {category}")
            else:
                # FALLBACK: Exemplar detection returned nothing, try text-only
                print(f"      [EXEMPLAR] Returned 0 results, trying text-only fallback...")
                text_detections = self._detect_with_text_only(image, category, description, img_width, img_height)
                if text_detections:
                    print(f"      [FALLBACK] Text-only found {len(text_detections)} {category}")
                    return text_detections

            return detections

        except Exception as e:
            print(f"      [EXEMPLAR] ERROR for {category}: {e}")
            # Fallback to text-only detection
            return self._detect_with_text_only(image, category, description, img_width, img_height)

    def _parse_json_detection_response(self, response: str, img_width: int, img_height: int) -> List[Dict]:
        """Parse JSON detection response from Qwen."""
        import json
        detections = []

        try:
            # Find JSON in response
            json_str = None
            if '```json' in response:
                start = response.find('```json') + 7
                end = response.find('```', start)
                if end != -1:
                    json_str = response[start:end].strip()

            if not json_str:
                start = response.find('[')
                if start != -1:
                    bracket_count = 0
                    for i, c in enumerate(response[start:], start):
                        if c == '[': bracket_count += 1
                        elif c == ']':
                            bracket_count -= 1
                            if bracket_count == 0:
                                json_str = response[start:i+1]
                                break

            if json_str:
                parsed = json.loads(json_str)
                if isinstance(parsed, list):
                    for item in parsed:
                        bbox = item.get('bbox_2d') or item.get('bbox')
                        label = item.get('label', 'unknown')
                        if bbox and len(bbox) == 4:
                            x1, y1, x2, y2 = [int(float(b)) for b in bbox]
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(img_width, x2), min(img_height, y2)
                            if x1 < x2 and y1 < y2:
                                # No filtering - trust the model
                                detections.append({'label': label, 'bbox': [x1,y1,x2,y2], 'confidence': 0.8})
        except:
            pass

        return detections

    def _run_text_prompt_search(
        self,
        image: Image.Image,
        categories: List[str],
        tool_executor: ToolExecutor,
        start_time: float
    ) -> AgentResult:
        """
        Fallback: Search using SAM3 text prompts (old approach).

        Used when:
        - User specifies specific categories
        - Smart detection fails
        """
        print(f"\n{'='*60}")
        print(f"TEXT PROMPT SEARCH: {len(categories)} categories")
        print(f"{'='*60}")
        logger.info(f"=== SEARCHING {len(categories)} CATEGORIES ===")

        # Search EVERY category
        found_categories = []
        total_masks_found = 0
        for i, category in enumerate(categories):
            print(f"[{i+1}/{len(categories)}] Searching: {category}...", end=" ", flush=True)
            logger.info(f"[{i+1}/{len(categories)}] Searching: {category}")

            try:
                result = tool_executor.execute("segment_phrase", {"text_prompt": category})
                if result.success and result.data.get("num_masks", 0) > 0:
                    num_found = result.data['num_masks']
                    total_masks_found += num_found
                    print(f"✓ {num_found} mask(s)")
                    logger.info(f"  ✓ Found {num_found} mask(s) for '{category}'")
                    found_categories.append(category)
                else:
                    print("✗ 0 masks")
                    logger.debug(f"  ✗ No masks for '{category}'")
            except Exception as e:
                print(f"ERROR: {e}")
                logger.warning(f"  Error searching '{category}': {e}")

        # Get all masks found
        masks = tool_executor.get_all_masks()
        print(f"\n{'='*60}")
        print(f"TOTAL: {len(masks)} masks from {len(found_categories)} categories")
        print(f"Categories with detections: {found_categories}")
        print(f"{'='*60}")
        logger.info(f"=== TOTAL: {len(masks)} masks from {len(found_categories)} categories ===")

        # Filter by confidence threshold first
        if masks:
            before_count = len(masks)
            masks = [m for m in masks if m.score >= self.config.confidence_threshold]
            filtered_count = before_count - len(masks)
            if filtered_count > 0:
                print(f"Filtered {filtered_count} low-confidence masks (threshold: {self.config.confidence_threshold})")
                logger.info(f"Filtered {filtered_count} low-confidence masks (threshold: {self.config.confidence_threshold})")
            print(f"Keeping {len(masks)} masks after confidence filter")
            logger.info(f"Keeping {len(masks)} high-confidence masks")

        # LLM VALIDATION - The smart part!
        # Have the LLM look at each mask and decide if it's correct
        if masks and self.config.validate_with_llm:
            print(f"\n{'='*60}")
            print(f"LLM VALIDATION: Checking {len(masks)} masks...")
            print(f"{'='*60}")
            masks = self._validate_masks_with_llm(image, masks, tool_executor)
            print(f"After LLM validation: {len(masks)} masks kept")

        # Build final detections
        if masks:
            print(f"\nRendering {len(masks)} masks on image...")
            detections = [
                {
                    "mask_id": m.mask_id,
                    "category": m.category,
                    "severity": tool_executor._category_to_severity(m.category),
                    "confidence": m.score,
                    "bbox": m.bbox,
                    "mask": m.mask,
                }
                for m in masks
            ]
            final_image = tool_executor._render_masks(masks)
            print(f"✓ Mask overlay rendered successfully")
        else:
            print("\n⚠ No masks found - returning original image without overlay")
            detections = []
            final_image = image

        elapsed = time.time() - start_time
        logger.info(f"Detection complete: {len(detections)} issues in {elapsed:.2f}s")

        return AgentResult(
            success=True,
            detections=detections,
            num_detections=len(detections),
            final_image=final_image,
            turns_taken=len(categories),
            message=f"Found {len(detections)} infrastructure issues across {len(found_categories)} categories"
        )

    def _validate_masks_with_llm(
        self,
        image: Image.Image,
        masks: List,
        tool_executor: ToolExecutor
    ) -> List:
        """
        Use LLM to validate EACH mask individually - OFFICIAL SAM3 AGENT STYLE.

        For each mask, we show the LLM:
        1. The original raw image
        2. The image with mask overlay
        3. A zoomed-in view of the mask region

        Then ask for detailed reasoning + Accept/Reject verdict.

        Args:
            image: Original image
            masks: List of MaskData objects
            tool_executor: Tool executor for rendering

        Returns:
            Filtered list of validated masks
        """
        if not masks:
            return masks

        logger.info(f"Validating {len(masks)} masks with LLM (SAM3 iterative checking)...")

        validated_masks = []

        for i, mask in enumerate(masks):
            print(f"[{i+1}/{len(masks)}] Validating '{mask.category}'...", end=" ", flush=True)

            try:
                # Create mask overlay image
                single_mask_image = tool_executor._render_masks([mask])

                # Create zoomed view of mask region
                zoomed_image = self._create_zoomed_mask_view(image, mask)

                # OFFICIAL SAM3 ITERATIVE CHECKING PROMPT
                # This is the key to accuracy - detailed reasoning required
                validation_prompt = f"""You are a visual grounding validation assistant. Your task is to carefully analyze whether a predicted segmentation mask correctly identifies the claimed object.

USER QUERY: Detect "{mask.category}" in this road/infrastructure image.

PREDICTED MASK: The colored overlay shows what the segmentation model detected as "{mask.category}".

=== VALIDATION INSTRUCTIONS ===

1. CAREFULLY EXAMINE the masked region in the image.

2. DETERMINE what object the mask ACTUALLY covers. Describe it:
   - What is its shape? (circular, irregular, linear, rectangular)
   - What is its texture/appearance? (metal, broken pavement, smooth, rough)
   - What is its context? (on road surface, on sidewalk, infrastructure)

3. COMPARE to the claimed category "{mask.category}":

   MANHOLE/MANHOLE COVER characteristics:
   - ROUND or RECTANGULAR shape with CLEAN EDGES
   - METAL surface with patterns, text, grid, or raised design
   - FLUSH with road surface or slightly raised
   - Has a DELIBERATE manufactured appearance
   - Usually has utility company markings

   POTHOLE characteristics:
   - IRREGULAR shape with JAGGED EDGES
   - Shows BROKEN/MISSING pavement
   - Has DEPTH - you can see into it
   - Rough, damaged texture
   - May have debris or water inside

   CRACK characteristics:
   - LINEAR or BRANCHING pattern
   - Follows the pavement surface
   - No significant depth
   - May have vegetation growing

   GRAFFITI characteristics:
   - SPRAY PAINT or painted text/images on walls, surfaces
   - Has COLORS (not natural weathering)
   - Shows LETTERS, WORDS, or ARTISTIC patterns
   - On walls, buildings, signs, or infrastructure

   FALSE POSITIVES to REJECT:
   - Shadows (no physical depth, follows light direction)
   - Wet spots/puddles (reflective, no damage)
   - Oil stains (discoloration only, no structural damage)
   - Road paint/markings (deliberate, clean edges)
   - Normal pavement texture/joints

4. DECIDE: Does the mask correctly show a {mask.category}?

=== RESPONSE FORMAT ===

<think>
[Your detailed analysis here - describe what you see, compare to expected characteristics]
</think>

<verdict>Accept</verdict> OR <verdict>Reject</verdict>

IMPORTANT: Be STRICT. Only Accept if the mask clearly shows the claimed object type. Reject shadows, stains, and misidentified objects."""

                # Call LLM with the mask image
                result = self.qwen_detector.detect(single_mask_image, validation_prompt)

                if result.get("success"):
                    response = result.get("text", "")
                    response_lower = response.lower()

                    # Parse verdict from response
                    if "<verdict>accept</verdict>" in response_lower:
                        print("✓ Accept")
                        validated_masks.append(mask)
                    elif "<verdict>reject</verdict>" in response_lower:
                        # Extract reasoning if available
                        if "<think>" in response_lower:
                            think_text = response.split("<think>")[-1].split("</think>")[0][:100]
                            print(f"✗ Reject - {think_text.strip()}...")
                        else:
                            print("✗ Reject")
                        logger.info(f"Mask {mask.mask_id} rejected: {mask.category}")
                    elif "accept" in response_lower.split()[-10:]:
                        print("✓ Accept (implicit)")
                        validated_masks.append(mask)
                    elif "reject" in response_lower.split()[-10:]:
                        print("✗ Reject (implicit)")
                    else:
                        # Default to REJECT for unclear responses (be strict)
                        print("? Unclear → Reject (strict mode)")
                        logger.info(f"Mask {mask.mask_id} rejected (unclear): {mask.category}")
                else:
                    # LLM error (likely OOM) - keep the mask rather than reject
                    print("? LLM Error → Keep (fallback)")
                    validated_masks.append(mask)

            except Exception as e:
                print(f"? Error: {e} → Keep (fallback)")
                logger.error(f"Validation error for mask {mask.mask_id}: {e}")
                validated_masks.append(mask)  # Keep on error

        rejected_count = len(masks) - len(validated_masks)
        print(f"\n{'='*50}")
        print(f"VALIDATION COMPLETE: {len(validated_masks)} accepted, {rejected_count} rejected")
        print(f"{'='*50}")

        return validated_masks

    def _create_zoomed_mask_view(self, image: Image.Image, mask) -> Image.Image:
        """Create a zoomed-in view of the mask's bounding box area."""
        bbox = mask.bbox  # [x1, y1, x2, y2]

        if not bbox or bbox == [0, 0, 0, 0]:
            return image

        # Add padding around bbox (30% of bbox size)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        padding_x = int(width * 0.3)
        padding_y = int(height * 0.3)

        x1 = max(0, bbox[0] - padding_x)
        y1 = max(0, bbox[1] - padding_y)
        x2 = min(image.width, bbox[2] + padding_x)
        y2 = min(image.height, bbox[3] + padding_y)

        # Crop to bbox region
        cropped = image.crop((x1, y1, x2, y2))

        return cropped

    def _ask_qwen_what_to_detect(self, image: Image.Image) -> List[str]:
        """
        Ask Qwen VL to analyze the image and tell us what infrastructure issues are present.
        Returns list of category names for fallback mode.
        """
        detections = self._ask_qwen_to_detect_with_boxes(image)
        return [d['label'] for d in detections]

    def _ask_qwen_what_it_sees(self, image: Image.Image) -> List[str]:
        """
        SIMPLE: Ask Qwen what infrastructure issues it sees.
        Returns list of things to search with SAM3.
        """
        detection_prompt = """Analyze this street/road image and identify ANY of these issues if present:

1. ROAD DAMAGE: potholes, cracks, broken pavement
2. GRAFFITI: spray paint, tags, vandalism on walls
3. MANHOLES: metal covers, drain grates on road
4. TRASH/DEBRIS: garbage, litter, dumped items
5. HOMELESS: tents, encampments, sleeping bags, belongings
6. ABANDONED VEHICLES: cars, trucks that look abandoned
7. DAMAGED SIGNS: broken, bent, defaced signs
8. DAMAGED LIGHTS: broken street lights
9. ILLEGAL DUMPING: large items dumped on street
10. BLOCKED SIDEWALKS: obstructions

List everything you see. Be specific about any problems or issues."""

        try:
            result = self.qwen_detector.detect(image, detection_prompt)

            if not result.get("success"):
                logger.error("Qwen detection failed")
                return []

            response = result.get("text", "").lower()
            print(f"Qwen says: {response}")

            # Extract what Qwen found - map to search terms
            # COMPREHENSIVE keyword list for all infrastructure issues
            found_items = []
            keywords = {
                # Graffiti
                'graffiti': 'graffiti',
                'spray paint': 'graffiti',
                'vandalism': 'graffiti',
                'tag': 'graffiti',
                'tagged': 'graffiti',
                # Road damage
                'pothole': 'pothole',
                'potholes': 'pothole',
                'crack': 'crack',
                'cracked': 'crack',
                'broken pavement': 'pothole',
                # Manholes
                'manhole': 'manhole',
                'manhole cover': 'manhole',
                'metal cover': 'manhole',
                'grate': 'manhole',
                'drain': 'manhole',
                'drainage': 'manhole',
                # Trash & Debris
                'debris': 'trash',
                'trash': 'trash',
                'garbage': 'trash',
                'litter': 'trash',
                'dumped': 'illegal dumping',
                'illegal dumping': 'illegal dumping',
                # Homeless / Encampments - use simple terms SAM3 understands
                'homeless': 'tent',
                'tent': 'tent',
                'encampment': 'tent',
                'sleeping bag': 'tent',
                'belongings': 'tent',
                'camp': 'tent',
                'tarp': 'tent',
                'blanket': 'tent',
                # Abandoned vehicles - more keywords
                'abandoned': 'car',
                'abandoned vehicle': 'car',
                'abandoned car': 'car',
                'derelict': 'car',
                'broken down': 'car',
                'parked car': 'car',
                'old car': 'car',
                'rusty car': 'car',
                'damaged car': 'car',
                'wrecked': 'car',
                # Signs - only damaged ones are issues (normal signs are not problems)
                'damaged sign': 'damaged sign',
                'broken sign': 'damaged sign',
                'bent sign': 'damaged sign',
                'defaced sign': 'damaged sign',
                # Lights - only damaged ones are issues
                'broken light': 'damaged light',
                'damaged light': 'damaged light',
                'non-functioning light': 'damaged light',
                # Crosswalk - only damaged
                'faded crosswalk': 'damaged crosswalk',
                'damaged crosswalk': 'damaged crosswalk',
                # Sidewalk issues
                'blocked sidewalk': 'blocked sidewalk',
                'obstruction': 'blocked sidewalk',
            }

            for keyword, label in keywords.items():
                if keyword in response and label not in found_items:
                    found_items.append(label)

            if found_items:
                print(f"Qwen identified: {found_items}")
            else:
                print("Qwen didn't find any infrastructure issues")

            return found_items

        except Exception as e:
            logger.error(f"Qwen detection error: {e}")
            print(f"Qwen detection error: {e}")
            return []

    def _parse_detection_response(self, response: str, img_width: int, img_height: int) -> List[Dict]:
        """
        Parse Qwen's detection response to extract bboxes.

        Expected format:
        <detection>
        label: pothole
        bbox: [120, 340, 220, 440]
        confidence: 0.85
        </detection>
        """
        import re

        detections = []

        # Find all detection blocks
        detection_pattern = r'<detection>(.*?)</detection>'
        blocks = re.findall(detection_pattern, response, re.DOTALL | re.IGNORECASE)

        for block in blocks:
            try:
                # Extract label
                label_match = re.search(r'label\s*:\s*(\w+)', block, re.IGNORECASE)
                if not label_match:
                    continue
                label = label_match.group(1).lower().strip()

                if label == 'none':
                    continue

                # Extract bbox
                bbox_match = re.search(r'bbox\s*:\s*\[?\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]?', block)
                if not bbox_match:
                    continue

                x1, y1, x2, y2 = map(int, bbox_match.groups())

                # Validate bbox
                if x1 >= x2 or y1 >= y2:
                    continue
                if x1 < 0 or y1 < 0 or x2 > img_width or y2 > img_height:
                    # Clamp to image bounds
                    x1 = max(0, min(x1, img_width))
                    y1 = max(0, min(y1, img_height))
                    x2 = max(0, min(x2, img_width))
                    y2 = max(0, min(y2, img_height))

                # Extract confidence
                conf_match = re.search(r'confidence\s*:\s*([\d.]+)', block)
                confidence = float(conf_match.group(1)) if conf_match else 0.8

                # Validate confidence
                confidence = max(0.0, min(1.0, confidence))

                detections.append({
                    'label': label,
                    'bbox': [x1, y1, x2, y2],
                    'confidence': confidence
                })

            except Exception as e:
                logger.debug(f"Failed to parse detection block: {e}")
                continue

        return detections

    def _parse_keep_ids(self, response: str, masks: List) -> List[int]:
        """Parse mask IDs to keep from LLM response."""
        import re

        response_lower = response.lower()

        # Check for "none" - all are false positives
        if "keep: none" in response_lower or "keep:none" in response_lower:
            return []

        # Try to find KEEP: pattern
        keep_match = re.search(r'keep[:\s]+([0-9,\s]+)', response_lower)
        if keep_match:
            ids_str = keep_match.group(1)
            try:
                keep_ids = [int(x.strip()) for x in ids_str.split(",") if x.strip().isdigit()]
                # Validate IDs exist
                valid_ids = {m.mask_id for m in masks}
                return [id for id in keep_ids if id in valid_ids]
            except:
                pass

        # Try to find any numbers mentioned as valid
        numbers = re.findall(r'\b(\d+)\b', response)
        if numbers:
            valid_ids = {m.mask_id for m in masks}
            found_ids = [int(n) for n in numbers if int(n) in valid_ids]
            if found_ids:
                return found_ids

        # Default: keep all if can't parse
        return [m.mask_id for m in masks]

    def _run_agentic_loop(
        self,
        image: Image.Image,
        user_query: str,
        start_time: float
    ) -> AgentResult:
        """
        Run the smart agentic loop with LLM-driven detection.

        This is the TRUE agentic flow matching SAM3 agent:
        - LLM decides what to search for
        - LLM validates each mask (Accept/Reject)
        - Iterative refinement until done

        Args:
            image: PIL Image to analyze
            user_query: User query
            start_time: Start time for elapsed calculation

        Returns:
            AgentResult with detections
        """
        # Get system prompt
        system_prompt = get_system_prompt(self.config.categories)

        # Initialize message manager
        message_manager = MessageManager(system_prompt)
        message_manager.add_user_message_with_image(image, user_query)

        # Initialize tool executor
        tool_executor = ToolExecutor(self.sam3_processor, image)

        # Agentic loop
        turn = 0
        final_result = None

        while turn < self.config.max_turns:
            turn += 1
            logger.info(f"=== Turn {turn}/{self.config.max_turns} ===")

            # Prune messages if needed
            if turn > 1 and (turn % self.config.prune_after_turns == 0 or message_manager.should_prune()):
                logger.debug("Pruning message history...")
                message_manager.prune_messages()

            # Get LLM response
            response = self._call_llm(message_manager)

            if response is None:
                logger.error("LLM returned None response")
                continue

            # Add assistant response to history
            message_manager.add_assistant_message(response)

            # DEBUG: Log raw LLM response
            logger.info(f"LLM Response (first 500 chars): {response[:500] if response else 'EMPTY'}")

            # Parse tool call
            tool_name, parameters, thinking = parse_tool_call_flexible(response)

            logger.info(f"Parsed: tool={tool_name}, params={parameters}")

            if thinking:
                logger.debug(f"LLM thinking: {thinking[:200]}...")

            if tool_name is None:
                # No valid tool call found
                logger.warning("No tool call found in response")

                # Add error message and retry
                error_msg = format_error_message(
                    "No valid tool call found. Please use the <think>/<tool> format."
                )
                message_manager.add_tool_result_message(
                    "error",
                    error_msg
                )
                continue

            # Validate tool call
            is_valid, error = validate_tool_call(tool_name, parameters)
            if not is_valid:
                logger.warning(f"Invalid tool call: {error}")
                message_manager.add_tool_result_message(
                    "error",
                    format_error_message(error)
                )
                continue

            # Normalize parameters before execution
            parameters = normalize_parameters(tool_name, parameters)

            logger.info(f"Executing tool: {tool_name}")
            logger.info(f"Parameters: {parameters}")

            # Execute tool
            result = tool_executor.execute(tool_name, parameters)

            # Log for debugging
            self.debug_logger.log_turn(turn, response, tool_name, parameters, result)

            # Add tool result to conversation
            message_manager.add_tool_result_message(
                result.tool_name,
                result.message,
                result_image=result.image,
                result_data=result.data
            )

            # Check for verdict in LLM response (after examine_each_mask)
            # The LLM should provide Accept/Reject verdicts
            verdicts = parse_verdict(response)
            if verdicts:
                rejected_ids = get_rejected_mask_ids(response)
                if rejected_ids:
                    logger.info(f"LLM rejected masks: {rejected_ids}")
                    tool_executor.remove_masks(rejected_ids)

                accepted_ids = get_accepted_mask_ids(response)
                if accepted_ids:
                    logger.info(f"LLM accepted masks: {accepted_ids}")

            # Check if we should exit
            if result.should_exit:
                logger.info(f"Agent completed after {turn} turns")
                final_result = result
                break

            # Auto-exit: If we found masks and this is a segment_phrase call,
            # check if we should auto-complete
            if self.config.auto_exit_on_masks and tool_name == "segment_phrase":
                masks = tool_executor.get_all_masks()
                if len(masks) > 0 and turn >= 3:
                    # We have masks and have done at least 3 turns - auto complete
                    logger.info(f"Auto-exit: Found {len(masks)} masks after {turn} turns")

                    # Build final result from collected masks
                    detections = [
                        {
                            "mask_id": m.mask_id,
                            "category": m.category,
                            "severity": tool_executor._category_to_severity(m.category),
                            "confidence": m.score,
                            "bbox": m.bbox,
                            "mask": m.mask,
                        }
                        for m in masks
                    ]
                    final_image = tool_executor._render_masks(masks)

                    final_result = ToolResult(
                        success=True,
                        tool_name="auto_complete",
                        message=f"Auto-completed: Found {len(masks)} infrastructure issues",
                        data={"detections": detections, "num_detections": len(detections)},
                        image=final_image,
                        should_exit=True
                    )
                    break

        # Handle timeout (max turns reached)
        if final_result is None:
            logger.warning(f"Agent reached max turns ({self.config.max_turns})")

            # Try to compile any masks found so far
            masks = tool_executor.get_all_masks()
            if masks:
                detections = [
                    {
                        "mask_id": m.mask_id,
                        "category": m.category,  # Use stored category
                        "severity": tool_executor._category_to_severity(m.category),
                        "confidence": m.score,
                        "bbox": m.bbox,
                        "mask": m.mask,
                    }
                    for m in masks
                ]
                final_image = tool_executor._render_masks(masks)
            else:
                detections = []
                final_image = image

            return AgentResult(
                success=False,
                detections=detections,
                num_detections=len(detections),
                final_image=final_image,
                turns_taken=turn,
                message=f"Agent timeout after {turn} turns. Found {len(detections)} partial detections."
            )

        # Build final result
        elapsed = time.time() - start_time
        detections = final_result.data.get("detections", [])

        # Log final results
        self.debug_logger.log_final(detections, turn)

        logger.info(f"Detection complete: {len(detections)} issues found in {elapsed:.2f}s")

        return AgentResult(
            success=True,
            detections=detections,
            num_detections=len(detections),
            final_image=final_result.image,
            turns_taken=turn,
            message=f"Detection complete: {len(detections)} infrastructure issues found."
        )

    def _call_llm(self, message_manager: MessageManager) -> Optional[str]:
        """
        Call the LLM with current messages.

        Args:
            message_manager: Message manager with conversation history

        Returns:
            LLM response string or None on error
        """
        try:
            # Get messages in HuggingFace format
            messages = message_manager.get_messages_for_hf()

            # Build prompt from messages
            prompt = self._build_prompt(messages)

            # Get the latest image from messages
            image = self._extract_latest_image(messages)

            if image is None:
                logger.error("No image found in messages")
                return None

            # Call Qwen detector
            result = self.qwen_detector.detect(image, prompt)

            if result.get("success"):
                return result.get("text", "")
            else:
                logger.error(f"LLM call failed: {result.get('error')}")
                return None

        except Exception as e:
            logger.error(f"LLM call error: {e}", exc_info=True)
            return None

    def _build_prompt(self, messages: List[Dict]) -> str:
        """
        Build a text prompt from messages for the LLM.

        Args:
            messages: List of message dicts

        Returns:
            Combined prompt string
        """
        parts = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                parts.append(f"[System]\n{content}\n")
            elif role == "user":
                if isinstance(content, str):
                    parts.append(f"[User]\n{content}\n")
                elif isinstance(content, list):
                    text_parts = []
                    for item in content:
                        if item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                    if text_parts:
                        parts.append(f"[User]\n{' '.join(text_parts)}\n")
            elif role == "assistant":
                parts.append(f"[Assistant]\n{content}\n")

        # Add prompt for next response
        parts.append("[Assistant]\n")

        return "\n".join(parts)

    def _extract_latest_image(self, messages: List[Dict]) -> Optional[Image.Image]:
        """
        Extract the most recent image from messages.

        Args:
            messages: List of message dicts

        Returns:
            PIL Image or None
        """
        # Search from newest to oldest
        for msg in reversed(messages):
            content = msg.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if item.get("type") == "image":
                        return item.get("image")

        return None


def run_agent_inference(
    image_path: str,
    qwen_detector,
    sam3_processor,
    categories: Optional[List[str]] = None,
    debug: bool = False,
    output_dir: str = "output",
    exemplar_manager=None,
    exemplar_prompt_builder=None,
    exemplar_strategy: str = "visual_context",
    max_exemplars: int = 3
) -> AgentResult:
    """
    Convenience function to run agent inference on an image.

    Args:
        image_path: Path to image file
        qwen_detector: Loaded Qwen3VL detector
        sam3_processor: Loaded SAM3 processor
        categories: Categories to detect (default: all)
        debug: Enable debug logging
        output_dir: Output directory for debug files
        exemplar_manager: ExemplarManager for few-shot learning
        exemplar_prompt_builder: ExemplarPromptBuilder for multi-image prompts
        exemplar_strategy: Strategy for exemplar prompting
        max_exemplars: Maximum exemplars per category

    Returns:
        AgentResult with detections
    """
    # Load image
    image = Image.open(image_path).convert("RGB")

    # Create config
    config = AgentConfig(
        categories=categories,
        debug=debug,
        debug_dir=os.path.join(output_dir, "debug")
    )

    # Create and run agent
    agent = InfrastructureDetectionAgentCore(
        qwen_detector=qwen_detector,
        sam3_processor=sam3_processor,
        config=config,
        exemplar_manager=exemplar_manager,
        exemplar_prompt_builder=exemplar_prompt_builder,
        exemplar_strategy=exemplar_strategy,
        max_exemplars=max_exemplars
    )

    result = agent.run(image)

    # Save result image if available
    if result.final_image:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "detection_result.png")
        result.final_image.save(output_path)
        logger.info(f"Saved result image to {output_path}")

    return result
