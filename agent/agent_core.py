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
    """

    def __init__(
        self,
        qwen_detector,
        sam3_processor,
        config: Optional[AgentConfig] = None
    ):
        """
        Initialize the agent core.

        Args:
            qwen_detector: Loaded Qwen3VLDirectDetector instance
            sam3_processor: Loaded SAM3 processor
            config: Agent configuration
        """
        self.qwen_detector = qwen_detector
        self.sam3_processor = sam3_processor
        self.config = config or AgentConfig()

        # Debug logger
        self.debug_logger = DebugLogger(
            enabled=self.config.debug,
            output_dir=self.config.debug_dir
        )

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
        SIMPLE approach:
        1. Qwen looks at image and says what infrastructure issues it sees (plain text)
        2. SAM3 segments those things using text prompts

        NO JSON, NO BOUNDING BOXES - just simple text.
        """
        start_time = time.time()

        # Initialize tool executor
        tool_executor = ToolExecutor(self.sam3_processor, image)

        # If user specified categories, skip Qwen and search directly
        if self.config.categories:
            return self._run_text_prompt_search(image, self.config.categories, tool_executor, start_time)

        print(f"\n{'='*60}")
        print(f"SMART DETECTION MODE")
        print(f"Step 1: Asking Qwen what it sees...")
        print(f"{'='*60}")

        # Ask Qwen what infrastructure issues it sees (simple text response)
        found_items = self._ask_qwen_what_it_sees(image)

        if not found_items:
            print("Qwen didn't identify any specific issues.")
            print("Falling back to searching ALL categories...")
            # Fallback: search ALL infrastructure categories
            from .system_prompt import get_categories
            found_items = list(get_categories().keys())

        # MEMORY OPTIMIZATION: Clear Qwen from GPU before SAM3 segmentation
        if self.config.optimize_memory:
            self._optimize_memory_before_sam3()

        print(f"\n{'='*60}")
        print(f"Step 2: SAM3 searching for: {found_items}")
        print(f"{'='*60}")

        # SAM3 segments using text prompts
        all_masks = []
        for item in found_items:
            print(f"  Searching '{item}'...", end=" ", flush=True)
            try:
                result = tool_executor.execute("segment_phrase", {"text_prompt": item})
                if result.success and result.data.get("num_masks", 0) > 0:
                    num_found = result.data['num_masks']
                    print(f"✓ {num_found} found")
                else:
                    print("✗ none")
            except Exception as e:
                print(f"✗ error: {e}")

        # Get all accumulated masks
        masks = tool_executor.get_current_masks()

        # CONFIDENCE FILTERING - Remove low confidence detections
        if masks:
            before_count = len(masks)
            masks = [m for m in masks if m.score >= self.config.confidence_threshold]
            filtered = before_count - len(masks)
            if filtered > 0:
                print(f"  Filtered {filtered} low-confidence detections (threshold: {self.config.confidence_threshold})")
            print(f"  Keeping {len(masks)} high-confidence detections")

        # LLM VALIDATION - Filter out false positives (manholes as potholes, shadows, etc)
        if masks and self.config.validate_with_llm:
            print(f"\n{'='*60}")
            print(f"Step 3: LLM validating {len(masks)} detections...")
            print(f"{'='*60}")
            # Reload Qwen to GPU for validation
            if self.config.optimize_memory:
                self._reload_qwen_to_gpu()
            masks = self._validate_masks_with_llm(image, masks, tool_executor)
            print(f"After validation: {len(masks)} masks kept")

        # Build final detections
        if masks:
            print(f"\n{'='*60}")
            print(f"Step 4: Rendering {len(masks)} masks on image...")
            print(f"{'='*60}")

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
            print(f"✓ Detection complete!")

            # Summary
            categories_found = list(set(m.category for m in masks))
            print(f"\n{'='*60}")
            print(f"RESULTS: {len(masks)} objects detected")
            print(f"Categories: {categories_found}")
            print(f"{'='*60}")
        else:
            print("\n⚠ No objects found")
            detections = []
            final_image = image

        elapsed = time.time() - start_time
        logger.info(f"Detection complete: {len(detections)} issues in {elapsed:.2f}s")

        return AgentResult(
            success=True,
            detections=detections,
            num_detections=len(detections),
            final_image=final_image,
            turns_taken=1,
            message=f"Found {len(detections)} infrastructure issues"
        )

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
                # Homeless / Encampments
                'homeless': 'homeless encampment',
                'tent': 'homeless encampment',
                'encampment': 'homeless encampment',
                'sleeping bag': 'homeless encampment',
                'belongings': 'homeless encampment',
                'camp': 'homeless encampment',
                # Abandoned vehicles
                'abandoned': 'abandoned vehicle',
                'abandoned vehicle': 'abandoned vehicle',
                'derelict': 'abandoned vehicle',
                'broken down car': 'abandoned vehicle',
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

    def _parse_json_detection_response(self, response: str, img_width: int, img_height: int) -> List[Dict]:
        """
        Parse Qwen2.5-VL's native JSON grounding response.

        Expected format:
        [{"bbox_2d": [x1, y1, x2, y2], "label": "pothole", "confidence": 0.85}, ...]
        """
        import json
        import re

        detections = []
        json_str = None

        print(f"  Raw response length: {len(response)}")
        print(f"  Raw response: {response[:500]}...")

        try:
            # Method 1: Try to find JSON in ```json code block
            if '```json' in response:
                start_marker = response.find('```json')
                if start_marker != -1:
                    start_marker += 7  # Skip past ```json
                    end_marker = response.find('```', start_marker)
                    if end_marker != -1:
                        json_str = response[start_marker:end_marker].strip()
                        print(f"  Found code block JSON: {json_str[:100]}...")

            # Method 2: Use bracket counting to find complete [...] array
            if not json_str:
                start_idx = response.find('[')
                if start_idx != -1:
                    bracket_count = 0
                    end_idx = start_idx
                    for i, char in enumerate(response[start_idx:], start_idx):
                        if char == '[':
                            bracket_count += 1
                        elif char == ']':
                            bracket_count -= 1
                            if bracket_count == 0:
                                end_idx = i + 1
                                break
                    json_str = response[start_idx:end_idx]
                    print(f"  Found bracket-counted JSON: {json_str[:100]}...")

            # Method 3: Try the whole response as JSON
            if not json_str:
                json_str = response.strip()
                print(f"  Trying whole response as JSON")

            if json_str:
                print(f"  Parsing JSON ({len(json_str)} chars): {json_str[:200]}...")
                parsed = json.loads(json_str)
                print(f"  Parsed successfully! Type: {type(parsed)}, Length: {len(parsed) if isinstance(parsed, list) else 'N/A'}")

                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict):
                            # Handle bbox_2d format (Qwen2.5-VL native)
                            bbox = item.get('bbox_2d') or item.get('bbox') or item.get('box')
                            label = item.get('label', 'unknown')
                            confidence = item.get('confidence', 0.8)

                            if bbox and len(bbox) == 4:
                                x1, y1, x2, y2 = [int(float(b)) for b in bbox]

                                # Validate bbox
                                if x1 >= x2 or y1 >= y2:
                                    print(f"  Skipping invalid bbox: {bbox}")
                                    continue

                                # Clamp to image bounds
                                x1 = max(0, min(x1, img_width))
                                y1 = max(0, min(y1, img_height))
                                x2 = max(0, min(x2, img_width))
                                y2 = max(0, min(y2, img_height))

                                if x1 < x2 and y1 < y2:
                                    detections.append({
                                        'label': label.lower().strip(),
                                        'bbox': [x1, y1, x2, y2],
                                        'confidence': float(confidence)
                                    })
                                    print(f"  ✓ Parsed: {label} at [{x1}, {y1}, {x2}, {y2}]")

        except json.JSONDecodeError as e:
            print(f"  JSON parse error: {e}")
            print(f"  Attempted to parse: {json_str[:300] if json_str else 'None'}...")
            logger.debug(f"JSON parse failed: {e}")
        except Exception as e:
            print(f"  Parse error: {e}")
            import traceback
            traceback.print_exc()
            logger.debug(f"JSON detection parsing failed: {e}")

        print(f"  Total detections parsed: {len(detections)}")
        return detections

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
    output_dir: str = "output"
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
        config=config
    )

    result = agent.run(image)

    # Save result image if available
    if result.final_image:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "detection_result.png")
        result.final_image.save(output_path)
        logger.info(f"Saved result image to {output_path}")

    return result
