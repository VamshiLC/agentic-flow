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
    validate_with_llm: bool = False  # Skip slow LLM validation
    confidence_threshold: float = 0.25  # Lower threshold to catch more


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
        Search ALL categories with SAM3, then optionally validate with LLM.

        This is the HYBRID approach:
        1. SAM3 searches ALL 17 categories (comprehensive)
        2. LLM validates each mask to reject false positives (smart)
        """
        start_time = time.time()

        # ALL categories to search - comprehensive list
        if self.config.categories:
            categories = self.config.categories
        else:
            categories = [
                # Critical defects
                "pothole", "alligator crack", "road crack", "pavement crack",
                # Surface damage
                "longitudinal crack", "transverse crack", "road damage",
                # Objects on road
                "abandoned vehicle", "abandoned car", "debris", "trash", "garbage",
                # Infrastructure
                "manhole", "manhole cover", "street sign", "traffic sign",
                "traffic light", "crosswalk", "road marking",
                # Encampments
                "tent", "homeless encampment",
                # Marks
                "tyre mark", "skid mark"
            ]

        print(f"\n{'='*60}")
        print(f"SEARCHING ALL {len(categories)} CATEGORIES WITH SAM3")
        print(f"{'='*60}")
        logger.info(f"=== SEARCHING ALL {len(categories)} CATEGORIES ===")

        # Initialize tool executor
        tool_executor = ToolExecutor(self.sam3_processor, image)

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

        # Filter by confidence threshold (faster than LLM validation)
        if masks:
            before_count = len(masks)
            masks = [m for m in masks if m.score >= self.config.confidence_threshold]
            filtered_count = before_count - len(masks)
            if filtered_count > 0:
                print(f"Filtered {filtered_count} low-confidence masks (threshold: {self.config.confidence_threshold})")
                logger.info(f"Filtered {filtered_count} low-confidence masks (threshold: {self.config.confidence_threshold})")
            print(f"Keeping {len(masks)} high-confidence masks for final output")
            logger.info(f"Keeping {len(masks)} high-confidence masks")

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
        Use LLM to validate masks and reject false positives (shadows, etc).

        Args:
            image: Original image
            masks: List of MaskData objects
            tool_executor: Tool executor for rendering

        Returns:
            Filtered list of validated masks
        """
        if not masks:
            return masks

        logger.info(f"Validating {len(masks)} masks with LLM...")

        # Render all masks for LLM to see
        rendered_image = tool_executor._render_masks(masks)

        # Build validation prompt
        mask_list = "\n".join([
            f"- Mask {m.mask_id}: {m.category} (confidence: {m.score:.2f})"
            for m in masks
        ])

        validation_prompt = f"""Look at this road image with detected masks.

DETECTED MASKS:
{mask_list}

For EACH mask, tell me if it's REAL or FALSE POSITIVE:
- REAL: Actual infrastructure issue (pothole, crack, sign, etc.)
- FALSE: Shadow, reflection, normal road texture, wet spot

Reply with ONLY the mask IDs that are REAL issues.
Format: KEEP: 1, 3, 5
Or if all are false: KEEP: none"""

        try:
            # Call LLM for validation
            result = self.qwen_detector.detect(rendered_image, validation_prompt)

            if result.get("success"):
                response = result.get("text", "")
                logger.info(f"LLM validation response: {response[:200]}")

                # Parse which masks to keep
                keep_ids = self._parse_keep_ids(response, masks)

                if keep_ids:
                    validated_masks = [m for m in masks if m.mask_id in keep_ids]
                    rejected_count = len(masks) - len(validated_masks)
                    logger.info(f"LLM rejected {rejected_count} false positives")
                    return validated_masks
                else:
                    # If parsing fails, keep all masks
                    logger.warning("Could not parse LLM validation, keeping all masks")
                    return masks
            else:
                logger.warning("LLM validation failed, keeping all masks")
                return masks

        except Exception as e:
            logger.error(f"LLM validation error: {e}")
            return masks

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
