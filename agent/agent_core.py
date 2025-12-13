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
    DebugLogger
)

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for the agent."""
    max_turns: int = 25  # Enough turns to check all categories
    max_retries: int = 2  # Max retries per turn on parse errors
    categories: Optional[List[str]] = None  # Categories to detect
    debug: bool = False  # Enable debug logging
    debug_dir: str = "debug"  # Debug output directory
    prune_after_turns: int = 15  # Prune messages after N turns
    auto_exit_on_masks: bool = False  # Don't auto exit - let LLM decide
    force_all_categories: bool = False  # Use smart LLM-driven search


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

    def _run_direct_category_search(self, image: Image.Image) -> AgentResult:
        """
        Directly search for ALL categories without relying on LLM decisions.

        This is faster and more reliable - we call SAM3 once for each category.
        """
        start_time = time.time()

        # Get categories to search
        if self.config.categories:
            categories = self.config.categories
        else:
            categories = [
                "pothole", "alligator crack", "longitudinal crack", "transverse crack",
                "road damage", "abandoned vehicle", "homeless encampment", "tent",
                "manhole", "damaged road marking", "crosswalk", "trash", "debris",
                "street sign", "traffic light", "tyre mark", "skid mark"
            ]

        logger.info(f"Direct search for {len(categories)} categories")

        # Initialize tool executor
        tool_executor = ToolExecutor(self.sam3_processor, image)

        # Search each category
        for i, category in enumerate(categories):
            logger.info(f"[{i+1}/{len(categories)}] Searching for: {category}")

            try:
                result = tool_executor.execute("segment_phrase", {"text_prompt": category})
                if result.success and result.data.get("num_masks", 0) > 0:
                    logger.info(f"  Found {result.data['num_masks']} mask(s) for '{category}'")
                else:
                    logger.info(f"  No masks found for '{category}'")
            except Exception as e:
                logger.warning(f"  Error searching '{category}': {e}")

        # Compile all masks found
        masks = tool_executor.get_all_masks()

        if masks:
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
        else:
            detections = []
            final_image = image

        elapsed = time.time() - start_time
        logger.info(f"Direct search complete: {len(detections)} detections in {elapsed:.2f}s")

        return AgentResult(
            success=True,
            detections=detections,
            num_detections=len(detections),
            final_image=final_image,
            turns_taken=len(categories),
            message=f"Found {len(detections)} infrastructure issues across {len(categories)} categories"
        )

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
