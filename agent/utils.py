"""
Utility functions for Infrastructure Detection Agent

Includes:
- Tool call parsing from LLM responses
- Response validation
- Debug helpers

Based on: https://github.com/facebookresearch/sam3/blob/main/sam3/agent/agent_core.py
"""
import re
import json
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def parse_tool_call(response: str) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[str]]:
    """
    Parse tool call from LLM response.

    Expected format:
    <think>
    reasoning...
    </think>

    <tool>
    {"name": "tool_name", "parameters": {...}}
    </tool>

    Args:
        response: Full LLM response text

    Returns:
        Tuple of (tool_name, parameters, thinking)
        Returns (None, None, None) if parsing fails
    """
    thinking = None
    tool_name = None
    parameters = None

    # Extract <think> content
    think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()

    # Extract <tool> content
    tool_match = re.search(r'<tool>(.*?)</tool>', response, re.DOTALL)
    if tool_match:
        tool_json = tool_match.group(1).strip()
        try:
            tool_data = json.loads(tool_json)
            tool_name = tool_data.get("name")
            parameters = tool_data.get("parameters", {})
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse tool JSON: {e}")
            logger.debug(f"Tool JSON content: {tool_json}")

            # Try to extract tool name even if JSON is malformed
            name_match = re.search(r'"name"\s*:\s*"([^"]+)"', tool_json)
            if name_match:
                tool_name = name_match.group(1)
                parameters = {}
                logger.info(f"Extracted tool name despite JSON error: {tool_name}")

    return tool_name, parameters, thinking


def parse_tool_call_flexible(response: str) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[str]]:
    """
    Flexibly parse tool calls from various response formats.

    Handles:
    - Standard <think>/<tool> format
    - JSON-only responses
    - Function call format
    - Markdown code blocks

    Args:
        response: LLM response text

    Returns:
        Tuple of (tool_name, parameters, thinking)
    """
    # Try standard format first
    tool_name, parameters, thinking = parse_tool_call(response)
    if tool_name:
        return tool_name, parameters, thinking

    # Try JSON code block
    json_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
    if json_block_match:
        try:
            data = json.loads(json_block_match.group(1))
            if "name" in data:
                return data["name"], data.get("parameters", {}), None
        except json.JSONDecodeError:
            pass

    # Try raw JSON in response
    json_match = re.search(r'\{[^{}]*"name"\s*:\s*"[^"]+"\s*[^{}]*\}', response)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            return data.get("name"), data.get("parameters", {}), None
        except json.JSONDecodeError:
            pass

    # Try to find tool name mentions
    tool_names = ["segment_phrase", "examine_each_mask", "select_masks_and_return", "report_no_mask"]
    for name in tool_names:
        if name in response.lower():
            # Try to extract parameters
            if name == "segment_phrase":
                prompt_match = re.search(r'text_prompt["\s:]+(["\'])(.+?)\1', response)
                if prompt_match:
                    return name, {"text_prompt": prompt_match.group(2)}, None
            elif name == "select_masks_and_return":
                masks_match = re.search(r'final_answer_masks["\s:]+\[([^\]]+)\]', response)
                if masks_match:
                    try:
                        mask_ids = [int(x.strip()) for x in masks_match.group(1).split(",")]
                        return name, {"final_answer_masks": mask_ids, "detections": []}, None
                    except ValueError:
                        pass
            elif name == "report_no_mask":
                return name, {"reason": "No issues detected"}, None
            elif name == "examine_each_mask":
                return name, {}, None

    return None, None, thinking


def validate_tool_call(tool_name: str, parameters: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate a parsed tool call.

    Args:
        tool_name: Name of the tool
        parameters: Tool parameters

    Returns:
        Tuple of (is_valid, error_message)
    """
    valid_tools = {
        "segment_phrase": ["text_prompt"],
        "examine_each_mask": [],
        "select_masks_and_return": ["final_answer_masks"],
        "report_no_mask": [],
    }

    if tool_name not in valid_tools:
        return False, f"Unknown tool: {tool_name}"

    required_params = valid_tools[tool_name]
    for param in required_params:
        if param not in parameters:
            return False, f"Missing required parameter: {param}"

    # Additional validation
    if tool_name == "segment_phrase":
        prompt = parameters.get("text_prompt", "")
        if not prompt or len(prompt.strip()) == 0:
            return False, "text_prompt cannot be empty"

    if tool_name == "select_masks_and_return":
        masks = parameters.get("final_answer_masks", [])
        if not isinstance(masks, list):
            return False, "final_answer_masks must be a list"
        if len(masks) == 0:
            return False, "final_answer_masks cannot be empty"
        for m in masks:
            if not isinstance(m, int) or m < 1 or m > 100:
                return False, f"Invalid mask ID: {m}. Must be integer 1-100"

    return True, ""


def format_error_message(error: str) -> str:
    """Format error message for LLM."""
    return f"[ERROR] {error}\nPlease try again with a valid tool call."


def extract_detections_from_response(response: str) -> list:
    """
    Try to extract detection information from LLM response text.

    Useful when LLM provides detection details in thinking but
    doesn't properly format the tool call.

    Args:
        response: LLM response text

    Returns:
        List of detection dicts
    """
    detections = []

    # Look for category mentions
    categories = [
        "pothole", "alligator_crack", "longitudinal_crack", "transverse_crack",
        "abandoned_vehicle", "manhole", "damaged_paint", "damaged_crosswalk",
        "dumped_trash", "street_sign", "traffic_light", "tyre_mark"
    ]

    for cat in categories:
        if cat.replace("_", " ") in response.lower() or cat in response.lower():
            # Check if it's mentioned as detected
            patterns = [
                rf'{cat}[s]?\s+(?:detected|found|visible|present)',
                rf'(?:detected|found|see|visible)\s+.*{cat}',
                rf'mask\s+\d+.*{cat}',
            ]
            for pattern in patterns:
                if re.search(pattern, response, re.IGNORECASE):
                    detections.append({
                        "category": cat.replace(" ", "_"),
                        "mentioned": True
                    })
                    break

    return detections


def build_retry_prompt(error: str, attempt: int, max_attempts: int) -> str:
    """
    Build prompt for retry after error.

    Args:
        error: Error message
        attempt: Current attempt number
        max_attempts: Maximum attempts allowed

    Returns:
        Retry prompt string
    """
    return (
        f"[Attempt {attempt}/{max_attempts}]\n"
        f"Previous attempt failed: {error}\n\n"
        f"Please provide a valid response using the correct format:\n"
        f"<think>\n"
        f"[Your reasoning]\n"
        f"</think>\n\n"
        f"<tool>\n"
        f'{{"name": "tool_name", "parameters": {{}}}}\n'
        f"</tool>"
    )


class DebugLogger:
    """Helper for structured debug logging."""

    def __init__(self, enabled: bool = True, output_dir: str = "debug"):
        self.enabled = enabled
        self.output_dir = output_dir
        self.turn_count = 0

    def log_turn(
        self,
        turn: int,
        response: str,
        tool_name: str,
        parameters: Dict,
        result: Any
    ):
        """Log a single turn of the agentic loop."""
        if not self.enabled:
            return

        import os
        os.makedirs(self.output_dir, exist_ok=True)

        log_entry = {
            "turn": turn,
            "response_preview": response[:500] if response else None,
            "tool_name": tool_name,
            "parameters": parameters,
            "result_success": getattr(result, 'success', None),
            "result_message": getattr(result, 'message', None),
        }

        filepath = os.path.join(self.output_dir, f"turn_{turn:03d}.json")
        with open(filepath, 'w') as f:
            json.dump(log_entry, f, indent=2, default=str)

    def log_final(self, detections: list, total_turns: int):
        """Log final results."""
        if not self.enabled:
            return

        import os
        os.makedirs(self.output_dir, exist_ok=True)

        summary = {
            "total_turns": total_turns,
            "num_detections": len(detections),
            "categories": list(set(d.get("category") for d in detections if d.get("category"))),
        }

        filepath = os.path.join(self.output_dir, "summary.json")
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
