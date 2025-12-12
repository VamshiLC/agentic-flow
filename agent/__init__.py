"""
Agent module for infrastructure detection
"""
from .detection_agent import InfrastructureDetectionAgent
from .prompts import (
    SYSTEM_PROMPT,
    get_system_prompt,
    get_user_prompt,
    get_refinement_prompt,
    get_tool_result_prompt
)

__all__ = [
    "InfrastructureDetectionAgent",
    "SYSTEM_PROMPT",
    "get_system_prompt",
    "get_user_prompt",
    "get_refinement_prompt",
    "get_tool_result_prompt"
]
