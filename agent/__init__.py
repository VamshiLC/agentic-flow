"""
Agent module for infrastructure detection
"""
from .detection_agent_hf import InfrastructureDetectionAgentHF
from .prompts import (
    SYSTEM_PROMPT,
    get_system_prompt,
    get_user_prompt,
    get_refinement_prompt,
    get_tool_result_prompt
)

__all__ = [
    "InfrastructureDetectionAgentHF",
    "SYSTEM_PROMPT",
    "get_system_prompt",
    "get_user_prompt",
    "get_refinement_prompt",
    "get_tool_result_prompt"
]
