"""
Agent module for infrastructure detection

This module implements the TRUE agentic pattern matching the official SAM3 agent:
https://github.com/facebookresearch/sam3/blob/main/sam3/agent/

Components:
- agent_core.py: Main agentic loop orchestrator
- tools.py: Tool definitions and executor (segment_phrase, examine_each_mask, etc.)
- message_manager.py: Conversation history management
- system_prompt.py: System prompt with tool definitions
- utils.py: Tool parsing and debug utilities
- detection_agent_hf.py: High-level API using Hugging Face
"""

# Core agent components
from .agent_core import (
    InfrastructureDetectionAgentCore,
    AgentConfig,
    AgentResult,
    run_agent_inference
)

# Tools
from .tools import (
    ToolExecutor,
    ToolResult,
    MaskData
)

# Message management
from .message_manager import MessageManager

# System prompt
from .system_prompt import (
    get_system_prompt,
    get_tool_definitions,
    get_categories,
    INFRASTRUCTURE_CATEGORIES,
    TOOL_DEFINITIONS
)

# Utilities
from .utils import (
    parse_tool_call,
    parse_tool_call_flexible,
    validate_tool_call,
    DebugLogger
)

# High-level API (main entry point)
from .detection_agent_hf import (
    InfrastructureDetectionAgentHF,
    InfrastructureDetectionAgent  # Backwards compatible alias
)

# Legacy imports (for backwards compatibility)
from .detection_agent import InfrastructureDetectionAgent as LegacyAgent

__all__ = [
    # Core
    "InfrastructureDetectionAgentCore",
    "AgentConfig",
    "AgentResult",
    "run_agent_inference",

    # Tools
    "ToolExecutor",
    "ToolResult",
    "MaskData",

    # Messages
    "MessageManager",

    # Prompts
    "get_system_prompt",
    "get_tool_definitions",
    "get_categories",
    "INFRASTRUCTURE_CATEGORIES",
    "TOOL_DEFINITIONS",

    # Utils
    "parse_tool_call",
    "parse_tool_call_flexible",
    "validate_tool_call",
    "DebugLogger",

    # High-level API
    "InfrastructureDetectionAgentHF",
    "InfrastructureDetectionAgent",

    # Legacy
    "LegacyAgent",
]
