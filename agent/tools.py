"""
Tool definitions for SAM3 Agent
"""
from typing import Dict, Any, List, Callable
from dataclasses import dataclass


@dataclass
class ToolParameter:
    """Definition of a tool parameter"""
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None


@dataclass
class ToolDefinition:
    """Definition of a tool that the agent can use"""
    name: str
    description: str
    parameters: List[ToolParameter]
    function: Callable


# SAM3 Segmentation Tool
SAM3_SEGMENT_TOOL = ToolDefinition(
    name="sam3_segment",
    description="Segment objects in an image based on a natural language query. "
                "This tool uses the SAM3 model to identify and create segmentation masks "
                "for objects described in the query.",
    parameters=[
        ToolParameter(
            name="image_path",
            type="string",
            description="Path to the image file to segment",
            required=True
        ),
        ToolParameter(
            name="query",
            type="string",
            description="Natural language description of what to segment. "
                       "Examples: 'the leftmost child wearing blue vest', "
                       "'all cars in the image', 'the red apple on the table'",
            required=True
        ),
        ToolParameter(
            name="confidence_threshold",
            type="float",
            description="Minimum confidence score for masks (0.0 to 1.0)",
            required=False,
            default=0.5
        )
    ],
    function=None  # Will be set during initialization
)


def get_tool_schema(tool: ToolDefinition) -> Dict[str, Any]:
    """
    Convert tool definition to OpenAI-compatible function schema
    """
    parameters_schema = {
        "type": "object",
        "properties": {},
        "required": []
    }

    for param in tool.parameters:
        param_schema = {
            "type": param.type,
            "description": param.description
        }
        if param.default is not None:
            param_schema["default"] = param.default

        parameters_schema["properties"][param.name] = param_schema

        if param.required:
            parameters_schema["required"].append(param.name)

    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": parameters_schema
        }
    }


def get_all_tools() -> List[ToolDefinition]:
    """Get all available tools"""
    return [SAM3_SEGMENT_TOOL]


def get_tools_schema() -> List[Dict[str, Any]]:
    """Get OpenAI-compatible schema for all tools"""
    return [get_tool_schema(tool) for tool in get_all_tools()]


class ToolRegistry:
    """Registry for managing and executing tools"""

    def __init__(self):
        self.tools: Dict[str, ToolDefinition] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        """Register default tools"""
        for tool in get_all_tools():
            self.register_tool(tool)

    def register_tool(self, tool: ToolDefinition):
        """Register a new tool"""
        self.tools[tool.name] = tool

    def get_tool(self, name: str) -> ToolDefinition:
        """Get a tool by name"""
        if name not in self.tools:
            raise ValueError(f"Tool '{name}' not found in registry")
        return self.tools[name]

    def execute_tool(self, name: str, **kwargs) -> Any:
        """Execute a tool with given parameters"""
        tool = self.get_tool(name)
        if tool.function is None:
            raise ValueError(f"Tool '{name}' has no function implementation")
        return tool.function(**kwargs)

    def get_schemas(self) -> List[Dict[str, Any]]:
        """Get schemas for all registered tools"""
        return [get_tool_schema(tool) for tool in self.tools.values()]
