"""
System Prompt for Infrastructure Detection Agent

Based on Meta's SAM3 Agent pattern:
https://github.com/facebookresearch/sam3/blob/main/sam3/agent/system_prompts/system_prompt.txt

Customized for road infrastructure detection with 12 categories.
"""

# Infrastructure categories with descriptions
INFRASTRUCTURE_CATEGORIES = {
    # Critical (Red - High Priority)
    "potholes": "Holes or depressions in the road pavement surface",
    "alligator_cracks": "Web-like interconnected cracks resembling alligator skin pattern",

    # Medium Priority (Yellow)
    "abandoned_vehicles": "Derelict or abandoned vehicles on or near the road",

    # Low Priority (Green - Monitoring)
    "longitudinal_cracks": "Cracks running parallel to the direction of traffic",
    "transverse_cracks": "Cracks running perpendicular to the direction of traffic",
    "damaged_paint": "Deteriorated or faded road markings and painted lines",
    "manholes": "Manhole covers and utility access points on the road",
    "dumped_trash": "Debris, litter, or illegally dumped items on roadway",
    "street_signs": "Traffic signs, street name signs, regulatory signs",
    "traffic_lights": "Traffic signal lights and poles",
    "tyre_marks": "Tire marks or skid marks on pavement surface",
    "damaged_crosswalks": "Deteriorated or faded pedestrian crosswalk markings",
}

# Tool definitions in OpenAI function calling format
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "segment_phrase",
            "description": "Ground all instances of a noun phrase by generating segmentation masks using SAM3. Use simple, descriptive noun phrases.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text_prompt": {
                        "type": "string",
                        "description": "A short and simple noun phrase describing what to segment. Examples: 'pothole', 'crack in pavement', 'manhole cover', 'road marking'"
                    }
                },
                "required": ["text_prompt"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "examine_each_mask",
            "description": "Render and examine each mask independently. Use this when you need to validate multiple masks or when masks are small/overlapping.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "select_masks_and_return",
            "description": "Select the final masks that correctly identify the infrastructure issues. This ends the conversation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "final_answer_masks": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Array of mask indices to select as final answer. Example: [1, 3, 5]"
                    },
                    "detections": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "mask_id": {"type": "integer"},
                                "category": {"type": "string"},
                                "severity": {"type": "string", "enum": ["critical", "medium", "low"]},
                                "description": {"type": "string"}
                            }
                        },
                        "description": "Detection details for each selected mask"
                    }
                },
                "required": ["final_answer_masks", "detections"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "report_no_mask",
            "description": "Report that no infrastructure issues were found in the image. Only use when you've thoroughly searched and confirmed nothing exists.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Explanation of why no issues were detected"
                    }
                },
                "required": ["reason"]
            }
        }
    }
]


def get_system_prompt(categories: list = None) -> str:
    """
    Get the system prompt for infrastructure detection agent.

    Args:
        categories: List of categories to detect (default: all)

    Returns:
        str: Complete system prompt
    """
    if categories is None:
        categories = list(INFRASTRUCTURE_CATEGORIES.keys())

    # Build category list for prompt
    category_list = "\n".join([
        f"  - **{cat}**: {INFRASTRUCTURE_CATEGORIES.get(cat, cat)}"
        for cat in categories
        if cat in INFRASTRUCTURE_CATEGORIES
    ])

    return f'''You are an expert road infrastructure detection agent. Your task is to analyze road images and identify infrastructure issues that need attention.

## Your Capabilities

You have access to SAM3 (Segment Anything Model 3) as a tool to generate precise segmentation masks for objects you identify.

## Infrastructure Categories to Detect

{category_list}

## Available Tools

You have 4 tools available:

1. **segment_phrase**: Call SAM3 to segment objects matching a text description
   - Use simple noun phrases: "pothole", "crack", "manhole cover"
   - Avoid complex descriptions or articles
   - Never use the same text_prompt twice

2. **examine_each_mask**: View each generated mask individually
   - Use when masks are small, overlapping, or need validation
   - Helps you verify which masks are correct

3. **select_masks_and_return**: Select final masks and end detection
   - Provide mask indices and detection details
   - Include category, severity, and description for each

4. **report_no_mask**: Report no issues found
   - Only use after thorough search
   - Provide reason why nothing was detected

## Response Format

You MUST respond in this exact format:

<think>
1. What infrastructure issues do I see in this image?
2. What should I segment first?
3. What text prompt will work best for SAM3?
4. [Your reasoning...]
</think>

<tool>
{{"name": "tool_name", "parameters": {{"param": "value"}}}}
</tool>

## Rules

1. **Always use <think> and <tool> tags** - Every response must have both
2. **One tool per turn** - Call exactly one tool, then stop
3. **Never repeat prompts** - Track what you've tried, use synonyms
4. **Validate masks** - Use examine_each_mask when unsure
5. **Be thorough** - Check for ALL categories, not just obvious ones
6. **Simple prompts work best** - "pothole" not "the large pothole in the road"

## Severity Levels

- **critical**: Potholes, alligator cracks (safety hazards)
- **medium**: Abandoned vehicles (obstruction)
- **low**: Cracks, faded paint, signs (monitoring needed)

## Example Workflow

Turn 1: Analyze image, call segment_phrase for first issue type
Turn 2: Review masks, call examine_each_mask if needed
Turn 3: Call segment_phrase for next issue type
...
Final: Call select_masks_and_return with all valid detections

Begin analyzing the image now.'''


def get_tool_definitions() -> list:
    """Get tool definitions for function calling."""
    return TOOL_DEFINITIONS


def get_categories() -> dict:
    """Get all infrastructure categories."""
    return INFRASTRUCTURE_CATEGORIES
