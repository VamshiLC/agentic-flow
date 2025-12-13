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

    return f'''You are an expert road infrastructure detection agent. Your task is to analyze road images and identify ALL infrastructure issues.

## CRITICAL INSTRUCTION
You MUST search for ALL of these categories in order. Do NOT skip any category. Do NOT call select_masks_and_return until you have searched EVERY category.

## Infrastructure Categories to Detect (Search ALL of these)

{category_list}

## Available Tools

1. **segment_phrase**: Call SAM3 to segment objects
   - Use simple noun phrases: "pothole", "crack", "manhole"
   - Call this for EACH category you see in the image

2. **select_masks_and_return**: ONLY call this AFTER searching all categories
   - Provide all mask indices found

3. **report_no_mask**: ONLY if image has NO infrastructure at all

## Response Format

<think>
1. Which categories have I already searched?
2. Which categories still need to be searched?
3. What do I see in the image for the next category?
</think>

<tool>
{{"name": "segment_phrase", "parameters": {{"text_prompt": "category_name"}}}}
</tool>

## MANDATORY SEARCH ORDER

You MUST search in this order, one per turn:
1. pothole
2. crack (for all crack types)
3. manhole
4. road marking (for damaged paint)
5. crosswalk
6. trash
7. street sign
8. traffic light
9. vehicle (for abandoned vehicles)
10. tyre mark

After searching ALL categories, call select_masks_and_return with all masks found.

## Rules

1. **Search ALL categories** - Do not stop early
2. **One category per turn** - Call segment_phrase once, then wait
3. **Track progress** - Remember what you already searched
4. **Only finish when done** - Call select_masks_and_return only after all searches
5. **Use simple prompts** - "pothole" not "large pothole in road"
Final: Call select_masks_and_return with all valid detections

Begin analyzing the image now.'''


def get_tool_definitions() -> list:
    """Get tool definitions for function calling."""
    return TOOL_DEFINITIONS


def get_categories() -> dict:
    """Get all infrastructure categories."""
    return INFRASTRUCTURE_CATEGORIES
