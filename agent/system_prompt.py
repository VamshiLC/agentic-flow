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
    "spray paint on wall": "Spray paint, tags, or vandalism on walls and surfaces",
    "painted text on wall": "Graffiti text or writing on walls",

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

    return f'''You are an expert road infrastructure detection agent. Analyze road images and identify REAL infrastructure issues only.

## CRITICAL: Reject False Positives
- Shadows are NOT potholes
- Dark patches are NOT always damage
- You MUST validate each mask before accepting

## Categories to Detect

{category_list}

## Tools

1. **segment_phrase**: Search for objects
   - Use: {{"name": "segment_phrase", "parameters": {{"text_prompt": "pothole"}}}}

2. **examine_each_mask**: REQUIRED - Validate all masks
   - Use after EVERY segment_phrase call
   - Look at each mask and decide: Accept or Reject
   - Reject shadows, reflections, false positives

3. **select_masks_and_return**: Finish with accepted masks only
   - Only include masks you verified as REAL objects

## MANDATORY WORKFLOW

For each category:
1. Call segment_phrase
2. Call examine_each_mask (REQUIRED)
3. Provide verdict for each mask:
   <verdict>
   mask_1: Accept - real pothole with depth
   mask_2: Reject - this is just a shadow
   </verdict>
4. Move to next category

## Response Format

<think>
1. What am I searching for?
2. Looking at the masks - which are real, which are shadows/false?
</think>

<tool>
{{"name": "tool_name", "parameters": {{...}}}}
</tool>

## Validation Rules

**Accept if:**
- Clear physical damage (hole, crack, depression)
- Real object (sign, manhole cover, vehicle)
- Actual road marking damage

**Reject if:**
- Shadow that looks like a hole
- Wet spot or reflection
- Normal road texture
- Tree shadow on road
- Car shadow

## Search Order

1. pothole → examine_each_mask → verdict
2. crack → examine_each_mask → verdict
3. manhole → examine_each_mask → verdict
4. trash → examine_each_mask → verdict
5. sign → examine_each_mask → verdict

After all searches, call select_masks_and_return with ONLY accepted masks.

BE STRICT - It's better to miss a real defect than to mark shadows as damage.'''


def get_tool_definitions() -> list:
    """Get tool definitions for function calling."""
    return TOOL_DEFINITIONS


def get_categories() -> dict:
    """Get all infrastructure categories."""
    return INFRASTRUCTURE_CATEGORIES
