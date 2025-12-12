"""
Category-wise Model Configuration

Defines which model(s) to use for each infrastructure category:
- "qwen": Use Qwen3-VL only (better for complex scenes)
- "sam3": Use SAM3 only (better for segmentation)
- "both": Use Qwen (detect) + SAM3 (segment)
"""

CATEGORY_MODEL_MAPPING = {
    # Road defects - Use both (Qwen detects, SAM3 segments)
    "potholes": "both",
    "alligator_cracks": "both",
    "longitudinal_cracks": "both",
    "transverse_cracks": "both",
    "road_surface_damage": "both",

    # Homeless/social issues - Use Qwen only (better at complex scenes)
    "abandoned_vehicle": "sam3",  # SAM3 better for vehicles
    "homeless_encampment": "qwen",  # Qwen better for complex encampments
    "homeless_person": "qwen",      # Qwen better for people

    # Infrastructure - Mixed
    "manholes": "both",
    "damaged_paint": "both",
    "damaged_crosswalks": "both",
    "dumped_trash": "qwen",    # Complex scenes
    "street_signs": "sam3",     # Clear objects
    "traffic_lights": "sam3",   # Clear objects
    "tyre_marks": "both"
}

# Category groups for batch processing
CATEGORY_GROUPS = {
    "road_defects": [
        "potholes",
        "alligator_cracks",
        "longitudinal_cracks",
        "transverse_cracks",
        "road_surface_damage"
    ],
    "social_issues": [
        "abandoned_vehicle",
        "homeless_encampment",
        "homeless_person"
    ],
    "infrastructure": [
        "manholes",
        "damaged_paint",
        "damaged_crosswalks",
        "dumped_trash",
        "street_signs",
        "traffic_lights",
        "tyre_marks"
    ]
}


def get_model_for_category(category: str) -> str:
    """
    Get the model to use for a specific category.

    Args:
        category: Infrastructure category name

    Returns:
        "qwen", "sam3", or "both"
    """
    return CATEGORY_MODEL_MAPPING.get(category, "qwen")


def get_categories_by_model(model: str) -> list:
    """
    Get all categories that use a specific model.

    Args:
        model: "qwen", "sam3", or "both"

    Returns:
        List of category names
    """
    return [cat for cat, mod in CATEGORY_MODEL_MAPPING.items() if mod == model]


def should_use_qwen(category: str) -> bool:
    """Check if Qwen should be used for this category."""
    model = get_model_for_category(category)
    return model in ["qwen", "both"]


def should_use_sam3(category: str) -> bool:
    """Check if SAM3 should be used for this category."""
    model = get_model_for_category(category)
    return model in ["sam3", "both"]
