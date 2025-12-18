"""
Exemplar Prompt Builder for Qwen3-VL

Builds multi-image prompts that incorporate exemplar images as visual context
for improved detection accuracy.

Strategies:
- visual_context: Show exemplar images first, then ask to find similar objects
- contrastive: Show positive AND negative exemplars for better discrimination
- description: Emphasize text descriptions with visual support
"""

import logging
from typing import Dict, Any, List, Optional
from PIL import Image

from .exemplar_manager import ExemplarManager, Exemplar

logger = logging.getLogger(__name__)


class ExemplarPromptBuilder:
    """
    Builds prompts for Qwen3-VL that incorporate exemplar images.

    Supports various prompting strategies to leverage few-shot learning
    effectively with vision-language models.

    Example:
        builder = ExemplarPromptBuilder(exemplar_manager)

        # Build detection prompt
        prompt_data = builder.build_detection_prompt(
            category="potholes",
            target_image=image,
            strategy="visual_context"
        )

        # Use with Qwen
        result = qwen.detect_with_exemplars(
            target_image=prompt_data["target_image"],
            exemplar_images=prompt_data["exemplar_images"],
            prompt=prompt_data["prompt"]
        )
    """

    # Category display names for natural language prompts
    CATEGORY_DISPLAY_NAMES = {
        "potholes": "potholes",
        "alligator_cracks": "alligator cracks (web-like crack patterns)",
        "longitudinal_cracks": "longitudinal cracks (cracks parallel to road)",
        "transverse_cracks": "transverse cracks (cracks perpendicular to road)",
        "manholes": "manhole covers",
        "damaged_paint": "damaged or faded road paint markings",
        "damaged_crosswalks": "damaged or faded crosswalk markings",
        "dumped_trash": "dumped trash or debris",
        "street_signs": "street signs",
        "traffic_lights": "traffic lights",
        "tyre_marks": "tire/tyre marks or skid marks",
        "abandoned_vehicles": "abandoned vehicles"
    }

    def __init__(self, exemplar_manager: ExemplarManager):
        """
        Initialize prompt builder.

        Args:
            exemplar_manager: ExemplarManager instance with loaded exemplars
        """
        self.exemplar_manager = exemplar_manager

    def build_detection_prompt(
        self,
        category: str,
        target_image: Image.Image,
        strategy: str = "visual_context",
        max_exemplars: int = 3
    ) -> Dict[str, Any]:
        """
        Build multi-image prompt for Qwen3-VL detection.

        Args:
            category: Category to detect (e.g., "potholes")
            target_image: Image to analyze
            strategy: Prompting strategy - "visual_context", "contrastive", or "description"
            max_exemplars: Maximum number of exemplar images to include

        Returns:
            {
                "messages": [...],           # HuggingFace message format
                "exemplar_images": [...],    # PIL Images (exemplars only)
                "target_image": Image,       # Target image
                "all_images": [...],         # All images in order
                "prompt": str,               # Text prompt
                "has_exemplars": bool
            }
        """
        if strategy == "visual_context":
            return self._visual_context_strategy(category, target_image, max_exemplars)
        elif strategy == "contrastive":
            return self._contrastive_strategy(category, target_image, max_exemplars)
        elif strategy == "description":
            return self._description_strategy(category, target_image, max_exemplars)
        else:
            logger.warning(f"Unknown strategy '{strategy}', using visual_context")
            return self._visual_context_strategy(category, target_image, max_exemplars)

    def _visual_context_strategy(
        self,
        category: str,
        target_image: Image.Image,
        max_exemplars: int
    ) -> Dict[str, Any]:
        """
        Strategy: Show exemplar images first, then ask to find similar objects.

        Format:
        [Exemplar 1] This is an example of a pothole.
        [Exemplar 2] This is another example of a pothole.
        [Target Image] Find all objects similar to the examples above.
        """
        # Get FULL exemplar images (not cropped) for better pattern matching
        # Cropped regions confuse the model because they look different from full images
        positive_exemplars = self.exemplar_manager.get_positive_exemplars(category, max_count=max_exemplars)

        exemplar_images = []
        descriptions = []
        for ex in positive_exemplars:
            if ex.image:  # Use FULL image, not cropped region
                exemplar_images.append(ex.image)
                descriptions.append(ex.description or f"Road image containing {category}")

        display_name = self.CATEGORY_DISPLAY_NAMES.get(category, category)

        # Build message content
        content = []

        # Add exemplar images with descriptions
        for i, (img, desc) in enumerate(zip(exemplar_images, descriptions), 1):
            content.append({"type": "image", "image": img})
            content.append({
                "type": "text",
                "text": f"[Example {i}] This is what a {display_name} looks like: {desc}"
            })

        # Add target image
        content.append({"type": "image", "image": target_image})

        # Build detection prompt - PIXEL COORDINATES, matching agent_core format
        if exemplar_images:
            prompt_text = f"""[Target Image] Now analyze this road image.

Find ALL instances of {display_name} that look similar to the examples shown above.

CRITICAL RULES FOR BOUNDING BOXES:
- Draw TIGHT bounding boxes around ONLY the defect
- Box should fit closely around the actual crack/pothole
- Do NOT draw huge boxes covering the whole road
- Each defect gets its own small, tight box

Output JSON array:
[{{"label": "{category}", "bbox_2d": [x1, y1, x2, y2]}}]

x1,y1 = top-left corner (pixels). x2,y2 = bottom-right corner (pixels).

If no {display_name} found, output: []"""
        else:
            # Fallback to text-only prompt
            prompt_text = f"""Analyze this road image and find ALL instances of {display_name}.

CRITICAL RULES FOR BOUNDING BOXES:
- Draw TIGHT bounding boxes around ONLY the defect
- Box should fit closely around the actual crack/pothole
- Do NOT draw huge boxes covering the whole road

Output JSON array:
[{{"label": "{category}", "bbox_2d": [x1, y1, x2, y2]}}]

x1,y1 = top-left corner (pixels). x2,y2 = bottom-right corner (pixels).

If no {display_name} found, output: []"""

        content.append({"type": "text", "text": prompt_text})

        messages = [{"role": "user", "content": content}]

        return {
            "messages": messages,
            "exemplar_images": exemplar_images,
            "target_image": target_image,
            "all_images": exemplar_images + [target_image],
            "prompt": prompt_text,
            "has_exemplars": len(exemplar_images) > 0,
            "strategy": "visual_context"
        }

    def _contrastive_strategy(
        self,
        category: str,
        target_image: Image.Image,
        max_exemplars: int
    ) -> Dict[str, Any]:
        """
        Strategy: Show positive AND negative exemplars for better discrimination.

        Format:
        [Positive examples] - "These ARE potholes"
        [Negative examples] - "These are NOT potholes (shadows, stains)"
        [Target] - "Find the real ones"
        """
        qwen_data = self.exemplar_manager.prepare_for_qwen(
            category=category,
            target_image=None,
            max_exemplars=max_exemplars,
            include_negative=True
        )

        exemplar_images = qwen_data["images"]
        pos_descriptions = qwen_data["positive_descriptions"]
        neg_descriptions = qwen_data["negative_descriptions"]
        display_name = self.CATEGORY_DISPLAY_NAMES.get(category, category)

        # Get negative exemplar images
        negative_exemplars = self.exemplar_manager.get_negative_exemplars(category)[:2]
        negative_images = [e.image for e in negative_exemplars if e.image]

        content = []

        # Add positive examples
        if exemplar_images:
            content.append({
                "type": "text",
                "text": f"=== POSITIVE EXAMPLES (These ARE {display_name}) ==="
            })
            for i, (img, desc) in enumerate(zip(exemplar_images, pos_descriptions), 1):
                content.append({"type": "image", "image": img})
                content.append({"type": "text", "text": f"[YES {i}] {desc}"})

        # Add negative examples
        if negative_images:
            content.append({
                "type": "text",
                "text": f"\n=== NEGATIVE EXAMPLES (These are NOT {display_name}) ==="
            })
            for i, (img, exemplar) in enumerate(zip(negative_images, negative_exemplars), 1):
                content.append({"type": "image", "image": img})
                reason = exemplar.confusion_reason or "False positive"
                content.append({"type": "text", "text": f"[NO {i}] {reason}"})

        # Add target image
        content.append({
            "type": "text",
            "text": f"\n=== TARGET IMAGE (Find the real {display_name}) ==="
        })
        content.append({"type": "image", "image": target_image})

        prompt_text = f"""Based on the positive and negative examples above, find ALL REAL {display_name} in this target image.

REMEMBER:
- Match the visual characteristics of the POSITIVE examples
- AVOID confusing it with things shown in the NEGATIVE examples
- Shadows, wet spots, and stains are NOT {display_name}

Respond in JSON format:
{{
    "detections": [
        {{"bbox": [x1, y1, x2, y2], "confidence": 0.9, "description": "...", "matches_positive": true}}
    ]
}}"""

        content.append({"type": "text", "text": prompt_text})

        messages = [{"role": "user", "content": content}]
        all_images = exemplar_images + negative_images + [target_image]

        return {
            "messages": messages,
            "exemplar_images": exemplar_images,
            "negative_images": negative_images,
            "target_image": target_image,
            "all_images": all_images,
            "prompt": prompt_text,
            "has_exemplars": len(exemplar_images) > 0,
            "strategy": "contrastive"
        }

    def _description_strategy(
        self,
        category: str,
        target_image: Image.Image,
        max_exemplars: int
    ) -> Dict[str, Any]:
        """
        Strategy: Emphasize detailed text descriptions with visual support.

        Uses rich text descriptions from category_prompts combined with
        visual exemplars for the best of both worlds.
        """
        qwen_data = self.exemplar_manager.prepare_for_qwen(
            category=category,
            target_image=None,
            max_exemplars=min(max_exemplars, 2),  # Fewer images, more text
            include_negative=False
        )

        exemplar_images = qwen_data["images"]
        display_name = self.CATEGORY_DISPLAY_NAMES.get(category, category)

        # Get detailed description from category config
        detailed_desc = self._get_detailed_description(category)

        content = []

        # Start with detailed text description
        content.append({
            "type": "text",
            "text": f"""=== DETECTION TARGET: {display_name.upper()} ===

{detailed_desc}

"""
        })

        # Add visual examples
        if exemplar_images:
            content.append({
                "type": "text",
                "text": "=== VISUAL REFERENCE EXAMPLES ==="
            })
            for i, img in enumerate(exemplar_images, 1):
                content.append({"type": "image", "image": img})
                content.append({"type": "text", "text": f"[Reference {i}]"})

        # Add target
        content.append({
            "type": "text",
            "text": "\n=== IMAGE TO ANALYZE ==="
        })
        content.append({"type": "image", "image": target_image})

        prompt_text = f"""Find all {display_name} matching the description and visual references above.

Output JSON format:
{{
    "detections": [
        {{"bbox": [x1, y1, x2, y2], "confidence": 0.9, "description": "matches visual cue X"}}
    ]
}}"""

        content.append({"type": "text", "text": prompt_text})

        messages = [{"role": "user", "content": content}]

        return {
            "messages": messages,
            "exemplar_images": exemplar_images,
            "target_image": target_image,
            "all_images": exemplar_images + [target_image],
            "prompt": prompt_text,
            "has_exemplars": len(exemplar_images) > 0,
            "strategy": "description"
        }

    def _get_detailed_description(self, category: str) -> str:
        """Get detailed description for a category."""
        # Try to import from category_prompts
        try:
            from prompts.category_prompts import DETAILED_PROMPTS
            if category in DETAILED_PROMPTS:
                prompt_data = DETAILED_PROMPTS[category]
                desc = f"What to look for: {prompt_data.get('what', '')}\n\n"
                desc += "Visual cues:\n"
                for cue in prompt_data.get('visual_cues', []):
                    desc += f"  - {cue}\n"
                desc += f"\nTypical location: {prompt_data.get('location', 'Road surface')}\n"
                desc += "\nNOT this (false positives):\n"
                for fp in prompt_data.get('not_this', [])[:3]:
                    desc += f"  - {fp}\n"
                return desc
        except ImportError:
            pass

        # Fallback descriptions
        fallback_descriptions = {
            "potholes": """What to look for: Holes or depressions in the road pavement

Visual cues:
  - Dark cavity or depression in road surface
  - Broken, jagged edges around the hole
  - May contain water, debris, or exposed base layer
  - Typically 6+ inches in diameter with visible depth

NOT this:
  - Shadows from trees or objects
  - Wet spots or puddles on flat surface
  - Dark patches without depth""",

            "alligator_cracks": """What to look for: Web-like interconnected cracks resembling alligator skin

Visual cues:
  - Multiple cracks forming a connected pattern
  - Resembles scales or chicken wire pattern
  - Usually covers an area, not just a line
  - Often in wheel paths

NOT this:
  - Single isolated cracks
  - Regular road texture
  - Expansion joints""",
        }

        return fallback_descriptions.get(
            category,
            f"Look for {self.CATEGORY_DISPLAY_NAMES.get(category, category)} on the road surface."
        )

    def build_validation_prompt(
        self,
        category: str,
        candidate_region: Image.Image,
        candidate_bbox: List[int]
    ) -> Dict[str, Any]:
        """
        Build prompt for validating a detection against exemplars.

        Used to double-check if a detected region really matches the category.

        Args:
            category: Category being validated
            candidate_region: Cropped image of the detection
            candidate_bbox: Original bbox coordinates

        Returns:
            Prompt data for validation
        """
        exemplars = self.exemplar_manager.get_positive_exemplars(category, max_count=2)
        display_name = self.CATEGORY_DISPLAY_NAMES.get(category, category)

        content = []

        # Show exemplars
        if exemplars:
            content.append({
                "type": "text",
                "text": f"=== REFERENCE: Known examples of {display_name} ==="
            })
            for i, ex in enumerate(exemplars, 1):
                img = ex.get_cropped_region() if ex.bbox else ex.image
                if img:
                    content.append({"type": "image", "image": img})
                    content.append({"type": "text", "text": f"[Reference {i}]"})

        # Show candidate
        content.append({
            "type": "text",
            "text": f"\n=== CANDIDATE DETECTION ==="
        })
        content.append({"type": "image", "image": candidate_region})

        prompt_text = f"""Is this candidate image a real {display_name}?

Compare it to the reference examples above.

Respond with JSON:
{{
    "is_valid": true/false,
    "confidence": 0.0-1.0,
    "reason": "explanation"
}}"""

        content.append({"type": "text", "text": prompt_text})

        messages = [{"role": "user", "content": content}]
        exemplar_images = [e.get_cropped_region() or e.image for e in exemplars if e.image]

        return {
            "messages": messages,
            "exemplar_images": exemplar_images,
            "candidate_image": candidate_region,
            "all_images": exemplar_images + [candidate_region],
            "prompt": prompt_text,
            "category": category,
            "bbox": candidate_bbox
        }

    def build_batch_detection_prompt(
        self,
        categories: List[str],
        target_image: Image.Image,
        max_exemplars_per_category: int = 2
    ) -> Dict[str, Any]:
        """
        Build prompt for detecting multiple categories at once.

        Args:
            categories: List of categories to detect
            target_image: Image to analyze
            max_exemplars_per_category: Max exemplars per category

        Returns:
            Multi-category detection prompt data
        """
        content = []
        all_exemplar_images = []

        content.append({
            "type": "text",
            "text": "=== MULTI-CATEGORY DETECTION ==="
        })

        # Add exemplars for each category
        for category in categories:
            exemplars = self.exemplar_manager.get_positive_exemplars(
                category, max_count=max_exemplars_per_category
            )

            if exemplars:
                display_name = self.CATEGORY_DISPLAY_NAMES.get(category, category)
                content.append({
                    "type": "text",
                    "text": f"\n--- {display_name.upper()} examples ---"
                })

                for ex in exemplars:
                    img = ex.get_cropped_region() if ex.bbox else ex.image
                    if img:
                        content.append({"type": "image", "image": img})
                        all_exemplar_images.append(img)

        # Add target
        content.append({
            "type": "text",
            "text": "\n=== TARGET IMAGE ==="
        })
        content.append({"type": "image", "image": target_image})

        # Build category list for prompt
        category_list = ", ".join([
            self.CATEGORY_DISPLAY_NAMES.get(c, c) for c in categories
        ])

        prompt_text = f"""Find all instances of the following in this image: {category_list}

Use the exemplar images above as reference for what each category looks like.

Respond in JSON format:
{{
    "detections": [
        {{"category": "potholes", "bbox": [x1, y1, x2, y2], "confidence": 0.9, "description": "..."}}
    ]
}}"""

        content.append({"type": "text", "text": prompt_text})

        messages = [{"role": "user", "content": content}]

        return {
            "messages": messages,
            "exemplar_images": all_exemplar_images,
            "target_image": target_image,
            "all_images": all_exemplar_images + [target_image],
            "prompt": prompt_text,
            "categories": categories,
            "has_exemplars": len(all_exemplar_images) > 0
        }
