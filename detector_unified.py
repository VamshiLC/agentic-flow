"""
Agentic Infrastructure Detector - Qwen2.5-VL + SAM3

Uses agentic pipeline:
1. Qwen2.5-VL-7B-Instruct: Detects infrastructure issues with bounding boxes
2. SAM3 (Segment Anything Model 3): Generates segmentation masks for each detection
"""
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Optional, Union

from config import Config


# Infrastructure defect categories (combines both your lists)
INFRASTRUCTURE_CATEGORIES = {
    # Road defects
    "potholes": "Severe holes or depressions in the road surface",
    "alligator_cracks": "Web-like interconnected cracks resembling alligator skin",
    "longitudinal_cracks": "Cracks running parallel to the direction of traffic",
    "transverse_cracks": "Cracks running perpendicular to the direction of traffic",

    # Homeless/social issues
    "abandoned_vehicle": "Vehicle clearly abandoned or with someone living inside",
    "homeless_encampment": "Tents, tarps, or makeshift shelters in public areas",
    "homeless_person": "Person living on streets with belongings",

    # Infrastructure
    "manholes": "Manhole covers and utility access points",
    "damaged_paint": "Deteriorated or faded road markings",
    "damaged_crosswalks": "Faded or deteriorated crosswalk markings",
    "dumped_trash": "Debris or illegally dumped items",
    "graffiti": "Unauthorized spray paint or tags on public infrastructure",
    "street_signs": "Traffic or regulatory signs",
    "traffic_lights": "Traffic signal lights and poles",
    "tyre_marks": "Tire or skid marks on pavement"
}


# Modern color palette (BGR for OpenCV) - Grouped by severity
# Import from visualization_styles for consistency
try:
    from visualization_styles import MODERN_COLORS as DEFECT_COLORS
except ImportError:
    # Fallback colors if visualization_styles not available
    DEFECT_COLORS = {
        # CRITICAL - Red tones (requires immediate attention)
        "potholes": (40, 50, 240),                    # Bright red
        "alligator_cracks": (60, 90, 255),            # Orange-red

        # HIGH PRIORITY - Orange/Yellow tones
        "transverse_cracks": (0, 140, 255),           # Deep orange
        "longitudinal_cracks": (0, 200, 255),         # Orange
        "damaged_crosswalks": (0, 180, 240),          # Dark orange
        "damaged_paint": (0, 165, 255),               # Medium orange

        # SOCIAL ISSUES - Purple/Magenta tones
        "homeless_encampment": (180, 50, 200),        # Purple
        "homeless_person": (220, 80, 255),            # Magenta
        "abandoned_vehicle": (140, 0, 180),           # Dark purple
        "dumped_trash": (160, 60, 180),               # Purple-gray

        # INFRASTRUCTURE - Blue/Cyan tones
        "manholes": (200, 150, 50),                   # Steel blue
        "street_signs": (255, 200, 0),                # Bright cyan-blue
        "traffic_lights": (220, 180, 0),              # Deep cyan

        # MINOR - Green/Gray tones
        "tyre_marks": (100, 120, 100),                # Muted green-gray
        "graffiti": (180, 100, 220),                  # Pink-purple
    }


class UnifiedInfrastructureDetector:
    """
    Agentic Infrastructure Detector using Qwen2.5-VL + SAM3.

    This detector uses:
    - Qwen2.5-VL-7B-Instruct: Vision-language model for detection
    - SAM3: Segment Anything Model 3 for precise segmentation masks
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        categories: Optional[List[str]] = None,
        device: Optional[str] = None,
        use_quantization: bool = False,
        low_memory: bool = False,
        exclude_categories: Optional[List[str]] = None
    ):
        """
        Initialize agentic detector with Qwen3-VL + SAM3.

        Args:
            model_name: Hugging Face model ID for Qwen3-VL
            categories: List of categories to detect (default: all 15 categories)
            device: Device to use ("cuda", "cpu", or None for auto-detect)
            use_quantization: Use 8-bit quantization (reduces memory by ~50%)
            low_memory: Enable low memory optimizations
            exclude_categories: Categories to completely exclude (e.g., ["graffiti", "tyre_marks"])
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.exclude_categories = exclude_categories or []

        # Set categories
        if categories is None:
            self.categories = list(INFRASTRUCTURE_CATEGORIES.keys())
        else:
            self.categories = categories

        print(f"Initializing Agentic Infrastructure Detector (Qwen3-VL + SAM3)")
        print(f"  Model: {model_name}")
        print(f"  Categories: {len(self.categories)}")
        print(f"  Device: {self.device}")

        # Load agentic detector (Qwen3-VL + SAM3)
        from agent.detection_agent_hf import InfrastructureDetectionAgentHF

        self.detector = InfrastructureDetectionAgentHF(
            model_name=model_name,
            sam3_processor=None,  # Will load internally
            categories=self.categories,
            device=device,
            use_quantization=use_quantization,
            low_memory=low_memory,
            exclude_categories=self.exclude_categories
        )

    def detect_infrastructure(
        self,
        image: Union[Image.Image, np.ndarray],
        categories: Optional[List[str]] = None,
        use_sam3: bool = True
    ) -> Dict:
        """
        Detect infrastructure issues in an image using Qwen3-VL + SAM3.

        Args:
            image: PIL Image or numpy array
            categories: Categories to detect (default: all)
            use_sam3: Whether to generate segmentation masks (default: True)

        Returns:
            dict: {
                "detections": [...],  # List of detections with bboxes and masks
                "num_detections": int,
                "has_masks": bool
            }
        """
        # Convert numpy to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Use agentic mode (Qwen3-VL + SAM3)
        result = self.detector.detect_infrastructure(image, use_sam3=use_sam3)
        return result



def get_detector(
    categories: Optional[List[str]] = None,
    use_quantization: bool = False,
    low_memory: bool = False
) -> UnifiedInfrastructureDetector:
    """
    Factory function to get agentic detector (Qwen3-VL + SAM3).

    Args:
        categories: Categories to detect (default: all)
        use_quantization: Enable 8-bit quantization
        low_memory: Enable low memory optimizations

    Returns:
        UnifiedInfrastructureDetector instance
    """
    config = Config()

    return UnifiedInfrastructureDetector(
        model_name=config.QWEN_MODEL,
        categories=categories,
        use_quantization=use_quantization,
        low_memory=low_memory
    )


if __name__ == "__main__":
    # Test agentic detector
    print("Testing Agentic Infrastructure Detector (Qwen3-VL + SAM3)...")

    # Create test image
    test_image = Image.new('RGB', (640, 480), color='gray')

    # Get detector
    detector = get_detector(categories=["potholes", "alligator_cracks"])

    result = detector.detect_infrastructure(test_image)
    print(f"\nDetections: {result['num_detections']}")
    print(f"Has masks: {result.get('has_masks', False)}")
