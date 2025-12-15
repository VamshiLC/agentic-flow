"""
Agentic Infrastructure Detector - Qwen3-VL + SAM3

Uses agentic pipeline:
1. Qwen3-VL-4B-Instruct: Detects infrastructure issues with bounding boxes
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


# Professional color mapping (BGR format for OpenCV)
# Each category has a distinct, vibrant color for easy identification
DEFECT_COLORS = {
    "potholes": (0, 0, 255),                    # Bright Red
    "alligator_cracks": (0, 140, 255),          # Dark Orange
    "longitudinal_cracks": (0, 215, 255),       # Gold
    "transverse_cracks": (0, 255, 255),         # Yellow
    "road_surface_damage": (80, 127, 255),      # Coral
    "abandoned_vehicle": (255, 100, 0),         # Deep Blue
    "homeless_encampment": (255, 0, 180),       # Purple-Pink
    "homeless_person": (255, 105, 180),         # Hot Pink
    "manholes": (255, 255, 0),                  # Cyan
    "damaged_paint": (255, 0, 255),             # Magenta
    "damaged_crosswalks": (203, 192, 255),      # Pink
    "dumped_trash": (42, 105, 165),             # Brown
    "graffiti": (0, 255, 128),                  # Spring Green
    "street_signs": (0, 215, 255),              # Gold (distinct from yellow)
    "traffic_lights": (200, 255, 0),            # Lime Green
    "tyre_marks": (255, 200, 150)               # Light Blue
}


class UnifiedInfrastructureDetector:
    """
    Agentic Infrastructure Detector using Qwen3-VL + SAM3.

    This detector uses:
    - Qwen3-VL-4B-Instruct: Vision-language model for detection
    - SAM3: Segment Anything Model 3 for precise segmentation masks
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-4B-Instruct",
        categories: Optional[List[str]] = None,
        device: Optional[str] = None,
        use_quantization: bool = False,
        low_memory: bool = False
    ):
        """
        Initialize agentic detector with Qwen3-VL + SAM3.

        Args:
            model_name: Hugging Face model ID for Qwen3-VL
            categories: List of categories to detect (default: all 12 categories)
            device: Device to use ("cuda", "cpu", or None for auto-detect)
            use_quantization: Use 8-bit quantization (reduces memory by ~50%)
            low_memory: Enable low memory optimizations
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

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
            low_memory=low_memory
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
