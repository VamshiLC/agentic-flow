"""
Unified Infrastructure Detector - Combines Qwen3-VL + SAM3

Supports two modes:
1. Direct Mode: Qwen3-VL direct loading (simpler, no server needed)
2. Agent Mode: vLLM server + SAM3 Agent (faster for batch processing)
"""
import re
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional, Union

from models.qwen_direct_loader import Qwen3VLDirectDetector
from config import Config
from prompts.category_prompts import build_detailed_prompt


# Infrastructure defect categories (combines both your lists)
INFRASTRUCTURE_CATEGORIES = {
    # Road defects
    "potholes": "Severe holes or depressions in the road surface",
    "alligator_cracks": "Web-like interconnected cracks resembling alligator skin",
    "longitudinal_cracks": "Cracks running parallel to the direction of traffic",
    "transverse_cracks": "Cracks running perpendicular to the direction of traffic",
    "road_surface_damage": "General pavement deterioration or distress",

    # Homeless/social issues
    "abandoned_vehicle": "Vehicle clearly abandoned or with someone living inside",
    "homeless_encampment": "Tents, tarps, or makeshift shelters in public areas",
    "homeless_person": "Person living on streets with belongings",

    # Infrastructure
    "manholes": "Manhole covers and utility access points",
    "damaged_paint": "Deteriorated or faded road markings",
    "damaged_crosswalks": "Faded or deteriorated crosswalk markings",
    "dumped_trash": "Debris or illegally dumped items",
    "street_signs": "Traffic or regulatory signs",
    "traffic_lights": "Traffic signal lights and poles",
    "tyre_marks": "Tire or skid marks on pavement"
}


# Color mapping (BGR for OpenCV)
DEFECT_COLORS = {
    "potholes": (0, 0, 255),              # Red
    "alligator_cracks": (0, 200, 255),    # Orange
    "longitudinal_cracks": (0, 255, 0),   # Green
    "transverse_cracks": (255, 0, 0),     # Blue
    "road_surface_damage": (255, 255, 0), # Cyan
    "abandoned_vehicle": (0, 0, 200),     # Dark red
    "homeless_encampment": (0, 165, 255), # Orange
    "homeless_person": (255, 0, 255),     # Magenta
    "manholes": (128, 128, 128),          # Gray
    "damaged_paint": (128, 0, 128),       # Purple
    "damaged_crosswalks": (255, 0, 255),  # Magenta
    "dumped_trash": (100, 100, 100),      # Dark gray
    "street_signs": (255, 255, 0),        # Yellow
    "traffic_lights": (0, 255, 255),      # Yellow
    "tyre_marks": (200, 200, 200)         # Light gray
}


class UnifiedInfrastructureDetector:
    """
    Unified detector supporting both direct Qwen3-VL and agent mode.

    Combines:
    - Your detector.py pattern (direct loading, detailed prompts)
    - My agent pattern (vLLM server, SAM3 agent)
    """

    def __init__(
        self,
        mode: str = "direct",
        model_name: str = "Qwen/Qwen3-VL-4B-Instruct",
        categories: Optional[List[str]] = None,
        device: Optional[str] = None,
        use_quantization: bool = False,
        low_memory: bool = False
    ):
        """
        Initialize unified detector.

        Args:
            mode: "direct" (Hugging Face) or "agent" (vLLM server)
            model_name: Hugging Face model ID
            categories: List of categories to detect (default: all)
            device: Device to use ("cuda", "cpu", or None for auto-detect)
            use_quantization: Use 8-bit quantization (reduces memory by ~50%)
            low_memory: Enable low memory optimizations
        """
        self.mode = mode
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Set categories
        if categories is None:
            self.categories = list(INFRASTRUCTURE_CATEGORIES.keys())
        else:
            self.categories = categories

        print(f"Initializing Unified Infrastructure Detector")
        print(f"  Mode: {mode}")
        print(f"  Model: {model_name}")
        print(f"  Categories: {len(self.categories)}")
        print(f"  Device: {self.device}")

        # Load model based on mode
        if mode == "direct":
            self.detector = Qwen3VLDirectDetector(
                model_name,
                device,
                use_quantization=use_quantization,
                low_memory=low_memory
            )
        elif mode == "agent-hf":
            # Agentic pattern with Hugging Face (no vLLM server required)
            from agent.detection_agent_hf import InfrastructureDetectionAgentHF

            self.detector = InfrastructureDetectionAgentHF(
                model_name=model_name,
                sam3_processor=None,  # Will load internally
                categories=self.categories,
                device=device,
                use_quantization=use_quantization,
                low_memory=low_memory
            )
        elif mode == "agent":
            # Use agent pattern (requires vLLM server)
            from agent.detection_agent import InfrastructureDetectionAgent
            from models.sam3_loader import load_sam3_model
            from models.qwen_loader import get_qwen_config

            sam3_processor = load_sam3_model()
            llm_config = get_qwen_config(model=model_name)
            self.detector = InfrastructureDetectionAgent(sam3_processor, llm_config)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'direct', 'agent-hf', or 'agent'")

    def _build_detection_prompt(self, categories: Optional[List[str]] = None) -> str:
        """
        Build detailed detection prompt with visual cues and negative examples.

        Uses comprehensive prompts from prompts/category_prompts.py
        """
        if categories is None:
            categories = self.categories

        # Use detailed prompts for better accuracy
        return build_detailed_prompt(categories)

    def detect_infrastructure(
        self,
        image: Union[Image.Image, np.ndarray],
        categories: Optional[List[str]] = None,
        confidence_threshold: float = 0.3
    ) -> Dict:
        """
        Detect infrastructure issues in an image.

        Args:
            image: PIL Image or numpy array
            categories: Categories to detect (default: all)
            confidence_threshold: Minimum confidence for detections

        Returns:
            dict: {
                "detections": [...],
                "text_response": "...",
                "num_detections": int
            }
        """
        # Convert numpy to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        if categories is None:
            categories = self.categories

        if self.mode == "direct":
            # Build prompt
            prompt = self._build_detection_prompt(categories)

            # Run detection
            result = self.detector.detect(image, prompt)
            text_response = result["text"]

            # Parse detections
            detections = self._parse_detections(
                text_response,
                image.size,
                confidence_threshold
            )

            return {
                "detections": detections,
                "text_response": text_response,
                "num_detections": len(detections)
            }

        elif self.mode == "agent":
            # Use agent mode (autonomous detection)
            result = self.detector.process_frame(image)

            # Parse result (format depends on agent implementation)
            # TODO: Adapt based on actual agent output format

            return {
                "detections": [],  # Parse from agent result
                "text_response": str(result),
                "num_detections": 0
            }

    def _parse_detections(
        self,
        response: str,
        image_size: Tuple[int, int],
        confidence_threshold: float = 0.3
    ) -> List[Dict]:
        """
        Parse detection results from text response.

        Based on your detector.py parsing logic.

        Args:
            response: Text response from model
            image_size: (width, height) of image
            confidence_threshold: Minimum confidence

        Returns:
            list: List of detection dicts
        """
        detections = []
        width, height = image_size

        if "no defects detected" in response.lower():
            return detections

        # Pattern: Defect: <type>, Box: [x1, y1, x2, y2]
        pattern = r'Defect:\s*([^,\n]+),\s*Box:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
        matches = re.findall(pattern, response, re.IGNORECASE)

        for match in matches:
            try:
                label = self._normalize_label(match[0].strip().lower())

                # Convert normalized 0-1000 coords to pixel coords
                x1 = int(float(match[1]) * width / 1000)
                y1 = int(float(match[2]) * height / 1000)
                x2 = int(float(match[3]) * width / 1000)
                y2 = int(float(match[4]) * height / 1000)

                # Clamp to image bounds
                x1 = max(0, min(x1, width))
                y1 = max(0, min(y1, height))
                x2 = max(0, min(x2, width))
                y2 = max(0, min(y2, height))

                # Validate bbox
                if x2 > x1 and y2 > y1:
                    color = DEFECT_COLORS.get(label, (0, 255, 0))

                    detection = {
                        "label": label,
                        "category": label,
                        "bbox": [x1, y1, x2, y2],
                        "confidence": 0.8,  # Default confidence (model doesn't output this)
                        "color": color
                    }

                    detections.append(detection)

            except (ValueError, IndexError) as e:
                print(f"Warning: Failed to parse detection: {match} ({e})")
                continue

        return detections

    def _normalize_label(self, label: str) -> str:
        """
        Normalize detected label to match predefined categories.
        """
        label = label.lower().strip()

        # Direct match
        if label in INFRASTRUCTURE_CATEGORIES:
            return label

        # Keyword matching
        label_mappings = {
            "pothole": "potholes",
            "hole": "potholes",
            "alligator crack": "alligator_cracks",
            "alligator": "alligator_cracks",
            "longitudinal crack": "longitudinal_cracks",
            "longitudinal": "longitudinal_cracks",
            "transverse crack": "transverse_cracks",
            "transverse": "transverse_cracks",
            "vehicle": "abandoned_vehicle",
            "car": "abandoned_vehicle",
            "encampment": "homeless_encampment",
            "tent": "homeless_encampment",
            "homeless": "homeless_person",
            "person": "homeless_person",
            "manhole": "manholes",
            "paint": "damaged_paint",
            "crosswalk": "damaged_crosswalks",
            "trash": "dumped_trash",
            "sign": "street_signs",
            "light": "traffic_lights",
            "tyre": "tyre_marks",
            "tire": "tyre_marks"
        }

        for keyword, category in label_mappings.items():
            if keyword in label:
                return category

        # Return as-is if no match
        return label


def get_detector(
    mode: str = "direct",
    categories: Optional[List[str]] = None,
    use_quantization: bool = False,
    low_memory: bool = False
) -> UnifiedInfrastructureDetector:
    """
    Factory function to get detector.

    Args:
        mode: "direct" (Hugging Face) or "agent" (vLLM server)
        categories: Categories to detect
        use_quantization: Enable 8-bit quantization
        low_memory: Enable low memory optimizations

    Returns:
        UnifiedInfrastructureDetector instance
    """
    config = Config()

    return UnifiedInfrastructureDetector(
        mode=mode,
        model_name=config.QWEN_MODEL,
        categories=categories,
        use_quantization=use_quantization,
        low_memory=low_memory
    )


if __name__ == "__main__":
    # Test detector
    print("Testing Unified Infrastructure Detector...")

    # Create test image
    test_image = Image.new('RGB', (640, 480), color='gray')

    # Test direct mode
    detector = get_detector(mode="direct", categories=["potholes", "alligator_cracks"])

    result = detector.detect_infrastructure(test_image)
    print(f"\nDetections: {result['num_detections']}")
    print(f"Response: {result['text_response'][:200]}...")
