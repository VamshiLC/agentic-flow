"""
Infrastructure Detection Agent using Qwen3-VL (Hugging Face) + SAM3

This implements the agentic flow where:
1. Qwen3-VL detects and describes infrastructure issues
2. SAM3 segments each detection based on Qwen's description

No vLLM server required - uses Hugging Face Transformers directly.
"""
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Optional, Union
import logging

from models.qwen_direct_loader import Qwen3VLDirectDetector
from models.sam3_loader import load_sam3_model

logger = logging.getLogger(__name__)


class InfrastructureDetectionAgentHF:
    """
    Agentic infrastructure detector using Qwen3-VL (HF) + SAM3.

    Architecture:
    - Qwen3-VL: Detects infrastructure issues and provides descriptions
    - SAM3: Segments each detection based on description (acts as a tool)

    This is the agentic pattern without vLLM server dependency.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-4B-Instruct",
        sam3_processor=None,
        categories: Optional[List[str]] = None,
        device: Optional[str] = None,
        use_quantization: bool = False,
        low_memory: bool = False,
        sam3_confidence: float = 0.5
    ):
        """
        Initialize the agentic detector.

        Args:
            model_name: Hugging Face model ID for Qwen3-VL
            sam3_processor: Pre-loaded SAM3 processor (or None to load)
            categories: List of infrastructure categories to detect
            device: Device to use ("cuda", "cpu", or None for auto)
            use_quantization: Use 8-bit quantization for Qwen
            low_memory: Enable memory optimizations
            sam3_confidence: Confidence threshold for SAM3 segmentation
        """
        self.model_name = model_name
        self.categories = categories
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"\nInitializing Agentic Detector (Qwen3-VL + SAM3)")
        print(f"  Model: {model_name}")
        print(f"  Device: {self.device}")
        print(f"  SAM3 confidence: {sam3_confidence}")

        # Load Qwen3-VL detector (the "brain")
        self.qwen_detector = Qwen3VLDirectDetector(
            model_name=model_name,
            device=device,
            use_quantization=use_quantization,
            low_memory=low_memory
        )

        # Load SAM3 processor (the "tool")
        if sam3_processor is None:
            print("\nLoading SAM3 segmentation tool...")
            self.sam3_processor = load_sam3_model(
                confidence_threshold=sam3_confidence,
                device=self.device
            )
        else:
            self.sam3_processor = sam3_processor
            print("Using pre-loaded SAM3 processor")

        print("✓ Agentic detector initialized!")

    def detect_infrastructure(
        self,
        image: Union[Image.Image, str],
        use_sam3: bool = True
    ) -> Dict:
        """
        Detect infrastructure issues with optional SAM3 segmentation.

        Agentic workflow:
        1. Qwen3-VL analyzes image → detections with descriptions
        2. For each detection → SAM3 segments based on description
        3. Return detections with bounding boxes + segmentation masks

        Args:
            image: PIL Image or path to image file
            use_sam3: If True, add SAM3 segmentation masks

        Returns:
            dict: {
                'detections': [...],
                'num_detections': int,
                'has_masks': bool
            }
        """
        # Step 1: Qwen detects infrastructure issues
        logger.debug("Step 1: Qwen3-VL detection...")
        qwen_result = self.qwen_detector.detect_infrastructure(image)

        # Handle None or invalid result
        if qwen_result is None:
            logger.error("Qwen detection returned None")
            return {
                'detections': [],
                'num_detections': 0,
                'has_masks': False
            }

        if not use_sam3 or qwen_result.get('num_detections', 0) == 0:
            # No SAM3 segmentation needed/possible
            return {
                **qwen_result,
                'has_masks': False
            }

        # Step 2: For each detection, get SAM3 segmentation mask
        logger.debug(f"Step 2: SAM3 segmentation for {qwen_result['num_detections']} detections...")

        enhanced_detections = []
        for det in qwen_result.get('detections', []):
            # Use Qwen's description to guide SAM3
            try:
                label = det.get('label', 'unknown')
                description = det.get('description', label)
                mask = self._segment_with_sam3(image, query=description)
                det['mask'] = mask
                det['has_mask'] = True
            except Exception as e:
                label = det.get('label', 'unknown')
                logger.warning(f"SAM3 segmentation failed for {label}: {e}")
                det['mask'] = None
                det['has_mask'] = False

            enhanced_detections.append(det)

        return {
            'detections': enhanced_detections,
            'num_detections': len(enhanced_detections),
            'has_masks': True
        }

    def _segment_with_sam3(
        self,
        image: Union[Image.Image, str],
        query: str
    ) -> Optional[np.ndarray]:
        """
        Segment an object using SAM3 based on a text query.

        Args:
            image: PIL Image or path
            query: Text description of what to segment

        Returns:
            np.ndarray: Segmentation mask or None if failed
        """
        try:
            # Convert to PIL if needed
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')

            # Call SAM3 processor
            # SAM3's processor expects: (image, text_query)
            result = self.sam3_processor(image, query)

            # Extract mask from result
            if isinstance(result, dict) and 'mask' in result:
                return result['mask']
            elif isinstance(result, np.ndarray):
                return result
            else:
                logger.warning(f"Unexpected SAM3 result type: {type(result)}")
                return None

        except Exception as e:
            logger.error(f"SAM3 segmentation error: {e}")
            return None

    def detect_infrastructure_batch(
        self,
        images: List[Union[Image.Image, str]],
        use_sam3: bool = True
    ) -> List[Dict]:
        """
        Batch detection with agentic SAM3 segmentation.

        Args:
            images: List of PIL Images or image paths
            use_sam3: If True, add SAM3 segmentation

        Returns:
            List of detection dictionaries
        """
        results = []
        for image in images:
            result = self.detect_infrastructure(image, use_sam3=use_sam3)
            results.append(result)
        return results
