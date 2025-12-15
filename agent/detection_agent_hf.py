"""
Infrastructure Detection Agent using Qwen3-VL (Hugging Face) + SAM3

This implements the TRUE agentic flow matching the official SAM3 agent:
https://github.com/facebookresearch/sam3/blob/main/sam3/agent/agent_core.py

Features:
- Multi-turn conversation with LLM
- 4 tools: segment_phrase, examine_each_mask, select_masks_and_return, report_no_mask
- Iterative refinement when masks are unsatisfactory
- Message pruning for context management
- Debug logging

No vLLM server required - uses Hugging Face Transformers directly.
"""
import numpy as np
from PIL import Image
from typing import List, Dict, Optional, Union
import logging

from models.qwen_direct_loader import Qwen3VLDirectDetector
from models.sam3_loader import load_sam3_model
from .agent_core import InfrastructureDetectionAgentCore, AgentConfig, AgentResult

logger = logging.getLogger(__name__)


class InfrastructureDetectionAgentHF:
    """
    Agentic infrastructure detector using Qwen3-VL (HF) + SAM3.

    This is the TRUE agentic implementation with:
    - Multi-turn conversation loop
    - Tool calling (segment_phrase, examine_each_mask, etc.)
    - Iterative mask refinement
    - LLM-driven mask validation

    Architecture:
        Qwen3-VL (brain) <-> Agent Core (orchestrator) <-> SAM3 (tool)
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",  # 3B to fit with SAM3 on 22GB GPU
        sam3_processor=None,
        categories: Optional[List[str]] = None,
        device: Optional[str] = None,
        use_quantization: bool = False,
        low_memory: bool = False,
        sam3_confidence: float = 0.25,
        max_turns: int = 10,  # Fast detection
        debug: bool = False
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
            max_turns: Maximum agentic loop iterations
            debug: Enable debug logging
        """
        import torch

        self.model_name = model_name
        self.categories = categories
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_turns = max_turns
        self.debug = debug

        print(f"\n{'='*60}")
        print("INITIALIZING AGENTIC DETECTOR (Qwen3-VL + SAM3)")
        print(f"{'='*60}")
        print(f"  Model: {model_name}")
        print(f"  Device: {self.device}")
        print(f"  SAM3 confidence: {sam3_confidence}")
        print(f"  Max turns: {max_turns} (smart search all categories)")
        print(f"  Debug mode: {debug}")

        # Load Qwen3-VL detector (the "brain")
        print("\n[1/2] Loading Qwen3-VL (Vision-Language Model)...")
        self.qwen_detector = Qwen3VLDirectDetector(
            model_name=model_name,
            device=device,
            use_quantization=use_quantization,
            low_memory=low_memory
        )

        # Load SAM3 processor (the "tool")
        print("\n[2/2] Loading SAM3 (Segmentation Tool)...")
        if sam3_processor is None:
            self.sam3_processor = load_sam3_model(
                confidence_threshold=sam3_confidence,
                device=self.device
            )
        else:
            self.sam3_processor = sam3_processor
            print("Using pre-loaded SAM3 processor")

        print(f"\n{'='*60}")
        print("AGENTIC DETECTOR READY")
        print(f"{'='*60}\n")

    def detect_infrastructure(
        self,
        image: Union[Image.Image, str],
        use_sam3: bool = True,
        user_query: str = None
    ) -> Dict:
        """
        Detect infrastructure issues using the agentic loop.

        This is the main entry point for detection. It:
        1. Initializes the agent core
        2. Runs the multi-turn agentic loop
        3. Returns formatted detections

        Args:
            image: PIL Image or path to image file
            use_sam3: If True, use SAM3 for segmentation (always True in agentic mode)
            user_query: Optional custom query for the agent

        Returns:
            dict: {
                'detections': [...],  # List of detections with masks
                'num_detections': int,
                'has_masks': bool,
                'turns_taken': int,
                'text_response': str
            }
        """
        # Convert to PIL if needed
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')

        # Default query
        if user_query is None:
            user_query = "Analyze this road image and detect all infrastructure issues."

        # Create agent config - SMART mode with Qwen bbox detection
        # NEW FLOW: Qwen detects with bboxes → SAM3 segments those boxes
        # No LLM validation needed - Qwen already classified objects correctly
        config = AgentConfig(
            max_turns=self.max_turns,
            categories=self.categories,
            debug=self.debug,
            debug_dir="debug",
            force_all_categories=True,  # Use smart detection flow
            validate_with_llm=False,  # Disable slow validation - use high confidence instead
            confidence_threshold=0.8,  # HIGH threshold to reduce false positives
            optimize_memory=True,  # Clear Qwen before SAM3 for better GPU usage
        )

        # Create and run agent
        agent = InfrastructureDetectionAgentCore(
            qwen_detector=self.qwen_detector,
            sam3_processor=self.sam3_processor,
            config=config
        )

        # Run agentic loop
        result: AgentResult = agent.run(image, user_query)

        # Format detections for output
        formatted_detections = self._format_detections(result.detections)

        return {
            'detections': formatted_detections,
            'num_detections': result.num_detections,
            'has_masks': any(d.get('has_mask', False) for d in formatted_detections),
            'turns_taken': result.turns_taken,
            'success': result.success,
            'text_response': result.message,
            'final_image': result.final_image
        }

    def _format_detections(self, detections: List[Dict]) -> List[Dict]:
        """
        Format raw detections for output.

        Adds color coding and normalizes fields.

        Args:
            detections: Raw detections from agent

        Returns:
            List of formatted detection dicts
        """
        from detector_unified import DEFECT_COLORS

        formatted = []

        for det in detections:
            category = det.get('category', 'unknown')
            mask = det.get('mask')

            # Get color for category
            color = DEFECT_COLORS.get(category, (0, 255, 0))

            # Convert mask to list if numpy array
            if mask is not None:
                if hasattr(mask, 'tolist'):
                    mask_list = mask.tolist()
                elif isinstance(mask, np.ndarray):
                    mask_list = mask.tolist()
                else:
                    mask_list = mask
                has_mask = True
            else:
                mask_list = None
                has_mask = False

            formatted.append({
                'label': category,
                'category': category,
                'bbox': det.get('bbox', [0, 0, 0, 0]),
                'confidence': det.get('confidence', 0.0),
                'severity': det.get('severity', 'low'),
                'description': det.get('description', ''),
                'color': color,
                'mask': mask_list,
                'has_mask': has_mask,
                'mask_id': det.get('mask_id')
            })

        return formatted

    def detect_infrastructure_batch(
        self,
        images: List[Union[Image.Image, str]],
        use_sam3: bool = True
    ) -> List[Dict]:
        """
        Batch detection with agentic SAM3 segmentation.

        Note: Each image runs through the full agentic loop independently.
        This is not true batch processing but sequential processing for
        multiple images.

        Args:
            images: List of PIL Images or image paths
            use_sam3: If True, use SAM3 for segmentation

        Returns:
            List of detection results, one per image
        """
        results = []

        for i, image in enumerate(images):
            print(f"\nProcessing image {i+1}/{len(images)}...")
            result = self.detect_infrastructure(image, use_sam3=use_sam3)
            results.append(result)

        return results

    def cleanup(self):
        """Clean up models and free GPU memory."""
        import torch

        if hasattr(self.qwen_detector, 'cleanup'):
            self.qwen_detector.cleanup()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("GPU memory cleared")


# Backwards compatibility alias
InfrastructureDetectionAgent = InfrastructureDetectionAgentHF
