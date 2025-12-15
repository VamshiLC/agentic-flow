"""
Grounding DINO + SAM for Accurate Object Detection and Segmentation

This is the PROPER way to do text-based detection:
1. Grounding DINO: Detects objects from text prompts with bounding boxes (52.5 AP on COCO zero-shot!)
2. SAM: Segments the detected bounding boxes into precise masks

Much more accurate than SAM3 text prompts.

References:
- https://huggingface.co/docs/transformers/en/model_doc/grounding-dino
- https://github.com/IDEA-Research/Grounded-Segment-Anything
"""
import torch
import numpy as np
from PIL import Image, ImageDraw
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class GroundingDINOSAMDetector:
    """
    Accurate object detection using Grounding DINO + SAM.

    Grounding DINO achieves 52.5 AP on COCO zero-shot - far better than SAM3 text prompts.
    """

    # Colors for visualization
    COLORS = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
        (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    ]

    def __init__(
        self,
        grounding_dino_model: str = "IDEA-Research/grounding-dino-tiny",
        sam_model: str = "facebook/sam-vit-base",
        device: Optional[str] = None,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ):
        """
        Initialize Grounding DINO + SAM detector.

        Args:
            grounding_dino_model: HuggingFace model ID for Grounding DINO
            sam_model: HuggingFace model ID for SAM
            device: Device to use (auto-detect if None)
            box_threshold: Confidence threshold for bounding boxes
            text_threshold: Confidence threshold for text matching
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        print(f"Loading Grounding DINO + SAM on {self.device}...")

        # Load Grounding DINO
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        print("  [1/2] Loading Grounding DINO...")
        self.gdino_processor = AutoProcessor.from_pretrained(grounding_dino_model)
        self.gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            grounding_dino_model
        ).to(self.device)
        self.gdino_model.eval()

        # Load SAM for segmentation
        print("  [2/2] Loading SAM...")
        from transformers import SamModel, SamProcessor

        self.sam_processor = SamProcessor.from_pretrained(sam_model)
        self.sam_model = SamModel.from_pretrained(sam_model).to(self.device)
        self.sam_model.eval()

        print("✓ Grounding DINO + SAM loaded successfully!")

    def detect(
        self,
        image: Image.Image,
        text_prompt: str,
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
    ) -> Dict:
        """
        Detect objects matching text prompt and generate segmentation masks.

        Args:
            image: PIL Image
            text_prompt: Text description of objects to detect
                        Use periods to separate multiple classes: "a cat. a dog. a pothole."
            box_threshold: Override default box threshold
            text_threshold: Override default text threshold

        Returns:
            Dict with 'detections' list and 'annotated_image'
        """
        box_threshold = box_threshold or self.box_threshold
        text_threshold = text_threshold or self.text_threshold

        # Ensure image is RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Step 1: Grounding DINO detection
        print(f"  Detecting: '{text_prompt}'...")

        # Format text labels properly
        text_labels = [[label.strip() for label in text_prompt.replace(".", ". ").split(". ") if label.strip()]]
        if not text_labels[0]:
            text_labels = [[text_prompt]]

        inputs = self.gdino_processor(
            images=image,
            text=text_labels,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.gdino_model(**inputs)

        # Post-process detections
        results = self.gdino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]]  # (height, width)
        )[0]

        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"]  # text labels

        print(f"  Found {len(boxes)} detections")

        if len(boxes) == 0:
            return {
                "detections": [],
                "annotated_image": image,
                "num_detections": 0
            }

        # Step 2: SAM segmentation for each box
        print(f"  Generating masks with SAM...")

        detections = []
        masks_list = []

        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            # Convert box to SAM format
            box_for_sam = [[box.tolist()]]  # [[x1, y1, x2, y2]]

            sam_inputs = self.sam_processor(
                image,
                input_boxes=box_for_sam,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                sam_outputs = self.sam_model(**sam_inputs)

            # Get the mask
            masks = self.sam_processor.image_processor.post_process_masks(
                sam_outputs.pred_masks.cpu(),
                sam_inputs["original_sizes"].cpu(),
                sam_inputs["reshaped_input_sizes"].cpu()
            )[0]

            # Take the best mask (highest IoU score)
            mask = masks[0, 0].numpy()  # First mask, first prediction

            detection = {
                "id": i + 1,
                "label": label,
                "confidence": float(score),
                "bbox": box.tolist(),
                "mask": mask
            }
            detections.append(detection)
            masks_list.append((mask, label, score))

        # Create annotated image
        annotated_image = self._draw_detections(image, detections)

        return {
            "detections": detections,
            "annotated_image": annotated_image,
            "num_detections": len(detections)
        }

    def detect_infrastructure(
        self,
        image: Image.Image,
        categories: Optional[List[str]] = None,
    ) -> Dict:
        """
        Detect infrastructure issues in an image.

        Args:
            image: PIL Image
            categories: List of categories to detect (default: comprehensive list)

        Returns:
            Dict with detections and annotated image
        """
        if categories is None:
            # Comprehensive infrastructure categories
            categories = [
                "pothole", "crack in road", "road damage",
                "manhole cover", "manhole",
                "graffiti", "spray paint on wall",
                "abandoned vehicle", "abandoned car",
                "debris", "trash", "garbage",
                "street sign", "traffic sign", "traffic light",
                "crosswalk", "road marking",
                "tent", "encampment"
            ]

        # Create text prompt (periods separate classes for Grounding DINO)
        text_prompt = ". ".join(categories) + "."

        return self.detect(image, text_prompt)

    def _draw_detections(
        self,
        image: Image.Image,
        detections: List[Dict]
    ) -> Image.Image:
        """Draw detection boxes and masks on image."""
        from PIL import ImageFont

        # Create copy
        result = image.copy()

        # Draw masks first (semi-transparent)
        for i, det in enumerate(detections):
            mask = det["mask"]
            color = self.COLORS[i % len(self.COLORS)]

            # Create colored mask overlay
            mask_image = Image.new("RGBA", image.size, (0, 0, 0, 0))
            mask_array = np.array(mask_image)

            # Apply color where mask is True
            mask_bool = mask > 0.5
            mask_array[mask_bool] = (*color, 128)  # Semi-transparent

            mask_overlay = Image.fromarray(mask_array, mode="RGBA")
            result = Image.alpha_composite(result.convert("RGBA"), mask_overlay)

        result = result.convert("RGB")
        draw = ImageDraw.Draw(result)

        # Load font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        except:
            font = ImageFont.load_default()

        # Draw boxes and labels
        for i, det in enumerate(detections):
            box = det["bbox"]
            label = det["label"]
            score = det["confidence"]
            color = self.COLORS[i % len(self.COLORS)]

            x1, y1, x2, y2 = box

            # Draw box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            # Draw label
            label_text = f"{label}: {score:.2f}"
            text_bbox = draw.textbbox((x1, y1 - 20), label_text, font=font)
            draw.rectangle(text_bbox, fill=color)
            draw.text((x1, y1 - 20), label_text, fill="white", font=font)

        return result

    def cleanup(self):
        """Release GPU memory."""
        del self.gdino_model
        del self.sam_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def get_detector(
    device: Optional[str] = None,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
) -> GroundingDINOSAMDetector:
    """
    Factory function to create Grounding DINO + SAM detector.
    """
    return GroundingDINOSAMDetector(
        device=device,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )


if __name__ == "__main__":
    # Test
    print("Testing Grounding DINO + SAM detector...")

    detector = get_detector()

    # Create test image
    test_image = Image.new("RGB", (640, 480), color="gray")

    result = detector.detect(test_image, "pothole. manhole. crack.")
    print(f"Detections: {result['num_detections']}")

    detector.cleanup()
