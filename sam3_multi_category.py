#!/usr/bin/env python3
"""
SAM3 Multi-Category Detection + Privacy Blur

Detect multiple categories with SAM3:
- Blur: face, license plate
- Detect: pothole, crack, car, person, road sign, etc.
"""
import sys
import cv2
import json
import numpy as np
from pathlib import Path
from PIL import Image


def draw_detections(image, detections, category, color):
    """Draw bounding boxes for detected objects."""
    output = image.copy()

    for i, box in enumerate(detections):
        x1, y1, x2, y2 = [int(v) for v in box]

        # Draw rectangle
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

        # Add label
        label = f"{category} {i+1}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(output, (x1, y1 - label_size[1] - 10),
                     (x1 + label_size[0] + 10, y1), color, -1)
        cv2.putText(output, label, (x1 + 5, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return output


def blur_regions(image, boxes, strength=51):
    """Apply Gaussian blur to regions."""
    output = image.copy()

    if strength % 2 == 0:
        strength += 1

    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box]

        # Add padding
        pad = 20
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(image.shape[1], x2 + pad)
        y2 = min(image.shape[0], y2 + pad)

        region = output[y1:y2, x1:x2]
        if region.size == 0:
            continue

        blurred = cv2.GaussianBlur(region, (strength, strength), 0)
        output[y1:y2, x1:x2] = blurred

    return output


def detect_category(processor, inference_state, category, score_threshold=0.5):
    """Detect a single category using SAM3."""
    try:
        output = processor.set_text_prompt(state=inference_state, prompt=category)
        boxes = output["boxes"]
        scores = output["scores"]

        # Filter by score
        good_boxes = [box for box, score in zip(boxes, scores) if score > score_threshold]
        return good_boxes
    except Exception as e:
        print(f"   ✗ Error detecting '{category}': {e}")
        return []


def main():
    if len(sys.argv) < 2:
        print("Usage: python sam3_multi_category.py <input_image> [output_dir]")
        print()
        print("Examples:")
        print("  python sam3_multi_category.py road.jpg")
        print("  python sam3_multi_category.py road.jpg results/")
        print()
        print("Default categories:")
        print("  Blur:   face")
        print("  Detect: license plate, pothole, crack, car, person")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "sam3_detections"

    if not Path(input_path).exists():
        print(f"ERROR: File not found: {input_path}")
        sys.exit(1)

    # Configuration
    BLUR_CATEGORIES = ["face"]
    DETECT_CATEGORIES = ["license plate", "pothole", "crack", "car", "person", "road sign"]
    BLUR_STRENGTH = 51
    SCORE_THRESHOLD = 0.5

    # Colors for different categories (BGR)
    CATEGORY_COLORS = {
        "face": (0, 255, 0),          # Green
        "license plate": (0, 255, 255), # Yellow
        "pothole": (0, 0, 255),         # Red
        "crack": (255, 0, 0),           # Blue
        "car": (255, 0, 255),           # Magenta
        "person": (0, 165, 255),        # Orange
        "road sign": (255, 255, 0),     # Cyan
    }

    print("=" * 70)
    print("SAM3 Multi-Category Detection + Privacy Blur")
    print("=" * 70)
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}/")
    print(f"\nCategories to BLUR:   {', '.join(BLUR_CATEGORIES)}")
    print(f"Categories to DETECT: {', '.join(DETECT_CATEGORIES)}")
    print()

    # Read image
    image_cv = cv2.imread(input_path)
    if image_cv is None:
        print("ERROR: Could not read image")
        sys.exit(1)

    h, w = image_cv.shape[:2]
    print(f"Image size: {w}x{h}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Load SAM3
    print("\n1. Loading SAM3 model...")
    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        model = build_sam3_image_model()
        processor = Sam3Processor(model)
        print("   ✓ SAM3 loaded")
    except Exception as e:
        print(f"   ✗ ERROR: {e}")
        print("\nInstall SAM3:")
        print("  git clone https://github.com/facebookresearch/sam3.git")
        print("  cd sam3 && pip install -e .")
        print("  huggingface-cli login")
        sys.exit(1)

    # Prepare image
    print("\n2. Preparing image for inference...")
    image_pil = Image.open(input_path)
    inference_state = processor.set_image(image_pil)
    print("   ✓ Image prepared")

    # Detect all categories
    print("\n3. Detecting categories...")
    all_detections = {}

    all_categories = BLUR_CATEGORIES + DETECT_CATEGORIES

    for category in all_categories:
        print(f"   • {category}...", end=" ")
        boxes = detect_category(processor, inference_state, category, SCORE_THRESHOLD)
        all_detections[category] = boxes
        print(f"{len(boxes)} found")

    # Save original
    original_file = output_path / "0_original.jpg"
    cv2.imwrite(str(original_file), image_cv)

    # Create detection visualization (before blur)
    print("\n4. Creating detection visualization...")
    detection_viz = image_cv.copy()

    for category, boxes in all_detections.items():
        if len(boxes) > 0:
            color = CATEGORY_COLORS.get(category, (128, 128, 128))
            detection_viz = draw_detections(detection_viz, boxes, category, color)

    detection_file = output_path / "1_all_detections.jpg"
    cv2.imwrite(str(detection_file), detection_viz)
    print(f"   ✓ Saved: {detection_file}")

    # Apply blur to privacy categories
    print("\n5. Applying privacy blur...")
    blurred_image = image_cv.copy()

    total_blurred = 0
    for category in BLUR_CATEGORIES:
        boxes = all_detections.get(category, [])
        if len(boxes) > 0:
            blurred_image = blur_regions(blurred_image, boxes, BLUR_STRENGTH)
            total_blurred += len(boxes)
            print(f"   ✓ Blurred {len(boxes)} {category}(s)")

    blurred_file = output_path / "2_privacy_protected.jpg"
    cv2.imwrite(str(blurred_file), blurred_image)
    print(f"   ✓ Saved: {blurred_file}")

    # Create detection overlay on blurred image
    print("\n6. Creating final annotated image...")
    final_viz = blurred_image.copy()

    # Only draw non-blur categories on final image
    for category in DETECT_CATEGORIES:
        boxes = all_detections.get(category, [])
        if len(boxes) > 0:
            color = CATEGORY_COLORS.get(category, (128, 128, 128))
            final_viz = draw_detections(final_viz, boxes, category, color)

    final_file = output_path / "3_final_annotated.jpg"
    cv2.imwrite(str(final_file), final_viz)
    print(f"   ✓ Saved: {final_file}")

    # Create summary JSON
    print("\n7. Saving detection summary...")
    summary = {
        "image": str(input_path),
        "image_size": {"width": w, "height": h},
        "categories": {
            category: {
                "count": len(boxes),
                "action": "blur" if category in BLUR_CATEGORIES else "detect",
                "boxes": [[int(v) for v in box] for box in boxes]
            }
            for category, boxes in all_detections.items()
        },
        "total_detections": sum(len(boxes) for boxes in all_detections.values()),
        "total_blurred": total_blurred,
        "score_threshold": SCORE_THRESHOLD,
        "blur_strength": BLUR_STRENGTH
    }

    json_file = output_path / "detections.json"
    with open(json_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"   ✓ Saved: {json_file}")

    # Print summary
    print("\n" + "=" * 70)
    print("DETECTION SUMMARY")
    print("=" * 70)

    for category, boxes in all_detections.items():
        if len(boxes) > 0:
            action = "BLURRED" if category in BLUR_CATEGORIES else "DETECTED"
            print(f"{category:15} → {len(boxes):2} {action}")

    print("\n" + "=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)
    print(f"0_original.jpg         - Original image")
    print(f"1_all_detections.jpg   - All detections visualized")
    print(f"2_privacy_protected.jpg- Privacy blur applied")
    print(f"3_final_annotated.jpg  - Final with detections (privacy-safe)")
    print(f"detections.json        - Complete detection data")
    print()
    print(f"📁 All files saved to: {output_path}/")
    print()


if __name__ == "__main__":
    main()
