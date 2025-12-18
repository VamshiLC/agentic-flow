#!/usr/bin/env python3
"""
SAM3 Multi-Category Detector - Flexible & Configurable

Detect anything with SAM3 using text prompts:
- Use config file OR command-line arguments
- Blur privacy-sensitive categories
- Detect infrastructure defects
- Fully customizable
"""
import sys
import cv2
import json
import argparse
import numpy as np
from pathlib import Path
from PIL import Image


def load_config(config_path="categories_config.json"):
    """Load configuration from JSON file."""
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    return None


def draw_detections(image, detections, category, color):
    """Draw bounding boxes for detected objects."""
    output = image.copy()

    for i, box in enumerate(detections):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

        label = f"{category} {i+1}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(output, (x1, y1 - label_size[1] - 10),
                     (x1 + label_size[0] + 10, y1), color, -1)
        cv2.putText(output, label, (x1 + 5, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return output


def blur_regions(image, boxes, strength=51, padding=20):
    """Apply Gaussian blur to regions."""
    output = image.copy()

    if strength % 2 == 0:
        strength += 1

    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box]

        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image.shape[1], x2 + padding)
        y2 = min(image.shape[0], y2 + padding)

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
        good_boxes = [box for box, score in zip(boxes, scores) if score > score_threshold]
        return good_boxes
    except Exception as e:
        return []


def main():
    parser = argparse.ArgumentParser(
        description="SAM3 Multi-Category Detection with Privacy Blur",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default config
  python sam3_detect.py road.jpg

  # Custom categories via command line
  python sam3_detect.py road.jpg --blur face --detect pothole crack car "license plate"

  # Custom output directory
  python sam3_detect.py road.jpg --output results/

  # Adjust blur and threshold
  python sam3_detect.py road.jpg --blur-strength 71 --threshold 0.6

  # Use custom config file
  python sam3_detect.py road.jpg --config my_config.json
        """
    )

    parser.add_argument('input', help='Input image path')
    parser.add_argument('--output', '-o', default='sam3_detections', help='Output directory (default: sam3_detections)')
    parser.add_argument('--config', '-c', help='Config JSON file (default: categories_config.json)')
    parser.add_argument('--blur', nargs='+', help='Categories to blur (e.g., face "license plate")')
    parser.add_argument('--detect', nargs='+', help='Categories to detect (e.g., pothole crack car)')
    parser.add_argument('--blur-strength', type=int, default=51, help='Blur kernel size (default: 51)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Detection score threshold (default: 0.5)')
    parser.add_argument('--padding', type=int, default=20, help='Padding around blur regions (default: 20)')

    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"ERROR: File not found: {args.input}")
        sys.exit(1)

    # Load config or use command-line args
    if args.blur or args.detect:
        # Use command-line categories
        blur_categories = args.blur or []
        detect_categories = args.detect or []
        settings = {
            "blur_strength": args.blur_strength,
            "score_threshold": args.threshold,
            "add_padding": args.padding
        }
        category_colors = {}
    else:
        # Load from config file
        config_file = args.config or "categories_config.json"
        config = load_config(config_file)

        if config:
            print(f"✓ Loaded config from: {config_file}")
            blur_categories = config.get("blur_categories", [])
            detect_categories = config.get("detect_categories", [])
            settings = config.get("settings", {})
            category_colors = {k: tuple(v) for k, v in config.get("colors", {}).items()}
        else:
            # Default categories
            blur_categories = ["face"]
            detect_categories = ["license plate", "pothole", "crack", "car", "person"]
            settings = {
                "blur_strength": args.blur_strength,
                "score_threshold": args.threshold,
                "add_padding": args.padding
            }
            category_colors = {}

    # Default color scheme
    default_colors = {
        "face": (0, 255, 0), "license plate": (0, 255, 255),
        "pothole": (0, 0, 255), "crack": (255, 0, 0),
        "car": (255, 0, 255), "person": (0, 165, 255),
        "road sign": (255, 255, 0), "traffic light": (0, 255, 128),
    }

    print("=" * 70)
    print("SAM3 Multi-Category Detection + Privacy Blur")
    print("=" * 70)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}/")
    print(f"\nBLUR categories:   {', '.join(blur_categories) if blur_categories else 'None'}")
    print(f"DETECT categories: {', '.join(detect_categories) if detect_categories else 'None'}")
    print(f"\nSettings:")
    print(f"  Blur strength:    {settings.get('blur_strength', 51)}")
    print(f"  Score threshold:  {settings.get('score_threshold', 0.5)}")
    print(f"  Padding:          {settings.get('add_padding', 20)}px")
    print()

    # Read image
    image_cv = cv2.imread(args.input)
    if image_cv is None:
        print("ERROR: Could not read image")
        sys.exit(1)

    h, w = image_cv.shape[:2]
    print(f"Image size: {w}x{h}")

    # Create output directory
    output_path = Path(args.output)
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
    print("\n2. Preparing image...")
    image_pil = Image.open(args.input)
    inference_state = processor.set_image(image_pil)
    print("   ✓ Ready")

    # Detect all categories
    print("\n3. Detecting categories...")
    all_detections = {}
    all_categories = blur_categories + detect_categories

    for category in all_categories:
        print(f"   • {category:20}", end=" ")
        boxes = detect_category(
            processor, inference_state, category,
            settings.get('score_threshold', 0.5)
        )
        all_detections[category] = boxes
        print(f"→ {len(boxes)} found")

    # Save original
    cv2.imwrite(str(output_path / "0_original.jpg"), image_cv)

    # Detection visualization
    print("\n4. Creating visualizations...")
    detection_viz = image_cv.copy()

    for category, boxes in all_detections.items():
        if len(boxes) > 0:
            color = category_colors.get(category, default_colors.get(category, (128, 128, 128)))
            detection_viz = draw_detections(detection_viz, boxes, category, color)

    cv2.imwrite(str(output_path / "1_all_detections.jpg"), detection_viz)
    print("   ✓ All detections saved")

    # Apply blur
    if blur_categories:
        print("\n5. Applying privacy blur...")
        blurred_image = image_cv.copy()
        total_blurred = 0

        for category in blur_categories:
            boxes = all_detections.get(category, [])
            if len(boxes) > 0:
                blurred_image = blur_regions(
                    blurred_image, boxes,
                    settings.get('blur_strength', 51),
                    settings.get('add_padding', 20)
                )
                total_blurred += len(boxes)
                print(f"   ✓ Blurred {len(boxes)} {category}(s)")

        cv2.imwrite(str(output_path / "2_privacy_protected.jpg"), blurred_image)
    else:
        blurred_image = image_cv.copy()
        total_blurred = 0
        print("\n5. No blur categories specified, skipping...")

    # Final annotated
    print("\n6. Creating final annotated image...")
    final_viz = blurred_image.copy()

    for category in detect_categories:
        boxes = all_detections.get(category, [])
        if len(boxes) > 0:
            color = category_colors.get(category, default_colors.get(category, (128, 128, 128)))
            final_viz = draw_detections(final_viz, boxes, category, color)

    cv2.imwrite(str(output_path / "3_final_annotated.jpg"), final_viz)
    print("   ✓ Final annotated saved")

    # JSON summary
    summary = {
        "image": str(args.input),
        "image_size": {"width": w, "height": h},
        "categories": {
            category: {
                "count": len(boxes),
                "action": "blur" if category in blur_categories else "detect",
                "boxes": [[int(v) for v in box] for box in boxes]
            }
            for category, boxes in all_detections.items()
        },
        "total_detections": sum(len(boxes) for boxes in all_detections.values()),
        "total_blurred": total_blurred,
        "settings": settings
    }

    with open(output_path / "detections.json", 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for category in blur_categories:
        boxes = all_detections.get(category, [])
        if len(boxes) > 0:
            print(f"🔒 {category:20} → {len(boxes):2} BLURRED")

    for category in detect_categories:
        boxes = all_detections.get(category, [])
        if len(boxes) > 0:
            print(f"🎯 {category:20} → {len(boxes):2} DETECTED")

    print("\n" + "=" * 70)
    print(f"📁 Results: {output_path}/")
    print("=" * 70)
    print(f"   0_original.jpg          - Original image")
    print(f"   1_all_detections.jpg    - All detections visualized")
    if total_blurred > 0:
        print(f"   2_privacy_protected.jpg - Privacy blur applied ({total_blurred} blurred)")
    print(f"   3_final_annotated.jpg   - Final annotated (privacy-safe)")
    print(f"   detections.json         - Detection data")
    print()


if __name__ == "__main__":
    main()
