#!/usr/bin/env python3
"""
Simple Detection & Organization
Input: Single image
Output: Blurred image + category folders + report
"""
import sys
import cv2
import json
import argparse
from pathlib import Path
from PIL import Image


def detect_with_sam3(image_path, categories, score_threshold=0.5):
    """Detect all categories using SAM3."""
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    # Load SAM3
    print("Loading SAM3...")
    model = build_sam3_image_model()
    processor = Sam3Processor(model)

    # Prepare image
    image_pil = Image.open(image_path)
    inference_state = processor.set_image(image_pil)

    # Detect all categories
    print(f"Detecting {len(categories)} categories...")
    detections = {}

    for category in categories:
        try:
            output = processor.set_text_prompt(state=inference_state, prompt=category)
            boxes = output["boxes"]
            scores = output["scores"]
            good_boxes = [box.tolist() for box, score in zip(boxes, scores) if score > score_threshold]
            detections[category] = good_boxes
            print(f"  {category:20} → {len(good_boxes)}")
        except:
            detections[category] = []

    return detections


def blur_regions(image, boxes, strength=51, padding=20):
    """Apply Gaussian blur to regions."""
    output = image.copy()
    h, w = image.shape[:2]

    if strength % 2 == 0:
        strength += 1

    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)

        region = output[y1:y2, x1:x2]
        if region.size > 0:
            blurred = cv2.GaussianBlur(region, (strength, strength), 0)
            output[y1:y2, x1:x2] = blurred

    return output


def crop_and_save(image, boxes, category, output_dir, image_name):
    """Crop detections and save to category folder."""
    if len(boxes) == 0:
        return []

    category_dir = output_dir / category.replace(" ", "_")
    category_dir.mkdir(exist_ok=True)

    saved_files = []
    for i, box in enumerate(boxes, 1):
        x1, y1, x2, y2 = [int(v) for v in box]

        # Add small padding
        pad = 10
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(image.shape[1], x2 + pad)
        y2 = min(image.shape[0], y2 + pad)

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        filename = f"{image_name}_{category.replace(' ', '_')}_{i}.jpg"
        filepath = category_dir / filename
        cv2.imwrite(str(filepath), crop)
        saved_files.append(str(filepath.name))

    return saved_files


def main():
    parser = argparse.ArgumentParser(
        description="Auto-detect and organize image detections",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect everything
  python detect_and_organize.py image.jpg

  # Custom categories
  python detect_and_organize.py image.jpg --categories face pothole crack car

  # Custom output folder
  python detect_and_organize.py image.jpg --output results/

Categories will be auto-detected and organized into folders.
        """
    )

    parser.add_argument('image', help='Input image file')
    parser.add_argument('--output', '-o', default='detections', help='Output directory')
    parser.add_argument('--categories', nargs='+',
                       default=['face', 'license plate', 'pothole', 'crack', 'car', 'person'],
                       help='Categories to detect')
    parser.add_argument('--blur-categories', nargs='+', default=['face', 'license plate'],
                       help='Categories to blur (privacy)')
    parser.add_argument('--blur-strength', type=int, default=51, help='Blur strength')
    parser.add_argument('--threshold', type=float, default=0.5, help='Detection threshold')

    args = parser.parse_args()

    # Validate input
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"ERROR: Image not found: {args.image}")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    image_name = image_path.stem

    print("=" * 70)
    print("AUTO-DETECT AND ORGANIZE")
    print("=" * 70)
    print(f"Image:      {args.image}")
    print(f"Output:     {args.output}/")
    print(f"Categories: {', '.join(args.categories)}")
    print(f"Blur:       {', '.join(args.blur_categories)}")
    print("=" * 70)
    print()

    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        print("ERROR: Could not read image")
        sys.exit(1)

    h, w = image.shape[:2]
    print(f"Image size: {w}x{h}\n")

    # Detect all categories
    try:
        detections = detect_with_sam3(image_path, args.categories, args.threshold)
    except Exception as e:
        print(f"ERROR: Detection failed: {e}")
        print("\nMake sure SAM3 is installed:")
        print("  git clone https://github.com/facebookresearch/sam3.git")
        print("  cd sam3 && pip install -e .")
        print("  huggingface-cli login")
        sys.exit(1)

    # Blur privacy-sensitive regions
    print(f"\nApplying privacy blur...")
    blurred_image = image.copy()
    total_blurred = 0

    for category in args.blur_categories:
        boxes = detections.get(category, [])
        if len(boxes) > 0:
            blurred_image = blur_regions(blurred_image, boxes, args.blur_strength)
            total_blurred += len(boxes)
            print(f"  Blurred {len(boxes)} {category}(s)")

    # Save final blurred image
    final_image_path = output_dir / f"{image_name}_final.jpg"
    cv2.imwrite(str(final_image_path), blurred_image)
    print(f"\n✓ Saved: {final_image_path.name}")

    # Crop and organize detections
    print(f"\nOrganizing detections into folders...")
    report = {
        "image": str(image_path),
        "image_size": {"width": w, "height": h},
        "total_blurred": total_blurred,
        "categories": {}
    }

    for category, boxes in detections.items():
        if category in args.blur_categories:
            # Don't save crops of blurred items (privacy)
            report["categories"][category] = {
                "count": len(boxes),
                "action": "blurred",
                "crops": []
            }
            continue

        if len(boxes) == 0:
            continue

        saved_files = crop_and_save(blurred_image, boxes, category, output_dir, image_name)
        report["categories"][category] = {
            "count": len(boxes),
            "action": "detected",
            "crops": saved_files
        }
        print(f"  {category:20} → {len(saved_files)} crops saved to {category.replace(' ', '_')}/")

    # Save report
    report_path = output_dir / "report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n✓ Saved: report.json")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for category, info in report["categories"].items():
        if info["action"] == "blurred":
            print(f"🔒 {category:20} → {info['count']} blurred")
        else:
            print(f"🎯 {category:20} → {info['count']} detected, {len(info['crops'])} crops saved")

    print("\n" + "=" * 70)
    print(f"📁 {output_dir}/")
    print("=" * 70)
    print(f"  {image_name}_final.jpg    - Final blurred image")

    for category, info in report["categories"].items():
        if info["action"] == "detected" and info["count"] > 0:
            print(f"  {category.replace(' ', '_'):20}/ - {len(info['crops'])} crops")

    print(f"  report.json               - Detection report")
    print()


if __name__ == "__main__":
    main()
