#!/usr/bin/env python3
"""
SAM3 Batch Processor with Face Blur + Category Detection
Processes multiple images and organizes detections into category folders
"""
import sys
import cv2
import json
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from datetime import datetime


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
        print(f"    Warning: Detection failed for '{category}': {e}")
        return []


def crop_detection(image, box, padding=10):
    """Crop a detection region with padding."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]

    # Add padding
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    return image[y1:y2, x1:x2]


def process_image(image_path, processor, categories, blur_categories, output_root,
                  blur_strength=51, score_threshold=0.5, crop_padding=10):
    """Process a single image."""
    image_name = Path(image_path).stem

    print(f"\n{'='*70}")
    print(f"Processing: {image_path}")
    print(f"{'='*70}")

    # Read image
    image_cv = cv2.imread(str(image_path))
    if image_cv is None:
        print(f"  ✗ Could not read image")
        return None

    h, w = image_cv.shape[:2]
    print(f"  Image size: {w}x{h}")

    # Prepare for SAM3
    image_pil = Image.open(image_path)
    inference_state = processor.set_image(image_pil)

    # Detect all categories
    print(f"\n  Detecting categories...")
    all_detections = {}

    for category in categories:
        boxes = detect_category(processor, inference_state, category, score_threshold)
        all_detections[category] = boxes
        if len(boxes) > 0:
            print(f"    • {category:20} → {len(boxes)} found")

    # Blur privacy-sensitive regions
    blurred_image = image_cv.copy()
    total_blurred = 0

    if blur_categories:
        print(f"\n  Applying privacy blur...")
        for category in blur_categories:
            boxes = all_detections.get(category, [])
            if len(boxes) > 0:
                blurred_image = blur_regions(blurred_image, boxes, blur_strength, padding=20)
                total_blurred += len(boxes)
                print(f"    ✓ Blurred {len(boxes)} {category}(s)")

    # Save full blurred image
    blurred_dir = output_root / "blurred_images"
    blurred_dir.mkdir(exist_ok=True)
    blurred_path = blurred_dir / f"{image_name}_blurred.jpg"
    cv2.imwrite(str(blurred_path), blurred_image)
    print(f"\n  ✓ Saved blurred image: {blurred_path.relative_to(output_root)}")

    # Save cropped detections by category
    saved_crops = {}
    print(f"\n  Saving cropped detections...")

    for category, boxes in all_detections.items():
        if category in blur_categories:
            # Skip saving crops for blurred categories (privacy)
            continue

        if len(boxes) == 0:
            continue

        # Create category directory
        category_dir = output_root / category.replace(" ", "_")
        category_dir.mkdir(exist_ok=True)

        saved_crops[category] = []

        for i, box in enumerate(boxes):
            # Crop detection from blurred image (privacy-safe)
            crop = crop_detection(blurred_image, box, padding=crop_padding)

            if crop.size == 0:
                continue

            # Save crop
            crop_filename = f"{image_name}_{category.replace(' ', '_')}_{i+1}.jpg"
            crop_path = category_dir / crop_filename
            cv2.imwrite(str(crop_path), crop)
            saved_crops[category].append(str(crop_path.relative_to(output_root)))

        if saved_crops[category]:
            print(f"    ✓ {category:20} → {len(saved_crops[category])} crops saved")

    # Return summary
    return {
        "image": str(image_path),
        "image_size": {"width": w, "height": h},
        "total_blurred": total_blurred,
        "detections": {
            category: len(boxes)
            for category, boxes in all_detections.items()
        },
        "crops_saved": saved_crops
    }


def main():
    parser = argparse.ArgumentParser(
        description="SAM3 Batch Processor - Face Blur + Category Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all images in a directory
  python batch_process_sam3.py input_images/ --blur face --detect pothole crack

  # Custom output directory
  python batch_process_sam3.py images/ --blur face "license plate" --detect pothole --output results/

  # With custom settings
  python batch_process_sam3.py images/ --blur face --detect pothole crack car --blur-strength 71 --threshold 0.6

Output Structure:
  output/
    blurred_images/          - All images with faces blurred
    pothole/                 - Cropped pothole detections
    crack/                   - Cropped crack detections
    car/                     - Cropped car detections
    batch_summary.json       - Complete processing summary
        """
    )

    parser.add_argument('input_dir', help='Input directory with images')
    parser.add_argument('--output', '-o', default='batch_output', help='Output directory (default: batch_output)')
    parser.add_argument('--blur', nargs='+', required=True, help='Categories to blur (e.g., face "license plate")')
    parser.add_argument('--detect', nargs='+', required=True, help='Categories to detect and crop (e.g., pothole crack car)')
    parser.add_argument('--blur-strength', type=int, default=51, help='Blur kernel size (default: 51)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Detection score threshold (default: 0.5)')
    parser.add_argument('--crop-padding', type=int, default=10, help='Padding around crops (default: 10)')
    parser.add_argument('--extensions', nargs='+', default=['jpg', 'jpeg', 'png'], help='Image extensions to process')

    args = parser.parse_args()

    # Validate input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {args.input_dir}")
        sys.exit(1)

    # Find all images
    image_files = []
    for ext in args.extensions:
        image_files.extend(input_dir.glob(f"*.{ext}"))
        image_files.extend(input_dir.glob(f"*.{ext.upper()}"))

    image_files = sorted(set(image_files))

    if len(image_files) == 0:
        print(f"ERROR: No images found in {args.input_dir}")
        print(f"Looking for extensions: {args.extensions}")
        sys.exit(1)

    # Create output directory
    output_root = Path(args.output)
    output_root.mkdir(exist_ok=True)

    # Print configuration
    print("=" * 70)
    print("SAM3 BATCH PROCESSOR")
    print("Face Blur + Category Detection")
    print("=" * 70)
    print(f"Input directory:  {args.input_dir}")
    print(f"Output directory: {args.output}/")
    print(f"Images found:     {len(image_files)}")
    print(f"\nBLUR categories:   {', '.join(args.blur)}")
    print(f"DETECT categories: {', '.join(args.detect)}")
    print(f"\nSettings:")
    print(f"  Blur strength:    {args.blur_strength}")
    print(f"  Score threshold:  {args.threshold}")
    print(f"  Crop padding:     {args.crop_padding}px")
    print()

    # Load SAM3
    print("=" * 70)
    print("Loading SAM3 Model...")
    print("=" * 70)
    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        model = build_sam3_image_model()
        processor = Sam3Processor(model)
        print("✓ SAM3 loaded successfully\n")
    except Exception as e:
        print(f"✗ ERROR: {e}")
        print("\nInstall SAM3:")
        print("  git clone https://github.com/facebookresearch/sam3.git")
        print("  cd sam3 && pip install -e .")
        print("  huggingface-cli login")
        sys.exit(1)

    # Process all images
    all_categories = args.blur + args.detect
    batch_results = []
    start_time = datetime.now()

    for idx, image_path in enumerate(image_files, 1):
        print(f"\n{'='*70}")
        print(f"Image {idx}/{len(image_files)}")
        print(f"{'='*70}")

        try:
            result = process_image(
                image_path, processor, all_categories, args.blur,
                output_root, args.blur_strength, args.threshold, args.crop_padding
            )

            if result:
                batch_results.append(result)

        except Exception as e:
            print(f"  ✗ ERROR processing {image_path}: {e}")
            import traceback
            traceback.print_exc()

    # Save batch summary
    duration = (datetime.now() - start_time).total_seconds()

    summary = {
        "batch_info": {
            "input_directory": str(input_dir),
            "output_directory": str(output_root),
            "total_images": len(image_files),
            "processed_successfully": len(batch_results),
            "processing_time_seconds": duration,
            "timestamp": datetime.now().isoformat()
        },
        "settings": {
            "blur_categories": args.blur,
            "detect_categories": args.detect,
            "blur_strength": args.blur_strength,
            "score_threshold": args.threshold,
            "crop_padding": args.crop_padding
        },
        "results": batch_results,
        "category_totals": {}
    }

    # Calculate category totals
    for category in all_categories:
        total = sum(result['detections'].get(category, 0) for result in batch_results)
        summary['category_totals'][category] = total

    summary_path = output_root / "batch_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print final summary
    print("\n" + "=" * 70)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 70)
    print(f"Time taken:       {duration:.1f} seconds ({duration/len(image_files):.1f}s per image)")
    print(f"Images processed: {len(batch_results)}/{len(image_files)}")
    print()

    print("Category Summary:")
    for category in args.blur:
        count = summary['category_totals'].get(category, 0)
        if count > 0:
            print(f"  🔒 {category:20} → {count:3} BLURRED")

    for category in args.detect:
        count = summary['category_totals'].get(category, 0)
        if count > 0:
            print(f"  🎯 {category:20} → {count:3} DETECTED & CROPPED")

    print()
    print("=" * 70)
    print(f"📁 Results: {output_root}/")
    print("=" * 70)
    print(f"  blurred_images/          - {len(batch_results)} blurred images")

    for category in args.detect:
        category_dir = output_root / category.replace(" ", "_")
        if category_dir.exists():
            num_crops = len(list(category_dir.glob("*.jpg")))
            if num_crops > 0:
                print(f"  {category.replace(' ', '_'):20}/     - {num_crops} cropped detections")

    print(f"  batch_summary.json       - Complete processing log")
    print()


if __name__ == "__main__":
    main()
