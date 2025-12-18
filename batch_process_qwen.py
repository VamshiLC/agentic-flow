#!/usr/bin/env python3
"""
Qwen Batch Processor with Face Blur + Category Detection
Processes multiple images and organizes detections into category folders
"""
import sys
import cv2
import json
import re
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from datetime import datetime
from models.qwen_direct_loader import Qwen3VLDirectDetector


# Detection prompts for different categories
DETECTION_PROMPTS = {
    "face": """Analyze this image and detect all human faces.
For each face, provide: Face: <number>, Box: [x1, y1, x2, y2], Confidence: <0.0-1.0>
If NO faces detected, respond with: "No faces detected"
Now analyze:""",

    "pothole": """Detect all potholes in this road image.
For each pothole, provide: Defect: pothole, Box: [x1, y1, x2, y2], Confidence: <0.0-1.0>
If NO potholes detected, respond with: "No defects detected"
Now analyze:""",

    "crack": """Detect all cracks in this road/pavement image.
For each crack, provide: Defect: crack, Box: [x1, y1, x2, y2], Confidence: <0.0-1.0>
If NO cracks detected, respond with: "No defects detected"
Now analyze:""",

    "damaged_crosswalk": """Detect all damaged/faded crosswalks in this image.
For each damaged crosswalk, provide: Defect: damaged crosswalk, Box: [x1, y1, x2, y2], Confidence: <0.0-1.0>
If NO damaged crosswalks detected, respond with: "No defects detected"
Now analyze:""",

    "license_plate": """Detect all vehicle license plates in this image.
For each license plate, provide: Object: license plate, Box: [x1, y1, x2, y2], Confidence: <0.0-1.0>
If NO license plates detected, respond with: "No objects detected"
Now analyze:""",
}


def parse_detections(response, image_size, confidence_threshold=0.5):
    """Parse detections from Qwen response."""
    bboxes = []
    width, height = image_size

    if "no faces detected" in response.lower() or "no defects detected" in response.lower() or "no objects detected" in response.lower():
        return bboxes

    # Try multiple patterns
    patterns = [
        r'Face:\s*\d+,\s*Box:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\],\s*Confidence:\s*([\d.]+)',
        r'Defect:\s*[^,]+,\s*Box:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\],\s*Confidence:\s*([\d.]+)',
        r'Object:\s*[^,]+,\s*Box:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\],\s*Confidence:\s*([\d.]+)',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            break

    for match in matches:
        try:
            x1, y1, x2, y2, conf = map(float, match)

            if conf < confidence_threshold:
                continue

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Clamp to image bounds
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))

            if x2 > x1 and y2 > y1:
                bboxes.append((x1, y1, x2, y2))

        except (ValueError, IndexError):
            continue

    return bboxes


def blur_regions(image, boxes, strength=51, padding=20):
    """Apply Gaussian blur to regions."""
    output = image.copy()

    if strength % 2 == 0:
        strength += 1

    for (x1, y1, x2, y2) in boxes:
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


def crop_detection(image, box, padding=10):
    """Crop a detection region with padding."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box

    # Add padding
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    return image[y1:y2, x1:x2]


def process_image(image_path, detector, categories, blur_categories, output_root,
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
    image_pil = Image.open(image_path).convert('RGB')
    print(f"  Image size: {w}x{h}")

    # Detect all categories
    print(f"\n  Detecting categories...")
    all_detections = {}

    for category in categories:
        prompt = DETECTION_PROMPTS.get(category)
        if not prompt:
            print(f"    • {category:20} → No prompt available, skipped")
            continue

        try:
            result = detector.detect(image_pil, prompt, max_new_tokens=512)

            if result.get('success', False):
                response = result.get('text', '')
                boxes = parse_detections(response, (w, h), score_threshold)
                all_detections[category] = boxes

                if len(boxes) > 0:
                    print(f"    • {category:20} → {len(boxes)} found")
            else:
                all_detections[category] = []

        except Exception as e:
            print(f"    • {category:20} → Error: {e}")
            all_detections[category] = []

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
        description="Qwen Batch Processor - Face Blur + Category Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all images in a directory
  python batch_process_qwen.py input_images/ --blur face --detect pothole crack

  # Custom output directory
  python batch_process_qwen.py images/ --blur face license_plate --detect pothole --output results/

  # With custom settings
  python batch_process_qwen.py images/ --blur face --detect pothole crack damaged_crosswalk --blur-strength 71 --threshold 0.6

Available Categories:
  face, pothole, crack, damaged_crosswalk, license_plate

Output Structure:
  output/
    blurred_images/          - All images with faces blurred
    pothole/                 - Cropped pothole detections
    crack/                   - Cropped crack detections
    batch_summary.json       - Complete processing summary
        """
    )

    parser.add_argument('input_dir', help='Input directory with images')
    parser.add_argument('--output', '-o', default='batch_output', help='Output directory (default: batch_output)')
    parser.add_argument('--blur', nargs='+', required=True, help='Categories to blur (e.g., face license_plate)')
    parser.add_argument('--detect', nargs='+', required=True, help='Categories to detect and crop (e.g., pothole crack)')
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
    print("QWEN BATCH PROCESSOR")
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

    # Load Qwen
    print("=" * 70)
    print("Loading Qwen2.5-VL-7B Model...")
    print("=" * 70)
    try:
        detector = Qwen3VLDirectDetector(
            model_name="Qwen/Qwen2.5-VL-7B-Instruct",
            use_quantization=False,
            low_memory=False
        )
        print("✓ Qwen loaded successfully\n")
    except Exception as e:
        print(f"✗ ERROR: {e}")
        print("\nMake sure Qwen is installed:")
        print("  pip install transformers torch pillow")
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
                image_path, detector, all_categories, args.blur,
                output_root, args.blur_strength, args.threshold, args.crop_padding
            )

            if result:
                batch_results.append(result)

        except Exception as e:
            print(f"  ✗ ERROR processing {image_path}: {e}")
            import traceback
            traceback.print_exc()

    # Cleanup
    detector.cleanup()

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
