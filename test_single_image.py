#!/usr/bin/env python3
"""
Quick Test Script for Single Image Detection with SAM3 Segmentation

Usage:
    python test_single_image.py --input /path/to/image.jpg --output test_output/
"""
import argparse
import json
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

from detector_unified import UnifiedInfrastructureDetector, DEFECT_COLORS
from visualization_styles import draw_stylish_detections


def draw_detections_with_masks(image_array, detections):
    """Draw bounding boxes and colored segmentation masks on image with modern styling."""
    # Print detection info for debugging
    for idx, det in enumerate(detections):
        label = det.get('label', 'unknown')
        bbox = det.get('bbox', [])
        confidence = det.get('confidence', 0.0)
        has_mask = det.get('has_mask', False)

        if has_mask:
            print(f"  [{idx+1}] {label}: confidence={confidence:.3f}, bbox={bbox}, ✓ MASK RENDERED")
        else:
            print(f"  [{idx+1}] {label}: confidence={confidence:.3f}, bbox={bbox}, ✗ NO MASK")

    # Use the new stylish visualization module
    return draw_stylish_detections(image_array, detections, draw_masks=True)


def main():
    parser = argparse.ArgumentParser(description="Test single image detection with SAM3")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", default="test_output", help="Output directory")
    parser.add_argument("--device", choices=["cuda", "cpu"], help="Device to use")
    parser.add_argument("--quantize", action="store_true", help="Use 8-bit quantization")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("SINGLE IMAGE DETECTION TEST - Qwen3-VL + SAM3")
    print("="*70)
    print(f"Input: {args.input}")
    print(f"Output: {output_dir}")
    print(f"Device: {args.device or 'auto-detect'}")
    print("="*70)

    # Load image
    print("\nLoading image...")
    image = Image.open(args.input).convert('RGB')
    image_array = np.array(image)
    print(f"  Image size: {image.size}")

    # Initialize detector
    print("\nInitializing detector (Qwen3-VL + SAM3)...")
    detector = UnifiedInfrastructureDetector(
        device=args.device,
        use_quantization=args.quantize
    )

    # Run detection with SAM3
    print("\nRunning detection with SAM3 segmentation...")
    result = detector.detect_infrastructure(image, use_sam3=True)

    print(f"\n{'='*70}")
    print(f"RESULTS:")
    print(f"{'='*70}")
    print(f"Total detections: {result['num_detections']}")
    print(f"Has masks: {result.get('has_masks', False)}")

    if result['num_detections'] > 0:
        print(f"\nDetections found:")

        # Draw annotated image
        annotated = draw_detections_with_masks(image_array, result['detections'])

        # Save annotated image
        annotated_path = output_dir / f"{Path(args.input).stem}_detected.jpg"
        annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(annotated_path), annotated_bgr)
        print(f"\n✓ Saved annotated image: {annotated_path}")

        # Save JSON (without masks - too large)
        json_safe_detections = []
        for det in result['detections']:
            safe_det = {
                'label': det.get('label'),
                'bbox': det.get('bbox'),
                'confidence': float(det.get('confidence', 0.0)),
                'has_mask': det.get('has_mask', False)
            }
            json_safe_detections.append(safe_det)

        json_path = output_dir / f"{Path(args.input).stem}_detections.json"
        with open(json_path, 'w') as f:
            json.dump({
                'image': args.input,
                'num_detections': result['num_detections'],
                'has_masks': result.get('has_masks', False),
                'detections': json_safe_detections
            }, f, indent=2)
        print(f"✓ Saved JSON: {json_path}")

    else:
        print("\n✓ No detections found (this is good if image has no issues!)")

    print(f"\n{'='*70}")
    print("TEST COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
