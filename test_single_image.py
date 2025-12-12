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


def draw_detections_with_masks(image_array, detections):
    """Draw bounding boxes and colored segmentation masks on image."""
    annotated = image_array.copy()

    # Create overlay for masks
    mask_overlay = annotated.copy()

    for idx, det in enumerate(detections):
        label = det.get('label', 'unknown')
        bbox = det.get('bbox', [])
        confidence = det.get('confidence', 0.0)
        has_mask = det.get('has_mask', False)
        mask = det.get('mask', None)

        if len(bbox) == 4:
            x1, y1, x2, y2 = map(int, bbox)
            color = det.get('color', (0, 255, 0))

            # Convert BGR to RGB for display
            color_rgb = (color[2], color[1], color[0])

            # Draw segmentation mask if available
            if has_mask and mask is not None:
                try:
                    # Convert mask to numpy array if needed
                    if isinstance(mask, list):
                        mask_array = np.array(mask, dtype=np.uint8)
                    else:
                        mask_array = np.array(mask, dtype=np.uint8)

                    # Ensure mask is 2D
                    if mask_array.ndim > 2:
                        mask_array = mask_array.squeeze()

                    # Resize mask to image size if needed
                    if mask_array.shape != (image_array.shape[0], image_array.shape[1]):
                        mask_array = cv2.resize(mask_array,
                                              (image_array.shape[1], image_array.shape[0]),
                                              interpolation=cv2.INTER_NEAREST)

                    # Create colored mask overlay
                    colored_mask = np.zeros_like(annotated)
                    colored_mask[mask_array > 0] = color_rgb

                    # Blend with original image (30% transparency)
                    mask_overlay = cv2.addWeighted(mask_overlay, 1.0, colored_mask, 0.3, 0)

                    # Draw mask contour
                    contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(annotated, contours, -1, color_rgb, 2)

                    print(f"  [{idx+1}] {label}: confidence={confidence:.3f}, bbox={bbox}, ✓ MASK RENDERED")

                except Exception as e:
                    print(f"  [{idx+1}] {label}: confidence={confidence:.3f}, bbox={bbox}, ✗ MASK ERROR: {e}")
                    has_mask = False
            else:
                print(f"  [{idx+1}] {label}: confidence={confidence:.3f}, bbox={bbox}, ✗ NO MASK")

            # Draw bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color_rgb, 3)

            # Draw label with confidence and mask status
            mask_status = "✓MASK" if has_mask else "✗NO-MASK"
            text = f"{label} {confidence:.2f} {mask_status}"

            # Background for text
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(annotated, (x1, y1 - text_h - 15), (x1 + text_w + 10, y1), color_rgb, -1)

            # Text
            cv2.putText(
                annotated,
                text,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

    # Blend mask overlay with annotated image
    final = cv2.addWeighted(annotated, 0.7, mask_overlay, 0.3, 0)

    return final


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
