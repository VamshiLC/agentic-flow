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
    """
    Draw professional-looking bounding boxes and segmentation masks on image.

    Features:
    - Adaptive sizing based on image dimensions
    - Vibrant, distinct colors for each defect type
    - Semi-transparent mask overlays with visible contours
    - Clean text labels with readable backgrounds
    - Professional appearance suitable for reports/presentations
    """
    annotated = image_array.copy()
    height, width = image_array.shape[:2]

    # Adaptive sizing based on image dimensions
    # Scale factors relative to a 1920x1080 reference
    scale = min(width / 1920, height / 1080)
    scale = max(0.5, min(scale, 2.0))  # Clamp between 0.5x and 2x

    # Calculate adaptive parameters
    bbox_thickness = max(2, int(4 * scale))
    contour_thickness = max(2, int(3 * scale))
    font_scale = max(0.5, 0.8 * scale)
    font_thickness = max(1, int(2 * scale))
    text_padding = max(5, int(8 * scale))

    # Create overlay for semi-transparent masks
    mask_overlay = np.zeros_like(annotated)

    for idx, det in enumerate(detections):
        label = det.get('label', 'unknown')
        bbox = det.get('bbox', [])
        confidence = det.get('confidence', 0.0)
        has_mask = det.get('has_mask', False)
        mask = det.get('mask', None)

        if len(bbox) == 4:
            x1, y1, x2, y2 = map(int, bbox)

            # Get color (already in BGR format from DEFECT_COLORS)
            color_bgr = det.get('color', (0, 255, 0))

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
                    if mask_array.shape != (height, width):
                        mask_array = cv2.resize(mask_array, (width, height),
                                              interpolation=cv2.INTER_NEAREST)

                    # Create colored mask overlay (semi-transparent)
                    mask_overlay[mask_array > 0] = color_bgr

                    # Draw mask contour for better visibility
                    contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(annotated, contours, -1, color_bgr, contour_thickness)

                    print(f"  [{idx+1}] {label}: {confidence:.1%} | bbox={bbox} | ✓ SEGMENTED")

                except Exception as e:
                    print(f"  [{idx+1}] {label}: {confidence:.1%} | bbox={bbox} | ✗ MASK ERROR: {e}")
                    has_mask = False
            else:
                print(f"  [{idx+1}] {label}: {confidence:.1%} | bbox={bbox}")

            # Draw bounding box with adaptive thickness
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color_bgr, bbox_thickness)

            # Prepare label text (cleaner format)
            label_display = label.replace('_', ' ').title()
            text = f"{label_display} {confidence:.0%}"

            # Calculate text size for background
            (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                                         font_scale, font_thickness)

            # Position label background (above bbox, or below if too close to top)
            label_y_top = y1 - text_h - text_padding * 2 - baseline
            if label_y_top < 10:
                # Place below bbox if too close to top
                label_y_top = y2 + baseline
                label_y_bottom = label_y_top + text_h + text_padding * 2
                text_y = label_y_top + text_h + text_padding
            else:
                label_y_bottom = y1
                text_y = y1 - text_padding - baseline

            # Draw semi-transparent label background
            overlay = annotated.copy()
            cv2.rectangle(overlay,
                         (x1, label_y_top),
                         (x1 + text_w + text_padding * 2, label_y_bottom),
                         color_bgr, -1)
            cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)

            # Draw text border for better readability
            cv2.putText(annotated, text,
                       (x1 + text_padding, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       font_scale,
                       (0, 0, 0),  # Black border
                       font_thickness + 1)

            # Draw main text (white)
            cv2.putText(annotated, text,
                       (x1 + text_padding, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       font_scale,
                       (255, 255, 255),
                       font_thickness)

    # Blend mask overlay with annotated image (40% mask opacity)
    if np.any(mask_overlay):
        annotated = cv2.addWeighted(annotated, 1.0, mask_overlay, 0.4, 0)

    return annotated


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
