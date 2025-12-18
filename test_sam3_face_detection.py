#!/usr/bin/env python3
"""
Test SAM3 Face Detection with Visualization

Shows detected faces with bounding boxes and masks BEFORE blurring,
then saves the blurred versions.
"""
import sys
import cv2
import torch
import numpy as np
from pathlib import Path
from utils.face_blur import FaceBlurrer
from models.sam3_text_prompt_loader import load_sam3_text_prompt_model


def draw_face_detections(image, bboxes):
    """Draw bounding boxes around detected faces."""
    output = image.copy()

    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        # Draw rectangle
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Add label
        label = f"Face {i+1}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(output, (x1, y1 - label_size[1] - 10),
                     (x1 + label_size[0], y1), (0, 255, 0), -1)
        cv2.putText(output, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    return output


def test_sam3_face_detection(image_path: str, output_dir: str = "test_sam3_detections"):
    """Test SAM3 face detection with visualization."""

    print("="*70)
    print("SAM3 FACE DETECTION TEST (with visualization)")
    print("="*70)
    print(f"Image: {image_path}\n")

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Could not read image: {image_path}")
        return

    print(f"Image size: {image.shape[1]}x{image.shape[0]}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Load SAM3 model
    print("\n1. Loading SAM3 model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Device: {device}")

    try:
        model, processor, loader = load_sam3_text_prompt_model(device=device)
        print("   ✓ SAM3 model loaded\n")
    except Exception as e:
        print(f"   ✗ ERROR loading SAM3: {e}")
        return

    # Create face blurrer
    print("2. Initializing SAM3 face detector...")
    try:
        blurrer = FaceBlurrer(
            backend='sam3',
            blur_type='gaussian',
            blur_strength=51,
            sam3_model=model,
            sam3_processor=processor,
            device=device
        )
        print("   ✓ Face detector initialized\n")
    except Exception as e:
        print(f"   ✗ ERROR: {e}")
        return

    # Detect faces (get bounding boxes)
    print("3. Detecting faces...")
    print("   Processing with SAM3 text prompt: 'face'")

    try:
        # Get bounding boxes
        bboxes = blurrer.detect_faces_sam3(image)
        num_faces = len(bboxes)

        print(f"   ✓ Detected {num_faces} face(s)\n")

        if num_faces > 0:
            print("   Face bounding boxes:")
            for i, (x1, y1, x2, y2) in enumerate(bboxes):
                width = x2 - x1
                height = y2 - y1
                print(f"     Face {i+1}: [{x1}, {y1}, {x2}, {y2}] - Size: {width}x{height}px")
        else:
            print("   No faces detected in image.")
            print("\n   Possible reasons:")
            print("   - No faces in image")
            print("   - Faces too small")
            print("   - Unusual angles or lighting")
            return

    except Exception as e:
        print(f"   ✗ ERROR during detection: {e}")
        return

    # Save detection visualization
    print("\n4. Saving detection visualization...")

    # Draw bounding boxes
    detected_image = draw_face_detections(image, bboxes)
    detection_file = output_path / "1_face_detections.jpg"
    cv2.imwrite(str(detection_file), detected_image)
    print(f"   ✓ Saved detections: {detection_file}")

    # Save blurred versions
    print("\n5. Saving blurred versions...")

    # Gaussian blur
    blurred_gaussian, _ = blurrer.blur_faces(image.copy(), return_face_count=True)
    gaussian_file = output_path / "2_blurred_gaussian.jpg"
    cv2.imwrite(str(gaussian_file), blurred_gaussian)
    print(f"   ✓ Gaussian blur: {gaussian_file}")

    # Pixelation
    blurrer_pixel = FaceBlurrer(
        backend='sam3',
        blur_type='pixelate',
        blur_strength=20,
        sam3_model=model,
        sam3_processor=processor,
        device=device
    )
    blurred_pixel, _ = blurrer_pixel.blur_faces(image.copy(), return_face_count=True)
    pixel_file = output_path / "3_blurred_pixelate.jpg"
    cv2.imwrite(str(pixel_file), blurred_pixel)
    print(f"   ✓ Pixelation: {pixel_file}")

    # Strong blur
    blurrer_strong = FaceBlurrer(
        backend='sam3',
        blur_type='gaussian',
        blur_strength=101,
        sam3_model=model,
        sam3_processor=processor,
        device=device
    )
    blurred_strong, _ = blurrer_strong.blur_faces(image.copy(), return_face_count=True)
    strong_file = output_path / "4_blurred_strong.jpg"
    cv2.imwrite(str(strong_file), blurred_strong)
    print(f"   ✓ Strong blur: {strong_file}")

    # Save original
    original_file = output_path / "0_original.jpg"
    cv2.imwrite(str(original_file), image)
    print(f"   ✓ Original: {original_file}")

    # Create detection summary JSON
    import json
    summary = {
        "image": str(image_path),
        "image_size": {"width": image.shape[1], "height": image.shape[0]},
        "num_faces_detected": num_faces,
        "faces": [
            {
                "face_id": i + 1,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "width": int(x2 - x1),
                "height": int(y2 - y1),
                "area": int((x2 - x1) * (y2 - y1))
            }
            for i, (x1, y1, x2, y2) in enumerate(bboxes)
        ],
        "detection_method": "SAM3 text prompt: 'face'",
        "device": device
    }

    json_file = output_path / "face_detections.json"
    with open(json_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"   ✓ Detection JSON: {json_file}")

    print("\n" + "="*70)
    print("DETECTION TEST COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {output_path}/\n")
    print("Generated files:")
    print("  0_original.jpg           - Original image")
    print("  1_face_detections.jpg    - Detected faces (green boxes)")
    print("  2_blurred_gaussian.jpg   - Gaussian blur applied")
    print("  3_blurred_pixelate.jpg   - Pixelation applied")
    print("  4_blurred_strong.jpg     - Strong blur applied")
    print("  face_detections.json     - Detection metadata")
    print()
    print(f"Summary: Detected {num_faces} face(s)")
    print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_sam3_face_detection.py <image_path>")
        print()
        print("Example:")
        print("  python test_sam3_face_detection.py test_image.jpg")
        print()
        print("This will:")
        print("  1. Detect faces using SAM3")
        print("  2. Show detections with bounding boxes")
        print("  3. Save blurred versions")
        print("  4. Generate detection JSON")
        sys.exit(1)

    image_path = sys.argv[1]

    if not Path(image_path).exists():
        print(f"ERROR: Image not found: {image_path}")
        sys.exit(1)

    test_sam3_face_detection(image_path)


if __name__ == "__main__":
    main()
