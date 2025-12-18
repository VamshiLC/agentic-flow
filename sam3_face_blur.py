#!/usr/bin/env python3
"""
SAM3 Face Blur - Simple and Clean
Uses official Meta SAM3 for face detection + Gaussian blur
"""
import sys
import cv2
import numpy as np
from pathlib import Path
from PIL import Image


def blur_faces(image, boxes, strength=51):
    """Apply Gaussian blur to detected face regions."""
    output = image.copy()

    # Ensure odd strength
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

        face_region = output[y1:y2, x1:x2]
        if face_region.size == 0:
            continue

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(face_region, (strength, strength), 0)
        output[y1:y2, x1:x2] = blurred

    return output


def main():
    if len(sys.argv) < 2:
        print("Usage: python sam3_face_blur.py <input_image> [output_image] [blur_strength]")
        print()
        print("Examples:")
        print("  python sam3_face_blur.py peoples1.png")
        print("  python sam3_face_blur.py peoples1.png result.jpg")
        print("  python sam3_face_blur.py peoples1.png result.jpg 71")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    blur_strength = int(sys.argv[3]) if len(sys.argv) > 3 else 51

    if not Path(input_path).exists():
        print(f"ERROR: File not found: {input_path}")
        sys.exit(1)

    print("=" * 60)
    print("SAM3 Face Detection + Gaussian Blur")
    print("=" * 60)
    print(f"Input: {input_path}")
    print(f"Blur strength: {blur_strength}\n")

    # Read image
    image_cv = cv2.imread(input_path)
    if image_cv is None:
        print("ERROR: Could not read image")
        sys.exit(1)

    print(f"Image size: {image_cv.shape[1]}x{image_cv.shape[0]}")

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
        print("\nMake sure SAM3 is installed:")
        print("  git clone https://github.com/facebookresearch/sam3.git")
        print("  cd sam3")
        print("  pip install -e .")
        print("\nAlso authenticate with Hugging Face:")
        print("  huggingface-cli login")
        sys.exit(1)

    # Detect faces
    print("\n2. Detecting faces with text prompt 'face'...")
    try:
        image_pil = Image.open(input_path)
        inference_state = processor.set_image(image_pil)
        output = processor.set_text_prompt(state=inference_state, prompt="face")

        boxes = output["boxes"]
        scores = output["scores"]

        # Filter by score
        good_boxes = [box for box, score in zip(boxes, scores) if score > 0.5]

        num_faces = len(good_boxes)
        print(f"   ✓ Detected {num_faces} face(s)")

        if num_faces == 0:
            print("\n   No faces detected. Saving original.")
            if output_path is None:
                p = Path(input_path)
                output_path = p.parent / f"{p.stem}_blurred{p.suffix}"
            cv2.imwrite(str(output_path), image_cv)
            print(f"   Saved: {output_path}")
            return

        # Print boxes
        print("\n   Bounding boxes:")
        for i, box in enumerate(good_boxes):
            x1, y1, x2, y2 = [int(v) for v in box]
            print(f"     Face {i+1}: [{x1}, {y1}, {x2}, {y2}] - Size: {x2-x1}x{y2-y1}px")

    except Exception as e:
        print(f"   ✗ ERROR: {e}")
        sys.exit(1)

    # Blur faces
    print(f"\n3. Applying Gaussian blur (strength={blur_strength})...")
    blurred = blur_faces(image_cv, good_boxes, blur_strength)

    # Save
    if output_path is None:
        p = Path(input_path)
        output_path = p.parent / f"{p.stem}_blurred{p.suffix}"

    cv2.imwrite(str(output_path), blurred)
    print(f"   ✓ Saved blurred image: {output_path}")

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"Result: {output_path}")
    print(f"Blurred {num_faces} face(s)")


if __name__ == "__main__":
    main()
