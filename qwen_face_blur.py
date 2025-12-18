#!/usr/bin/env python3
"""
Qwen2.5-VL-7B Face Detection + Gaussian Blur

Uses your existing Qwen model for face detection, then applies Gaussian blur.
Much faster than SAM3 since Qwen is already loaded for infrastructure detection.
"""
import sys
import re
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from models.qwen_direct_loader import Qwen3VLDirectDetector


FACE_DETECTION_PROMPT = """Analyze this image and detect all human faces.

For each face you detect, provide the output in this exact format:
Face: <number>, Box: [x1, y1, x2, y2], Confidence: <0.0-1.0>

Where:
- x1, y1 are the top-left coordinates
- x2, y2 are the bottom-right coordinates
- Confidence is your certainty (0.0 to 1.0)

Important:
- Detect ALL faces visible in the image
- Include partially visible faces
- Include faces at any angle or size
- Use pixel coordinates relative to the image size

If NO faces are detected, respond with: "No faces detected"

Examples:
Face: 1, Box: [120, 80, 250, 210], Confidence: 0.95
Face: 2, Box: [450, 120, 560, 230], Confidence: 0.87

Now analyze the image:"""


def parse_face_detections(response, image_size, confidence_threshold=0.5):
    """
    Parse face detections from Qwen text response.

    Args:
        response: Text response from Qwen
        image_size: (width, height) tuple
        confidence_threshold: Minimum confidence to accept

    Returns:
        list: List of bounding boxes [(x1, y1, x2, y2), ...]
    """
    bboxes = []
    width, height = image_size

    if "no faces detected" in response.lower():
        return bboxes

    # Pattern: Face: <number>, Box: [x1, y1, x2, y2], Confidence: <score>
    pattern = r'Face:\s*\d+,\s*Box:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\],\s*Confidence:\s*([\d.]+)'
    matches = re.findall(pattern, response, re.IGNORECASE)

    for match in matches:
        try:
            x1, y1, x2, y2, conf = map(float, match)

            # Filter by confidence
            if conf < confidence_threshold:
                continue

            # Validate coordinates
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Ensure within image bounds
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))

            # Ensure valid box
            if x2 > x1 and y2 > y1:
                bboxes.append((x1, y1, x2, y2))

        except (ValueError, IndexError) as e:
            print(f"  Warning: Failed to parse detection: {match}")
            continue

    return bboxes


def blur_faces(image, bboxes, strength=51, padding=20):
    """Apply Gaussian blur to detected face regions."""
    output = image.copy()
    h, w = image.shape[:2]

    # Ensure odd kernel size
    if strength % 2 == 0:
        strength += 1

    for (x1, y1, x2, y2) in bboxes:
        # Add padding
        x1_pad = max(0, x1 - padding)
        y1_pad = max(0, y1 - padding)
        x2_pad = min(w, x2 + padding)
        y2_pad = min(h, y2 + padding)

        face_region = output[y1_pad:y2_pad, x1_pad:x2_pad]

        if face_region.size == 0:
            continue

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(face_region, (strength, strength), 0)
        output[y1_pad:y2_pad, x1_pad:x2_pad] = blurred

    return output


def main():
    if len(sys.argv) < 2:
        print("Usage: python qwen_face_blur.py <input_image> [output_image] [blur_strength]")
        print()
        print("Examples:")
        print("  python qwen_face_blur.py peoples1.png")
        print("  python qwen_face_blur.py peoples1.png result.jpg")
        print("  python qwen_face_blur.py peoples1.png result.jpg 71")
        print()
        print("Note: Uses your existing Qwen2.5-VL-7B model")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    blur_strength = int(sys.argv[3]) if len(sys.argv) > 3 else 51

    if not Path(input_path).exists():
        print(f"ERROR: File not found: {input_path}")
        sys.exit(1)

    print("=" * 70)
    print("Qwen2.5-VL-7B Face Detection + Gaussian Blur")
    print("=" * 70)
    print(f"Input: {input_path}")
    print(f"Blur strength: {blur_strength}\n")

    # Read image
    image_cv = cv2.imread(input_path)
    if image_cv is None:
        print("ERROR: Could not read image")
        sys.exit(1)

    image_pil = Image.open(input_path).convert('RGB')
    h, w = image_cv.shape[:2]
    print(f"Image size: {w}x{h}")

    # Load Qwen detector
    print("\n1. Loading Qwen2.5-VL-7B model...")
    print("   (This uses your existing infrastructure detection model)")

    try:
        detector = Qwen3VLDirectDetector(
            model_name="Qwen/Qwen2.5-VL-7B-Instruct",
            use_quantization=False,  # Set True if you want to save GPU memory
            low_memory=False
        )
        print("   ✓ Qwen loaded")
    except Exception as e:
        print(f"   ✗ ERROR: {e}")
        print("\nMake sure Qwen model is available:")
        print("  pip install transformers torch pillow")
        sys.exit(1)

    # Detect faces
    print("\n2. Detecting faces with Qwen2.5-VL-7B...")
    print("   (Asking Qwen to find and localize all faces)")

    try:
        result = detector.detect(image_pil, FACE_DETECTION_PROMPT, max_new_tokens=512)

        if not result.get('success', False):
            print("   ✗ Qwen detection failed")
            sys.exit(1)

        response_text = result.get('text', '')
        print(f"\n   Qwen's response:")
        print(f"   {'-' * 60}")
        print(f"   {response_text}")
        print(f"   {'-' * 60}")

        # Parse bounding boxes
        bboxes = parse_face_detections(response_text, (w, h), confidence_threshold=0.5)
        num_faces = len(bboxes)

        print(f"\n   ✓ Detected {num_faces} face(s)")

        if num_faces == 0:
            print("\n   No faces detected. Saving original image.")
            if output_path is None:
                p = Path(input_path)
                output_path = p.parent / f"{p.stem}_blurred{p.suffix}"
            cv2.imwrite(str(output_path), image_cv)
            print(f"   Saved: {output_path}")
            return

        # Print bounding boxes
        print("\n   Face bounding boxes:")
        for i, (x1, y1, x2, y2) in enumerate(bboxes):
            print(f"     Face {i+1}: [{x1}, {y1}, {x2}, {y2}] - Size: {x2-x1}x{y2-y1}px")

    except Exception as e:
        print(f"   ✗ ERROR during detection: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Apply blur
    print(f"\n3. Applying Gaussian blur (strength={blur_strength})...")
    blurred_image = blur_faces(image_cv, bboxes, blur_strength, padding=20)

    # Save output
    if output_path is None:
        p = Path(input_path)
        output_path = p.parent / f"{p.stem}_blurred{p.suffix}"

    cv2.imwrite(str(output_path), blurred_image)
    print(f"   ✓ Saved blurred image: {output_path}")

    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)
    print(f"Result: {output_path}")
    print(f"Blurred {num_faces} face(s) using Qwen2.5-VL-7B")
    print()

    # Cleanup
    detector.cleanup()


if __name__ == "__main__":
    main()
