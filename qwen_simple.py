#!/usr/bin/env python3
"""
Simple Qwen Face Blur + Detection - Debug Version
Shows exactly what Qwen outputs
"""
import sys
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from models.qwen_direct_loader import Qwen3VLDirectDetector


def blur_region(image, x1, y1, x2, y2, strength=51):
    """Blur a specific region."""
    output = image.copy()
    h, w = image.shape[:2]

    # Ensure valid coordinates
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        return output

    # Add padding
    pad = 20
    x1_pad = max(0, x1 - pad)
    y1_pad = max(0, y1 - pad)
    x2_pad = min(w, x2 + pad)
    y2_pad = min(h, y2 + pad)

    region = output[y1_pad:y2_pad, x1_pad:x2_pad]
    if region.size == 0:
        return output

    # Make sure kernel size is odd
    if strength % 2 == 0:
        strength += 1

    blurred = cv2.GaussianBlur(region, (strength, strength), 0)
    output[y1_pad:y2_pad, x1_pad:x2_pad] = blurred

    return output


def main():
    if len(sys.argv) < 2:
        print("Usage: python qwen_simple.py <image.jpg>")
        sys.exit(1)

    input_path = sys.argv[1]
    if not Path(input_path).exists():
        print(f"ERROR: File not found: {input_path}")
        sys.exit(1)

    print("="*70)
    print("QWEN SIMPLE PIPELINE - DEBUG MODE")
    print("="*70)
    print(f"Input: {input_path}\n")

    # Read image
    image_cv = cv2.imread(input_path)
    if image_cv is None:
        print("ERROR: Could not read image")
        sys.exit(1)

    image_pil = Image.open(input_path).convert('RGB')
    h, w = image_cv.shape[:2]
    print(f"Image size: {w}x{h}\n")

    # Create output dir
    output_dir = Path("qwen_output")
    output_dir.mkdir(exist_ok=True)

    # Save original
    cv2.imwrite(str(output_dir / "0_original.jpg"), image_cv)

    # Load Qwen
    print("Loading Qwen2.5-VL-7B...")
    try:
        detector = Qwen3VLDirectDetector(
            model_name="Qwen/Qwen2.5-VL-7B-Instruct",
            use_quantization=False
        )
        print("✓ Qwen loaded\n")
    except Exception as e:
        print(f"✗ ERROR loading Qwen: {e}")
        sys.exit(1)

    # STEP 1: Simple face prompt
    print("="*70)
    print("STEP 1: FACE DETECTION")
    print("="*70)

    face_prompt = "Look at this image. Are there any human faces? If yes, tell me the approximate pixel coordinates of each face in format: x1,y1,x2,y2"

    print("Asking Qwen to detect faces...\n")
    result = detector.detect(image_pil, face_prompt, max_new_tokens=512)

    if result.get('success'):
        response = result.get('text', '')
        print("QWEN RESPONSE:")
        print("-"*70)
        print(response)
        print("-"*70)
        print()

        # Try to find any numbers that look like coordinates
        import re
        # Look for patterns like: 100,200,300,400 or (100, 200, 300, 400) or [100, 200, 300, 400]
        numbers = re.findall(r'\d+', response)

        if len(numbers) >= 4:
            print(f"Found numbers: {numbers}")

            # Take first 4 numbers as a test
            try:
                x1, y1, x2, y2 = map(int, numbers[:4])
                print(f"\nTrying to blur region: [{x1}, {y1}, {x2}, {y2}]")

                blurred = blur_region(image_cv, x1, y1, x2, y2)
                cv2.imwrite(str(output_dir / "1_face_blurred.jpg"), blurred)
                print("✓ Saved blurred image")
            except Exception as e:
                print(f"✗ Could not blur: {e}")
                blurred = image_cv.copy()
        else:
            print("⚠ No coordinates found in response")
            blurred = image_cv.copy()
    else:
        print("✗ Qwen detection failed")
        blurred = image_cv.copy()

    # STEP 2: Crosswalk detection
    print("\n" + "="*70)
    print("STEP 2: CROSSWALK DETECTION")
    print("="*70)

    crosswalk_prompt = "Look at this road image. Are there any damaged, faded, or worn crosswalks (pedestrian crossing markings)? If yes, describe where they are located."

    print("Asking Qwen to detect damaged crosswalks...\n")

    blurred_pil = Image.fromarray(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
    result2 = detector.detect(blurred_pil, crosswalk_prompt, max_new_tokens=512)

    if result2.get('success'):
        response2 = result2.get('text', '')
        print("QWEN RESPONSE:")
        print("-"*70)
        print(response2)
        print("-"*70)
        print()

        # Try to find coordinates
        numbers2 = re.findall(r'\d+', response2)

        if len(numbers2) >= 4:
            print(f"Found numbers: {numbers2}")

            try:
                x1, y1, x2, y2 = map(int, numbers2[:4])
                print(f"\nDrawing box at: [{x1}, {y1}, {x2}, {y2}]")

                annotated = blurred.copy()
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(annotated, "Damaged Crosswalk", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.imwrite(str(output_dir / "2_crosswalk_detected.jpg"), annotated)
                print("✓ Saved annotated image")
            except Exception as e:
                print(f"✗ Could not draw box: {e}")
        else:
            print("⚠ No coordinates found in response")
    else:
        print("✗ Crosswalk detection failed")

    print("\n" + "="*70)
    print("DONE")
    print("="*70)
    print(f"Check output in: {output_dir}/")
    print()

    detector.cleanup()


if __name__ == "__main__":
    main()
