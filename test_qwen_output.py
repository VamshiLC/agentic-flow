#!/usr/bin/env python3
"""
Test what Qwen actually outputs
Just shows the raw responses
"""
import sys
from pathlib import Path
from PIL import Image
from models.qwen_direct_loader import Qwen3VLDirectDetector


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_qwen_output.py <image.jpg>")
        sys.exit(1)

    input_path = sys.argv[1]
    if not Path(input_path).exists():
        print(f"ERROR: File not found: {input_path}")
        sys.exit(1)

    print("="*70)
    print("TESTING QWEN OUTPUT")
    print("="*70)
    print(f"Image: {input_path}\n")

    # Load image
    image = Image.open(input_path).convert('RGB')
    w, h = image.size
    print(f"Size: {w}x{h}\n")

    # Load Qwen
    print("Loading Qwen...")
    detector = Qwen3VLDirectDetector(
        model_name="Qwen/Qwen2.5-VL-7B-Instruct"
    )
    print("✓ Loaded\n")

    # Test 1: Simple question
    print("="*70)
    print("TEST 1: Are there faces?")
    print("="*70)
    prompt1 = "Are there any human faces in this image? Answer yes or no, and if yes, how many?"

    result1 = detector.detect(image, prompt1)
    if result1.get('success'):
        print("\nQWEN SAYS:")
        print(result1.get('text', ''))
    else:
        print("FAILED")

    # Test 2: Face coordinates
    print("\n" + "="*70)
    print("TEST 2: Face coordinates")
    print("="*70)
    prompt2 = """Detect all human faces in this image.
For each face, provide the bounding box coordinates in this EXACT format:
Face 1: x1=VALUE, y1=VALUE, x2=VALUE, y2=VALUE

Where x1,y1 is top-left corner and x2,y2 is bottom-right corner in pixels.
The image is {w} pixels wide and {h} pixels tall.""".format(w=w, h=h)

    result2 = detector.detect(image, prompt2, max_new_tokens=512)
    if result2.get('success'):
        print("\nQWEN SAYS:")
        print(result2.get('text', ''))
    else:
        print("FAILED")

    # Test 3: Crosswalk
    print("\n" + "="*70)
    print("TEST 3: Damaged crosswalks")
    print("="*70)
    prompt3 = "Are there any damaged, faded, or worn crosswalks in this road image? Describe what you see."

    result3 = detector.detect(image, prompt3)
    if result3.get('success'):
        print("\nQWEN SAYS:")
        print(result3.get('text', ''))
    else:
        print("FAILED")

    print("\n" + "="*70)
    print("DONE")
    print("="*70)

    detector.cleanup()


if __name__ == "__main__":
    main()
