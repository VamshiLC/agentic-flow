#!/usr/bin/env python3
"""
Test Infrastructure Detection using Grounding DINO + SAM

This is the ACCURATE detector that uses:
- Grounding DINO: 52.5 AP on COCO zero-shot (best open-set detector)
- SAM: Precise segmentation masks

Usage:
    python test_grounding_dino.py --image image1.png
    python test_grounding_dino.py --image image1.png --prompt "pothole. manhole. graffiti."
    python test_grounding_dino.py --image image1.png --detect "pothole,manhole,crack"
"""
import argparse
import os
import sys
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Test Grounding DINO + SAM Detection")
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--prompt", type=str, help="Direct text prompt (use periods to separate: 'cat. dog.')")
    parser.add_argument("--detect", type=str, help="Comma-separated categories to detect")
    parser.add_argument("--threshold", type=float, default=0.35, help="Detection confidence threshold")
    parser.add_argument("--text-threshold", type=float, default=0.25, help="Text matching threshold")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    # Parse categories
    categories = None
    text_prompt = None

    if args.prompt:
        text_prompt = args.prompt
    elif args.detect:
        categories = [c.strip() for c in args.detect.split(",")]

    print("=" * 60)
    print("GROUNDING DINO + SAM DETECTION")
    print("=" * 60)
    print(f"Image: {args.image}")
    if text_prompt:
        print(f"Prompt: {text_prompt}")
    elif categories:
        print(f"Categories: {categories}")
    else:
        print("Mode: Full infrastructure detection")
    print(f"Thresholds: box={args.threshold}, text={args.text_threshold}")
    print("=" * 60)

    # Load image
    print("\nLoading image...")
    image = Image.open(args.image).convert("RGB")
    print(f"Image size: {image.size}")

    # Load detector
    print("\nLoading Grounding DINO + SAM detector...")
    from models.grounding_dino_sam import GroundingDINOSAMDetector

    detector = GroundingDINOSAMDetector(
        box_threshold=args.threshold,
        text_threshold=args.text_threshold
    )

    # Run detection
    print("\nRunning detection...")

    if text_prompt:
        result = detector.detect(image, text_prompt)
    elif categories:
        text_prompt = ". ".join(categories) + "."
        result = detector.detect(image, text_prompt)
    else:
        result = detector.detect_infrastructure(image)

    # Print results
    print("\n" + "=" * 60)
    print(f"RESULTS: Found {result['num_detections']} detections")
    print("=" * 60)

    for det in result["detections"]:
        print(f"  [{det['id']}] {det['label']}: {det['confidence']:.2f}")
        print(f"      bbox: {[round(x, 1) for x in det['bbox']]}")

    # Save annotated image
    output_path = os.path.join(args.output, os.path.basename(args.image).replace(".", "_detected."))
    result["annotated_image"].save(output_path)
    print(f"\nSaved: {output_path}")

    # Cleanup
    detector.cleanup()
    print("\nDone!")


if __name__ == "__main__":
    main()
