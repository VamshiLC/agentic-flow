#!/usr/bin/env python3
"""
Test Infrastructure Detection on Images

Usage:
    python test_image.py --image /path/to/image.jpg
    python test_image.py --image /path/to/image.jpg --output result.png
    python test_image.py --folder /path/to/images/
"""
import argparse
import os
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Test Infrastructure Detection")
    parser.add_argument("--image", type=str, help="Path to single image")
    parser.add_argument("--folder", type=str, help="Path to folder of images")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--quantization", action="store_true", help="Use 8-bit quantization (saves GPU memory)")
    parser.add_argument("--confidence", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    if not args.image and not args.folder:
        parser.error("Either --image or --folder is required")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("INFRASTRUCTURE DETECTION TEST")
    print("=" * 60)

    # Import and initialize detector
    from agent.detection_agent_hf import InfrastructureDetectionAgentHF
    from PIL import Image

    print("\nLoading detector...")
    detector = InfrastructureDetectionAgentHF(
        use_quantization=args.quantization,
        sam3_confidence=args.confidence,
        debug=True
    )

    # Get list of images
    if args.image:
        images = [args.image]
    else:
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        images = [
            str(p) for p in Path(args.folder).iterdir()
            if p.suffix.lower() in extensions
        ]
        print(f"Found {len(images)} images in folder")

    # Process each image
    for i, image_path in enumerate(images):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(images)}] Processing: {image_path}")
        print("=" * 60)

        try:
            # Run detection
            result = detector.detect_infrastructure(image_path)

            # Print results
            print(f"\n✓ Found {result['num_detections']} detections:")
            for det in result['detections']:
                category = det.get('category', 'unknown')
                confidence = det.get('confidence', 0)
                severity = det.get('severity', 'low')
                print(f"  - {category}: {confidence:.2f} ({severity})")

            # Save result image
            if result.get('final_image'):
                output_name = f"result_{Path(image_path).stem}.png"
                output_path = os.path.join(args.output, output_name)
                result['final_image'].save(output_path)
                print(f"\n✓ Saved: {output_path}")

        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("DONE!")
    print(f"Results saved to: {args.output}/")
    print("=" * 60)

    # Cleanup
    detector.cleanup()

if __name__ == "__main__":
    main()
