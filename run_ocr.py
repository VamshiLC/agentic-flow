#!/usr/bin/env python3
"""
License Plate OCR - Command Line Interface

Detect and read vehicle license plates using Qwen3-VL + SAM3.
Optimized for North American plate formats (US, Canada, Mexico).

Usage:
    # Single image
    python run_ocr.py --input vehicle.jpg --output ./results

    # Video processing
    python run_ocr.py --input dashcam.mp4 --output ./results --fps 1

    # With quantization (lower memory)
    python run_ocr.py --input video.mp4 --output ./results --quantize

    # Without SAM3 masks (faster)
    python run_ocr.py --input image.jpg --output ./results --no-sam3
"""

import argparse
import sys
from pathlib import Path

from ocr import LicensePlateOCR
from ocr.processor import process_image, process_video, process_batch_images


def main():
    parser = argparse.ArgumentParser(
        description="License Plate OCR - Detect and read vehicle plates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_ocr.py --input car.jpg --output ./results
  python run_ocr.py --input dashcam.mp4 --output ./results --fps 2
  python run_ocr.py --input video.mp4 --output ./results --quantize --no-sam3
        """
    )

    # Required arguments
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input image or video file path"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for results"
    )

    # Processing options
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Target FPS for video processing (default: 1.0)"
    )
    parser.add_argument(
        "--single-stage",
        action="store_true",
        help="Use single-stage detection+OCR (faster but less accurate)"
    )

    # Memory options
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Use 8-bit quantization (reduces memory by ~50%%)"
    )
    parser.add_argument(
        "--low-memory",
        action="store_true",
        help="Enable low memory mode"
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        help="Device to use (default: auto-detect)"
    )

    # Output options
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Don't save annotated video (video input only)"
    )
    parser.add_argument(
        "--no-frames",
        action="store_true",
        help="Don't save individual frames (video input only)"
    )

    args = parser.parse_args()

    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    # Determine input type
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    suffix = input_path.suffix.lower()

    if suffix in video_extensions:
        input_type = "video"
    elif suffix in image_extensions:
        input_type = "image"
    else:
        print(f"Error: Unsupported file type: {suffix}")
        print(f"Supported: {video_extensions | image_extensions}")
        sys.exit(1)

    # Initialize OCR agent
    print("\nInitializing License Plate OCR...")
    ocr_agent = LicensePlateOCR(
        device=args.device,
        use_quantization=args.quantize,
        low_memory=args.low_memory,
        two_stage=not args.single_stage
    )

    # Process based on input type
    if input_type == "video":
        print(f"\nProcessing video: {args.input}")
        result = process_video(
            video_path=args.input,
            output_dir=args.output,
            target_fps=args.fps,
            ocr_agent=ocr_agent,
            save_frames=not args.no_frames,
            save_video=not args.no_video,
            save_json=True
        )
    else:
        print(f"\nProcessing image: {args.input}")
        result = process_image(
            image_path=args.input,
            output_dir=args.output,
            ocr_agent=ocr_agent,
            save_annotated=True,
            save_json=True
        )

    # Cleanup
    ocr_agent.cleanup()

    print("\nDone!")


if __name__ == "__main__":
    main()
