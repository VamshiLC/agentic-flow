#!/usr/bin/env python3
"""
License Plate OCR - Command Line Interface

Architecture:
- SAM3: Detection + Tracking (when available)
- Qwen3-VL: OCR (read text from cropped plates)
- IoU Tracker: Fallback when SAM3 native tracking unavailable

Optimized for North American plate formats (US, Canada, Mexico).

Usage:
    # Single image
    python run_ocr.py --input vehicle.jpg --output ./results

    # All images in a folder (batch processing)
    python run_ocr.py --input ./images --output ./ocr_output --quantize

    # Video processing with tracking
    python run_ocr.py --input dashcam.mp4 --output ./results --fps 1

    # With quantization (lower memory)
    python run_ocr.py --input video.mp4 --output ./results --quantize

    # Without SAM3 (use Qwen for detection)
    python run_ocr.py --input video.mp4 --output ./results --no-sam3
"""

import argparse
import sys
import glob
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
  python run_ocr.py --input video.mp4 --output ./results --quantize
  python run_ocr.py --input video.mp4 --output ./results --no-sam3
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
        "--no-sam3",
        action="store_true",
        help="Disable SAM3 (use Qwen3-VL for detection)"
    )
    parser.add_argument(
        "--no-tracking",
        action="store_true",
        help="Disable plate tracking across frames"
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

    # Determine input type
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    # Check if input is a folder
    if input_path.is_dir():
        input_type = "folder"
        # Find all images in folder
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(str(input_path / f"*{ext}")))
            image_files.extend(glob.glob(str(input_path / f"*{ext.upper()}")))
        image_files = sorted(image_files)

        if not image_files:
            print(f"Error: No images found in folder: {args.input}")
            print(f"Supported formats: {image_extensions}")
            sys.exit(1)
        print(f"Found {len(image_files)} images in folder")
    elif not input_path.exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    else:
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
        use_sam3=not args.no_sam3
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
            save_json=True,
            enable_tracking=not args.no_tracking,
            use_sam3=not args.no_sam3
        )
    elif input_type == "folder":
        print(f"\nProcessing {len(image_files)} images from: {args.input}")
        result = process_batch_images(
            image_paths=image_files,
            output_dir=args.output,
            ocr_agent=ocr_agent
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
