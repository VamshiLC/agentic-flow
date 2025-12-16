#!/usr/bin/env python3
"""
SAM3-Only Infrastructure Detection
Main entry point for SAM3-only detection (without Qwen agentic flow).

Usage:
    # Single image
    python main_sam3_only.py --mode image --input path/to/image.jpg

    # Video
    python main_sam3_only.py --mode video --input path/to/video.mp4 --fps 2.0

    # Specific categories only
    python main_sam3_only.py --mode image --input image.jpg --categories potholes alligator_cracks

    # Low confidence threshold (more detections)
    python main_sam3_only.py --mode video --input video.mp4 --confidence 0.2
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import torch
import cv2
import numpy as np

# Import SAM3 components
from models.sam3_text_prompt_loader import load_sam3_text_prompt_model
from inference.sam3_only_single_frame import create_sam3_only_processor
from inference.sam3_only_video import create_sam3_only_video_processor

# Import existing visualization and utilities
from visualization_styles import draw_stylish_detections
from utils.output_formatter import format_detections_json
from prompts.category_prompts import CATEGORY_PROMPTS


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='SAM3-Only Infrastructure Detection (no Qwen)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Detect all categories in an image
  python main_sam3_only.py --mode image --input frame.jpg

  # Process video at 2 FPS
  python main_sam3_only.py --mode video --input video.mp4 --fps 2.0

  # Detect only potholes and cracks
  python main_sam3_only.py --mode image --input image.jpg \\
      --categories potholes alligator_cracks longitudinal_cracks

  # Process with lower confidence threshold
  python main_sam3_only.py --mode video --input video.mp4 --confidence 0.2

  # Limit video processing to first 50 frames
  python main_sam3_only.py --mode video --input video.mp4 --max-frames 50
        """
    )

    parser.add_argument(
        '--mode',
        choices=['image', 'video'],
        required=True,
        help='Processing mode: image or video'
    )

    parser.add_argument(
        '--input',
        required=True,
        help='Input image or video file path'
    )

    parser.add_argument(
        '--output-dir',
        default='output_sam3_only',
        help='Output directory (default: output_sam3_only)'
    )

    parser.add_argument(
        '--categories',
        nargs='+',
        default=None,
        help='Specific categories to detect (default: all)'
    )

    parser.add_argument(
        '--confidence',
        type=float,
        default=0.3,
        help='Confidence threshold (default: 0.3)'
    )

    parser.add_argument(
        '--fps',
        type=float,
        default=None,
        help='Target FPS for video processing (default: original FPS)'
    )

    parser.add_argument(
        '--max-frames',
        type=int,
        default=None,
        help='Maximum frames to process for video (default: all)'
    )

    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Skip visualization generation'
    )

    parser.add_argument(
        '--device',
        default=None,
        help='Device to use (cuda/cpu, default: auto)'
    )

    parser.add_argument(
        '--list-categories',
        action='store_true',
        help='List all available categories and exit'
    )

    return parser.parse_args()


def list_categories():
    """List all available detection categories."""
    print("\n" + "="*60)
    print("AVAILABLE DETECTION CATEGORIES")
    print("="*60)

    for category, info in CATEGORY_PROMPTS.items():
        short_name = info.get("short_name", category)
        description = info.get("description", "No description")
        print(f"\n{category}")
        print(f"  Prompt: {short_name}")
        print(f"  Description: {description}")

    print("\n" + "="*60)
    print(f"Total categories: {len(CATEGORY_PROMPTS)}")
    print("="*60 + "\n")


def process_image_mode(args, model, processor):
    """Process single image and organize by category."""
    print("\n" + "="*60)
    print("SAM3-ONLY IMAGE DETECTION")
    print("="*60)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create processor
    frame_processor = create_sam3_only_processor(
        model=model,
        processor=processor,
        categories=args.categories,
        confidence_threshold=args.confidence,
        device=args.device or "cuda",
    )

    # Process image
    print(f"\nProcessing: {args.input}")
    result = frame_processor.process_image(args.input)

    # Load original image
    original_image = cv2.imread(str(args.input))
    if original_image is None:
        from PIL import Image
        original_image = np.array(Image.open(args.input).convert('RGB'))
        original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)

    # Print summary
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Total Detections: {result['num_detections']}")

    # Organize detections by category
    if result['detections']:
        detections_by_category = {}
        for det in result['detections']:
            cat = det['category']
            if cat not in detections_by_category:
                detections_by_category[cat] = []
            detections_by_category[cat].append(det)

        print(f"\nDetections by category:")
        for cat, dets in sorted(detections_by_category.items()):
            print(f"  {cat}: {len(dets)}")

        # Save detections organized by category
        input_path = Path(args.input)
        base_name = input_path.stem

        for category, category_dets in detections_by_category.items():
            # Create category folder structure
            category_dir = output_dir / category
            category_dir.mkdir(parents=True, exist_ok=True)
            frames_dir = category_dir / "frames"
            frames_dir.mkdir(exist_ok=True)
            annotated_dir = category_dir / "annotated"
            annotated_dir.mkdir(exist_ok=True)
            crops_dir = category_dir / "crops"
            crops_dir.mkdir(exist_ok=True)

            # Save original image with detection
            frame_path = frames_dir / f"{base_name}.jpg"
            cv2.imwrite(str(frame_path), original_image)

            # Save annotated image with only this category
            annotated_image = draw_stylish_detections(
                image=original_image.copy(),
                detections=category_dets,
                show_masks=True,
            )
            annotated_path = annotated_dir / f"{base_name}_detected.jpg"
            cv2.imwrite(str(annotated_path), annotated_image)

            # Save cropped detections
            for idx, det in enumerate(category_dets):
                bbox = det.get('bbox', [])
                if len(bbox) == 4:
                    x1, y1, x2, y2 = map(int, bbox)
                    # Add padding
                    pad = 10
                    h, w = original_image.shape[:2]
                    x1 = max(0, x1 - pad)
                    y1 = max(0, y1 - pad)
                    x2 = min(w, x2 + pad)
                    y2 = min(h, y2 + pad)

                    crop = original_image[y1:y2, x1:x2]
                    crop_path = crops_dir / f"{base_name}_crop_{idx:03d}.jpg"
                    cv2.imwrite(str(crop_path), crop)

            # Save category-specific JSON
            json_path = category_dir / "detections.json"
            category_result = {
                "category": category,
                "image_path": str(args.input),
                "image_name": base_name,
                "detection_time": datetime.now().isoformat(),
                "num_detections": len(category_dets),
                "detections": [{k: v for k, v in d.items() if k != 'mask'} for d in category_dets]
            }
            with open(json_path, 'w') as f:
                json.dump(category_result, f, indent=2)

            print(f"\n✓ {category}:")
            print(f"  - Original: {frame_path}")
            print(f"  - Annotated: {annotated_path}")
            print(f"  - Crops: {len(category_dets)} saved to {crops_dir}")
            print(f"  - JSON: {json_path}")

    else:
        print("\nNo detections found.")

    print(f"\n{'='*60}\n")


def process_video_mode(args, model, processor):
    """Process video and organize by category."""
    print("\n" + "="*60)
    print("SAM3-ONLY VIDEO DETECTION")
    print("="*60)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create video processor
    video_processor = create_sam3_only_video_processor(
        model=model,
        processor=processor,
        categories=args.categories,
        confidence_threshold=args.confidence,
        device=args.device or "cuda",
    )

    # Process video
    result = video_processor.process_video(
        video_path=args.input,
        target_fps=args.fps,
        max_frames=args.max_frames,
    )

    # Open video to extract frames
    cap = cv2.VideoCapture(str(args.input))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or result['fps']
    video_name = Path(args.input).stem

    # Print summary
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Total frames: {result['total_frames']}")
    print(f"FPS: {result['fps']:.1f}")
    print(f"Total detections: {result['num_detections']}")

    if result['detections']:
        # Group detections by category and frame
        detections_by_category = {}
        for det in result['detections']:
            cat = det['category']
            if cat not in detections_by_category:
                detections_by_category[cat] = {}
            frame_num = det['frame_number']
            if frame_num not in detections_by_category[cat]:
                detections_by_category[cat][frame_num] = []
            detections_by_category[cat][frame_num].append(det)

        print(f"\nDetections by category:")
        for cat in sorted(detections_by_category.keys()):
            frames_count = len(detections_by_category[cat])
            dets_count = sum(len(dets) for dets in detections_by_category[cat].values())
            print(f"  {cat}: {dets_count} detections in {frames_count} frames")

        # Save detections organized by category
        for category, frames_dict in detections_by_category.items():
            # Create category folder structure
            category_dir = output_dir / category
            category_dir.mkdir(parents=True, exist_ok=True)
            frames_dir = category_dir / "frames"
            frames_dir.mkdir(exist_ok=True)
            annotated_dir = category_dir / "annotated"
            annotated_dir.mkdir(exist_ok=True)

            category_detections = []

            # Extract and save only frames with this category
            for frame_num, frame_dets in frames_dict.items():
                # Seek to frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                if not ret:
                    continue

                # Save original frame
                frame_path = frames_dir / f"frame_{frame_num:06d}.jpg"
                cv2.imwrite(str(frame_path), frame)

                # Save annotated frame (only this category)
                annotated_frame = draw_stylish_detections(
                    image=frame.copy(),
                    detections=frame_dets,
                    show_masks=True,
                )
                annotated_path = annotated_dir / f"frame_{frame_num:06d}_detected.jpg"
                cv2.imwrite(str(annotated_path), annotated_frame)

                # Store detection info
                category_detections.append({
                    "frame_index": frame_num,
                    "timestamp": frame_num / video_fps,
                    "frame_path": str(frame_path),
                    "annotated_path": str(annotated_path),
                    "detections": [{k: v for k, v in d.items() if k != 'mask'} for d in frame_dets]
                })

            # Save category-specific JSON
            json_path = category_dir / "detections.json"
            category_result = {
                "category": category,
                "video_path": str(args.input),
                "video_name": video_name,
                "detection_time": datetime.now().isoformat(),
                "total_frames_processed": result['total_frames'],
                "frames_with_detections": len(frames_dict),
                "total_detections": sum(len(dets) for dets in frames_dict.values()),
                "fps": result['fps'],
                "detections": category_detections
            }
            with open(json_path, 'w') as f:
                json.dump(category_result, f, indent=2)

            print(f"\n✓ {category}:")
            print(f"  - Frames: {len(frames_dict)} saved to {frames_dir}")
            print(f"  - Annotated: {len(frames_dict)} saved to {annotated_dir}")
            print(f"  - JSON: {json_path}")

        # Save overall summary
        summary_path = output_dir / "summary.json"
        summary = {}
        for cat, frames_dict in detections_by_category.items():
            summary[cat] = {
                "frames_with_detections": len(frames_dict),
                "total_detections": sum(len(dets) for dets in frames_dict.values())
            }
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n✓ Summary: {summary_path}")

    else:
        print("\nNo detections found.")

    cap.release()
    print(f"\n{'='*60}\n")


def save_json_output(result, output_path):
    """Save detection results to JSON."""
    # Remove mask arrays (too large for JSON)
    result_copy = {
        "video_path": result.get("video_path", result.get("image_path", "")),
        "video_name": result.get("video_name", Path(result.get("image_path", "")).stem),
        "detection_time": datetime.now().isoformat(),
        "total_frames": result.get("total_frames", 1),
        "fps": result.get("fps", 1.0),
        "num_detections": result["num_detections"],
        "categories_searched": result.get("categories_searched", []),
        "detections": []
    }

    for det in result['detections']:
        det_copy = {k: v for k, v in det.items() if k != 'mask'}
        result_copy['detections'].append(det_copy)

    with open(output_path, 'w') as f:
        json.dump(result_copy, f, indent=2)


def visualize_detections(image_path, detections, output_path):
    """Create visualization using existing styling."""
    import cv2
    from PIL import Image

    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        image = np.array(Image.open(image_path).convert('RGB'))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw detections using existing visualization
    annotated = draw_stylish_detections(
        image=image.copy(),
        detections=detections,
        show_masks=True,
    )

    # Save
    cv2.imwrite(str(output_path), annotated)


def create_annotated_video(input_video_path, result, output_path):
    """Create annotated video."""
    # Load original video
    cap = cv2.VideoCapture(str(input_video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or result['fps']
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Group detections by frame
    detections_by_frame = {}
    for det in result['detections']:
        frame_num = det['frame_number']
        if frame_num not in detections_by_frame:
            detections_by_frame[frame_num] = []
        detections_by_frame[frame_num].append(det)

    # Process frames
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw detections for this frame
        if frame_idx in detections_by_frame:
            frame = draw_stylish_detections(
                image=frame,
                detections=detections_by_frame[frame_idx],
                show_masks=True,
            )

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()


def main():
    """Main entry point."""
    args = parse_args()

    # List categories if requested
    if args.list_categories:
        list_categories()
        sys.exit(0)

    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    # Check CUDA
    if torch.cuda.is_available():
        print(f"\n✓ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("\n⚠ CUDA not available, using CPU (this will be slow)")

    # Load SAM3 model
    print("\nLoading SAM3 model...")
    model, processor, loader = load_sam3_text_prompt_model(
        device=args.device,
        dtype=torch.float32,
    )

    # Process based on mode
    if args.mode == 'image':
        process_image_mode(args, model, processor)
    elif args.mode == 'video':
        process_video_mode(args, model, processor)

    print("Done!")


if __name__ == '__main__':
    main()
