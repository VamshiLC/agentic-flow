#!/usr/bin/env python3
"""
Simple Infrastructure Detection - NO vLLM Server Required

Uses direct Qwen3-VL loading (like your detector.py).
Simpler setup, works immediately without server.

Usage:
    # Single image
    python main_simple.py --mode image --input frame.jpg

    # Video
    python main_simple.py --mode video --input video.mp4 --fps 1.0
"""
import argparse
import sys
import os
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from detector_unified import UnifiedInfrastructureDetector, DEFECT_COLORS


def draw_detections(frame: np.ndarray, detections: list) -> np.ndarray:
    """Draw bounding boxes on frame."""
    annotated = frame.copy()

    for det in detections:
        label = det["label"]
        bbox = det["bbox"]
        x1, y1, x2, y2 = bbox
        color = det.get("color", (0, 255, 0))
        confidence = det.get("confidence", 0.0)

        # Draw bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label_text = f"{label}: {confidence:.2f}"
        font_scale = 0.5
        thickness = 1

        (text_w, text_h), _ = cv2.getTextSize(
            label_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness
        )

        # Label background
        cv2.rectangle(
            annotated,
            (x1, y1 - text_h - 10),
            (x1 + text_w + 10, y1),
            color,
            -1
        )

        # Label text
        cv2.putText(
            annotated,
            label_text,
            (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness
        )

    return annotated


def process_image(image_path: str, detector, output_dir: str):
    """Process single image."""
    print(f"\nProcessing image: {image_path}")

    # Load image
    image = Image.open(image_path).convert('RGB')

    # Run detection
    result = detector.detect_infrastructure(image)

    print(f"  Detections: {result['num_detections']}")
    if result['num_detections'] > 0:
        for det in result['detections']:
            print(f"    - {det['label']}: {det['bbox']}")

    # Save annotated image
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to numpy for drawing
    frame = np.array(image)
    annotated = draw_detections(frame, result['detections'])

    # Save
    output_path = output_dir / f"{Path(image_path).stem}_detected.jpg"
    cv2.imwrite(str(output_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    print(f"  Saved: {output_path}")

    # Save JSON
    json_path = output_dir / f"{Path(image_path).stem}_detections.json"
    with open(json_path, 'w') as f:
        json.dump({
            "image": image_path,
            "num_detections": result['num_detections'],
            "detections": [
                {
                    "label": d["label"],
                    "bbox": d["bbox"],
                    "confidence": d["confidence"]
                }
                for d in result['detections']
            ],
            "response": result['text_response']
        }, f, indent=2)

    return result


def process_video(video_path: str, detector, output_dir: str, target_fps: float = 1.0):
    """Process video."""
    print(f"\nProcessing video: {video_path}")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"  Video FPS: {video_fps}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Total frames: {total_frames}")
    print(f"  Target processing FPS: {target_fps}")

    # Calculate frame interval
    frame_interval = max(1, int(video_fps / target_fps))
    print(f"  Processing every {frame_interval} frames")

    # Create output directory
    output_dir = Path(output_dir)
    video_name = Path(video_path).stem
    video_output_dir = output_dir / video_name
    video_output_dir.mkdir(parents=True, exist_ok=True)

    # Process frames
    all_detections = []
    annotated_frames = []
    frame_idx = 0
    processed_count = 0

    print("\nProcessing frames...")
    pbar = tqdm(total=total_frames)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process every Nth frame
        if frame_idx % frame_interval == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            # Run detection
            result = detector.detect_infrastructure(pil_image)

            # Store results
            all_detections.append({
                "frame_index": frame_idx,
                "timestamp": frame_idx / video_fps,
                "num_detections": result['num_detections'],
                "detections": result['detections']
            })

            # Draw annotations
            annotated = draw_detections(frame_rgb, result['detections'])
            annotated_frames.append(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

            processed_count += 1

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    print(f"\nProcessed {processed_count} frames")

    # Save annotated video
    if annotated_frames:
        output_video_path = video_output_dir / f"{video_name}_detected.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            str(output_video_path),
            fourcc,
            target_fps,
            (width, height)
        )

        for frame in annotated_frames:
            out.write(frame)

        out.release()
        print(f"Saved annotated video: {output_video_path}")

    # Save JSON
    json_path = video_output_dir / f"{video_name}_detections.json"
    with open(json_path, 'w') as f:
        json.dump({
            "video": video_path,
            "video_fps": video_fps,
            "processed_fps": target_fps,
            "total_frames": total_frames,
            "processed_frames": processed_count,
            "frame_detections": all_detections
        }, f, indent=2)

    print(f"Saved detections JSON: {json_path}")

    # Summary
    total_dets = sum(d['num_detections'] for d in all_detections)
    frames_with_dets = sum(1 for d in all_detections if d['num_detections'] > 0)

    print(f"\nSummary:")
    print(f"  Total detections: {total_dets}")
    print(f"  Frames with detections: {frames_with_dets}/{processed_count}")


def main():
    parser = argparse.ArgumentParser(
        description="Simple Infrastructure Detection (No vLLM Server)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single image
  python main_simple.py --mode image --input frame.jpg

  # Process video at 1 FPS
  python main_simple.py --mode video --input video.mp4 --fps 1.0

  # Custom categories
  python main_simple.py --mode image --input frame.jpg --categories potholes alligator_cracks

  # GPU/CPU selection
  python main_simple.py --mode image --input frame.jpg --device cuda
        """
    )

    parser.add_argument(
        "--mode",
        choices=["image", "video"],
        required=True,
        help="Processing mode"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input image or video path"
    )
    parser.add_argument(
        "--output",
        default="output_simple",
        help="Output directory (default: output_simple)"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Target FPS for video processing (default: 2.0)"
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        help="Categories to detect (default: all)"
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        help="Device to use (default: auto-detect)"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-VL-4B-Instruct",
        help="Model to use (default: Qwen3-VL-4B-Instruct)"
    )

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1

    # Initialize detector (direct mode - no server)
    print("="*60)
    print("INFRASTRUCTURE DETECTION (DIRECT MODE)")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Device: {args.device or 'auto-detect'}")

    try:
        detector = UnifiedInfrastructureDetector(
            mode="direct",
            model_name=args.model,
            categories=args.categories,
            device=args.device
        )
    except Exception as e:
        print(f"\nError loading detector: {e}")
        return 1

    print("\nCategories to detect:")
    for cat in detector.categories:
        print(f"  - {cat}")

    # Process based on mode
    try:
        if args.mode == "image":
            process_image(args.input, detector, args.output)
        elif args.mode == "video":
            process_video(args.input, detector, args.output, args.fps)

        print("\n" + "="*60)
        print("PROCESSING COMPLETE")
        print("="*60)
        return 0

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
