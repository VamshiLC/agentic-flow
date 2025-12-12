"""
Multi-Category Detection Pipeline

CORRECT FLOW:
1. Qwen3-VL detects ALL categories in ONE pass per frame
2. SAM3 segments EACH detection from Qwen
3. Save frame to MULTIPLE category folders (one per detected category)

Example:
  Frame has pothole + homeless person + car
  → Saved to potholes/, homeless_person/, and abandoned_vehicle/ folders
"""
import os
import cv2
import json
import logging
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from typing import List, Dict, Optional
import numpy as np

from detector_unified import UnifiedInfrastructureDetector, INFRASTRUCTURE_CATEGORIES, DEFECT_COLORS

logger = logging.getLogger(__name__)


def process_video_multi_detection(
    video_path: str,
    output_dir: str,
    target_fps: float = 2.0,
    device: Optional[str] = None,
    use_quantization: bool = False,
    low_memory: bool = False,
    batch_size: int = 1
):
    """
    Process video with multi-category detection per frame.

    Correct Flow:
    - Qwen3 analyzes frame → detects ALL categories at once
    - SAM3 segments each detection
    - Save frame to each category folder where detection found

    Args:
        video_path: Path to video file
        output_dir: Output directory
        target_fps: Target processing FPS
        device: Device to use
        use_quantization: Use 8-bit quantization
        low_memory: Enable low memory mode
        batch_size: Batch size for frame processing

    Output:
        results/
        ├── potholes/           (only frames with potholes)
        ├── homeless_person/    (only frames with homeless people)
        ├── abandoned_vehicle/  (only frames with vehicles)
        └── summary.json
    """
    video_path = str(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n{'='*70}")
    print(f"MULTI-CATEGORY DETECTION: Qwen3 + SAM3 Pipeline")
    print(f"{'='*70}")
    print(f"Video: {video_path}")
    print(f"Total frames: {total_frames}")
    print(f"Video FPS: {video_fps}")
    print(f"Processing FPS: {target_fps}")
    print(f"Categories: {len(INFRASTRUCTURE_CATEGORIES)}")
    print(f"{'='*70}\n")

    # Calculate frame sampling
    frame_interval = max(1, int(video_fps / target_fps))

    # Initialize detector with ALL categories (Qwen3-VL + SAM3)
    print("Initializing Qwen3 + SAM3 detector...")
    detector = UnifiedInfrastructureDetector(
        categories=None,   # ALL categories
        device=device,
        use_quantization=use_quantization,
        low_memory=low_memory
    )

    # Create category output directories
    category_dirs = {}
    for category in INFRASTRUCTURE_CATEGORIES.keys():
        cat_dir = output_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        (cat_dir / "frames").mkdir(exist_ok=True)
        (cat_dir / "annotated").mkdir(exist_ok=True)
        category_dirs[category] = cat_dir

    # Extract frames
    print("\nExtracting frames...")
    all_frames = []
    frame_indices = []
    frame_idx = 0

    pbar = tqdm(total=total_frames, desc="Extracting")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            all_frames.append((frame_rgb, pil_image, frame_idx))
            frame_indices.append(frame_idx)

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    print(f"✓ Extracted {len(all_frames)} frames\n")

    # Process all frames with multi-detection
    print("Processing frames with Qwen3 + SAM3...")
    category_data = {cat: [] for cat in INFRASTRUCTURE_CATEGORIES.keys()}
    category_counts = {cat: 0 for cat in INFRASTRUCTURE_CATEGORIES.keys()}

    pbar = tqdm(all_frames, desc="Multi-detecting")

    for frame_rgb, pil_image, frame_idx in pbar:
        try:
            # Qwen3 detects ALL categories, SAM3 segments each
            result = detector.detect_infrastructure(pil_image)

            # Handle None or invalid result
            if result is None:
                logger.error(f"Detection returned None for frame {frame_idx}")
                continue

            if result.get('num_detections', 0) == 0:
                continue

            # Group detections by category
            detections_by_category = {}
            for det in result['detections']:
                category = det.get('label', 'unknown')
                if category not in detections_by_category:
                    detections_by_category[category] = []
                detections_by_category[category].append(det)

            # Save frame to each category folder
            for category, detections in detections_by_category.items():
                if category not in category_dirs:
                    continue

                # Save original frame
                frame_path = category_dirs[category] / "frames" / f"frame_{frame_idx:06d}.jpg"
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(frame_path), frame_bgr)

                # Save annotated frame (only this category's detections)
                annotated = draw_category_detections(frame_rgb, detections, category)
                annotated_path = category_dirs[category] / "annotated" / f"frame_{frame_idx:06d}_detected.jpg"
                annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(annotated_path), annotated_bgr)

                # Store detection data
                category_data[category].append({
                    "frame_index": frame_idx,
                    "timestamp": frame_idx / video_fps,
                    "frame_path": str(frame_path),
                    "annotated_path": str(annotated_path),
                    "detections": detections
                })

                category_counts[category] += 1

                # Update progress bar with counts
                pbar.set_postfix({
                    'detected': sum(1 for c in category_counts.values() if c > 0)
                })

        except Exception as e:
            logger.error(f"Error processing frame {frame_idx}: {e}")

    pbar.close()

    # Save per-category JSON files
    print("\nSaving category detections...")
    summary = {}

    for category in INFRASTRUCTURE_CATEGORIES.keys():
        if category_counts[category] == 0:
            continue

        json_path = category_dirs[category] / "detections.json"
        with open(json_path, 'w') as f:
            json.dump({
                "category": category,
                "video": video_path,
                "total_frames_processed": len(all_frames),
                "frames_with_detections": category_counts[category],
                "total_detections": sum(len(d['detections']) for d in category_data[category]),
                "detections": category_data[category]
            }, f, indent=2)

        summary[category] = {
            "frames_with_detections": category_counts[category],
            "total_detections": sum(len(d['detections']) for d in category_data[category])
        }

        print(f"  ✓ {category}: {category_counts[category]} frames, "
              f"{summary[category]['total_detections']} detections")

    # Save overall summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print("PROCESSING COMPLETE!")
    print(f"{'='*70}")
    print(f"Summary: {summary_path}")
    print(f"Total categories detected: {len([c for c in category_counts.values() if c > 0])}")
    print(f"{'='*70}\n")


def draw_category_detections(frame_rgb, detections, category):
    """Draw detections for a specific category on frame."""
    frame = frame_rgb.copy()
    color_bgr = DEFECT_COLORS.get(category, (0, 255, 0))
    # Convert BGR to RGB for drawing
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])

    for det in detections:
        bbox = det.get('bbox', [])
        if len(bbox) == 4:
            x1, y1, x2, y2 = map(int, bbox)

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_rgb, 3)

            # Draw label with confidence
            label = det.get('label', 'unknown')
            conf = det.get('confidence', 0.0)
            text = f"{label} {conf:.2f}"

            # Background for text
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), color_rgb, -1)

            # Text
            cv2.putText(
                frame,
                text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

    return frame


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-category detection: Qwen3 + SAM3 pipeline"
    )
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--fps", type=float, default=2.0, help="Target FPS (default: 2.0)")
    parser.add_argument("--device", choices=["cuda", "cpu"], help="Device to use")
    parser.add_argument("--quantize", action="store_true", help="Use 8-bit quantization")
    parser.add_argument("--low-memory", action="store_true", help="Low memory mode")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")

    args = parser.parse_args()

    process_video_multi_detection(
        video_path=args.input,
        output_dir=args.output,
        target_fps=args.fps,
        device=args.device,
        use_quantization=args.quantize,
        low_memory=args.low_memory,
        batch_size=args.batch_size
    )
