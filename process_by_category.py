"""
Category-wise Video Processing

Process videos one category at a time:
- Extracts frames
- Processes ALL frames for category X → saves only frames with detections
- Then processes ALL frames for category Y → saves only frames with detections
- etc.

This is more efficient than processing all categories on every frame.
"""
import os
import cv2
import json
import logging
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from typing import List, Dict, Optional

from detector_unified import UnifiedInfrastructureDetector
from category_config import (
    get_model_for_category,
    should_use_qwen,
    should_use_sam3,
    CATEGORY_GROUPS
)

logger = logging.getLogger(__name__)


def process_video_by_category(
    video_path: str,
    output_dir: str,
    categories: Optional[List[str]] = None,
    target_fps: float = 2.0,
    device: Optional[str] = None,
    use_quantization: bool = False,
    low_memory: bool = False
):
    """
    Process video one category at a time.

    Args:
        video_path: Path to video file
        output_dir: Output directory
        categories: List of categories to process (default: all)
        target_fps: Target processing FPS
        device: Device to use
        use_quantization: Use 8-bit quantization
        low_memory: Enable low memory mode

    Output structure:
        results/
        ├── potholes/
        │   ├── frames/           # Only frames with potholes
        │   ├── annotated/        # Annotated frames
        │   └── detections.json
        ├── cracks/
        │   ├── frames/
        │   ├── annotated/
        │   └── detections.json
        └── ...
    """
    video_path = str(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if categories is None:
        from detector_unified import INFRASTRUCTURE_CATEGORIES
        categories = list(INFRASTRUCTURE_CATEGORIES.keys())

    # Open video
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n{'='*70}")
    print(f"CATEGORY-WISE VIDEO PROCESSING")
    print(f"{'='*70}")
    print(f"Video: {video_path}")
    print(f"Total frames: {total_frames}")
    print(f"Video FPS: {video_fps}")
    print(f"Processing FPS: {target_fps}")
    print(f"Categories to process: {len(categories)}")
    print(f"{'='*70}\n")

    # Calculate frame sampling
    frame_interval = max(1, int(video_fps / target_fps))

    # Extract ALL frames once
    print("Step 1: Extracting all frames...")
    all_frames = []
    frame_indices = []
    frame_idx = 0

    pbar = tqdm(total=total_frames, desc="Extracting frames")
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

    # Process each category separately
    summary = {}

    for category_idx, category in enumerate(categories, 1):
        print(f"\n{'='*70}")
        print(f"Processing Category {category_idx}/{len(categories)}: {category}")
        print(f"{'='*70}")

        # Determine which model to use
        model_type = get_model_for_category(category)
        use_qwen = should_use_qwen(category)
        use_sam3 = should_use_sam3(category)

        print(f"Model: {model_type.upper()}")
        print(f"  - Qwen3-VL: {'✓' if use_qwen else '✗'}")
        print(f"  - SAM3: {'✓' if use_sam3 else '✗'}")

        # Initialize detector for this category
        detector_mode = "agent-hf" if model_type == "both" else "direct"

        detector = UnifiedInfrastructureDetector(
            mode=detector_mode,
            categories=[category],  # Only this category!
            device=device,
            use_quantization=use_quantization,
            low_memory=low_memory
        )

        # Create category output directory
        category_dir = output_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        frames_dir = category_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        annotated_dir = category_dir / "annotated"
        annotated_dir.mkdir(exist_ok=True)

        # Process all frames for this category
        category_detections = []
        frames_with_detections = 0

        print(f"\nProcessing {len(all_frames)} frames...")
        pbar = tqdm(all_frames, desc=f"Detecting {category}")

        for frame_rgb, pil_image, frame_idx in pbar:
            try:
                # Run detection for this category only
                result = detector.detect_infrastructure(pil_image)

                if result['num_detections'] > 0:
                    # Found detection for this category!
                    frames_with_detections += 1

                    # Save original frame
                    frame_path = frames_dir / f"frame_{frame_idx:06d}.jpg"
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(frame_path), frame_bgr)

                    # Save annotated frame
                    annotated = draw_detections_simple(frame_rgb, result['detections'])
                    annotated_path = annotated_dir / f"frame_{frame_idx:06d}_detected.jpg"
                    annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(annotated_path), annotated_bgr)

                    # Store detection
                    category_detections.append({
                        "frame_index": frame_idx,
                        "timestamp": frame_idx / video_fps,
                        "frame_path": str(frame_path),
                        "annotated_path": str(annotated_path),
                        "detections": result['detections']
                    })

            except Exception as e:
                logger.error(f"Error processing frame {frame_idx} for {category}: {e}")

        pbar.close()

        # Save category detections
        json_path = category_dir / "detections.json"
        with open(json_path, 'w') as f:
            json.dump({
                "category": category,
                "model_used": model_type,
                "video": video_path,
                "total_frames_processed": len(all_frames),
                "frames_with_detections": frames_with_detections,
                "detections": category_detections
            }, f, indent=2)

        summary[category] = {
            "model": model_type,
            "frames_with_detections": frames_with_detections,
            "total_detections": sum(len(d['detections']) for d in category_detections)
        }

        print(f"\n✓ {category}:")
        print(f"  - Frames with detections: {frames_with_detections}/{len(all_frames)}")
        print(f"  - Total detections: {summary[category]['total_detections']}")
        print(f"  - Saved to: {category_dir}")

    # Save overall summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print("PROCESSING COMPLETE!")
    print(f"{'='*70}")
    print(f"Summary saved to: {summary_path}")
    print("\nResults by category:")
    for cat, stats in summary.items():
        if stats['frames_with_detections'] > 0:
            print(f"  ✓ {cat}: {stats['frames_with_detections']} frames, {stats['total_detections']} detections")


def draw_detections_simple(frame_rgb, detections):
    """
    Draw professional-looking bounding boxes with adaptive sizing.

    Features:
    - Adaptive sizing based on frame dimensions
    - Vibrant colors using DEFECT_COLORS palette
    - Clean text labels with readable backgrounds
    """
    import cv2
    import numpy as np
    from detector_unified import DEFECT_COLORS

    frame = frame_rgb.copy()
    height, width = frame.shape[:2]

    # Adaptive sizing
    scale = min(width / 1920, height / 1080)
    scale = max(0.5, min(scale, 2.0))

    bbox_thickness = max(2, int(4 * scale))
    font_scale = max(0.5, 0.8 * scale)
    font_thickness = max(1, int(2 * scale))
    text_padding = max(5, int(8 * scale))

    for det in detections:
        bbox = det.get('bbox', [])
        if len(bbox) == 4:
            x1, y1, x2, y2 = map(int, bbox)
            label = det.get('label', 'unknown')
            confidence = det.get('confidence', 0.0)

            # Get color from palette
            color_bgr = det.get('color', DEFECT_COLORS.get(label, (0, 255, 0)))

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, bbox_thickness)

            # Prepare label text
            label_display = label.replace('_', ' ').title()
            text = f"{label_display} {confidence:.0%}"

            # Calculate text size
            (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                                         font_scale, font_thickness)

            # Position label
            label_y_top = y1 - text_h - text_padding * 2 - baseline
            if label_y_top < 10:
                label_y_top = y2 + baseline
                label_y_bottom = label_y_top + text_h + text_padding * 2
                text_y = label_y_top + text_h + text_padding
            else:
                label_y_bottom = y1
                text_y = y1 - text_padding - baseline

            # Draw semi-transparent background
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, label_y_top),
                         (x1 + text_w + text_padding * 2, label_y_bottom),
                         color_bgr, -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

            # Draw text with border
            cv2.putText(frame, text, (x1 + text_padding, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                       (0, 0, 0), font_thickness + 1)
            cv2.putText(frame, text, (x1 + text_padding, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                       (255, 255, 255), font_thickness)

    return frame


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process video by category")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--categories", nargs="+", help="Categories to process")
    parser.add_argument("--fps", type=float, default=2.0, help="Target FPS")
    parser.add_argument("--device", choices=["cuda", "cpu"], help="Device")
    parser.add_argument("--quantize", action="store_true", help="Use quantization")
    parser.add_argument("--low-memory", action="store_true", help="Low memory mode")

    args = parser.parse_args()

    process_video_by_category(
        video_path=args.input,
        output_dir=args.output,
        categories=args.categories,
        target_fps=args.fps,
        device=args.device,
        use_quantization=args.quantize,
        low_memory=args.low_memory
    )
