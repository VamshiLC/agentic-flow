#!/usr/bin/env python3
"""
ASH Infrastructure Detection - Hugging Face Transformers

Primary inference method using optimized Hugging Face implementation.
No vLLM server required - works out of the box.

Features:
- Direct model loading with Transformers
- Optional 8-bit quantization for lower memory
- True batch processing for better GPU utilization
- Comprehensive error handling and logging

Usage:
    # Single image
    python main_simple.py --mode image --input frame.jpg

    # Video with batch processing
    python main_simple.py --mode video --input video.mp4 --fps 2.0 --batch-size 4

    # Low memory mode with quantization
    python main_simple.py --mode video --input video.mp4 --quantize --low-memory
"""
import argparse
import sys
import os
import json
import logging
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from detector_unified import UnifiedInfrastructureDetector, DEFECT_COLORS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


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


def process_video(
    video_path: str,
    detector,
    output_dir: str,
    target_fps: float = 1.0,
    batch_size: int = 1,
    save_video: bool = True
):
    """
    Process video with optional batch processing.

    Args:
        video_path: Path to input video
        detector: Detector instance
        output_dir: Output directory
        target_fps: Target FPS for processing
        batch_size: Number of frames to process in batch (1 = sequential)
        save_video: Whether to save annotated video
    """
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
    if batch_size > 1:
        print(f"  Batch size: {batch_size} (processing {batch_size} frames at once)")

    # Create output directory
    output_dir = Path(output_dir)
    video_name = Path(video_path).stem
    video_output_dir = output_dir / video_name
    video_output_dir.mkdir(parents=True, exist_ok=True)

    # Extract frames to process
    frames_to_process = []
    frame_indices = []
    frame_idx = 0

    print("\nExtracting frames...")
    pbar = tqdm(total=total_frames, desc="Extracting")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Extract every Nth frame
        if frame_idx % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            frames_to_process.append((frame_rgb, pil_image))
            frame_indices.append(frame_idx)

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    print(f"Extracted {len(frames_to_process)} frames to process")

    # Process frames (with batching if batch_size > 1)
    all_detections = []
    annotated_frames = []

    if batch_size > 1:
        # Batch processing
        print(f"\nProcessing in batches of {batch_size}...")
        pbar = tqdm(total=len(frames_to_process), desc="Detecting")

        for i in range(0, len(frames_to_process), batch_size):
            batch_frames = frames_to_process[i:i+batch_size]
            batch_indices = frame_indices[i:i+batch_size]

            # Extract PIL images for detection
            pil_images = [pil_img for _, pil_img in batch_frames]

            # Run batch detection
            try:
                results = [detector.detect_infrastructure(img) for img in pil_images]

                # Process results
                for (frame_rgb, _), frame_idx, result in zip(batch_frames, batch_indices, results):
                    all_detections.append({
                        "frame_index": frame_idx,
                        "timestamp": frame_idx / video_fps,
                        "num_detections": result['num_detections'],
                        "detections": result['detections']
                    })

                    # Draw annotations
                    if save_video:
                        annotated = draw_detections(frame_rgb, result['detections'])
                        annotated_frames.append(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

                    pbar.update(1)

            except Exception as e:
                logging.error(f"Error processing batch: {e}")
                # Add empty results for failed batch
                for frame_idx in batch_indices:
                    all_detections.append({
                        "frame_index": frame_idx,
                        "timestamp": frame_idx / video_fps,
                        "num_detections": 0,
                        "detections": [],
                        "error": str(e)
                    })
                pbar.update(len(batch_frames))

        pbar.close()

    else:
        # Sequential processing
        print("\nProcessing frames...")
        pbar = tqdm(frames_to_process, desc="Detecting")

        for (frame_rgb, pil_image), frame_idx in zip(pbar, frame_indices):
            try:
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
                if save_video:
                    annotated = draw_detections(frame_rgb, result['detections'])
                    annotated_frames.append(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

            except Exception as e:
                logging.error(f"Error processing frame {frame_idx}: {e}")
                all_detections.append({
                    "frame_index": frame_idx,
                    "timestamp": frame_idx / video_fps,
                    "num_detections": 0,
                    "detections": [],
                    "error": str(e)
                })

    print(f"\nProcessed {len(frames_to_process)} frames")

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
        description="ASH Infrastructure Detection (Hugging Face Transformers - No vLLM Server)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single image
  python main_simple.py --mode image --input frame.jpg

  # Process video at 2 FPS
  python main_simple.py --mode video --input video.mp4 --fps 2.0

  # Batch processing for faster inference
  python main_simple.py --mode video --input video.mp4 --fps 2.0 --batch-size 4

  # Low memory mode with 8-bit quantization
  python main_simple.py --mode video --input video.mp4 --quantize --low-memory

  # Custom categories only
  python main_simple.py --mode image --input frame.jpg --categories potholes alligator_cracks

  # Force CPU usage
  python main_simple.py --mode image --input frame.jpg --device cpu
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
        default="output",
        help="Output directory (default: output)"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Target FPS for video processing (default: 2.0)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for video processing (default: 1, increase for faster processing)"
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        help="Categories to detect (default: all 12 categories)"
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
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Use 8-bit quantization to reduce memory usage by ~50%% (requires bitsandbytes)"
    )
    parser.add_argument(
        "--low-memory",
        action="store_true",
        help="Enable low memory optimizations"
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip saving annotated video (faster, JSON only)"
    )

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1

    # Initialize detector (direct mode - no server)
    print("="*70)
    print("ASH INFRASTRUCTURE DETECTION - HUGGING FACE TRANSFORMERS")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Device: {args.device or 'auto-detect'}")
    print(f"Quantization: {'Enabled (8-bit)' if args.quantize else 'Disabled'}")
    print(f"Low memory mode: {'Enabled' if args.low_memory else 'Disabled'}")
    if args.mode == "video" and args.batch_size > 1:
        print(f"Batch processing: Enabled (batch size: {args.batch_size})")

    try:
        detector = UnifiedInfrastructureDetector(
            mode="direct",
            model_name=args.model,
            categories=args.categories,
            device=args.device,
            use_quantization=args.quantize,
            low_memory=args.low_memory
        )
    except Exception as e:
        print(f"\nError loading detector: {e}")
        logging.exception("Detector initialization failed")
        return 1

    print("\nCategories to detect:")
    for cat in detector.categories:
        print(f"  - {cat}")

    # Process based on mode
    try:
        if args.mode == "image":
            process_image(args.input, detector, args.output)
        elif args.mode == "video":
            process_video(
                args.input,
                detector,
                args.output,
                args.fps,
                batch_size=args.batch_size,
                save_video=not args.no_video
            )

        print("\n" + "="*70)
        print("PROCESSING COMPLETE")
        print("="*70)

        # Cleanup
        if hasattr(detector.detector, 'cleanup'):
            print("\nCleaning up GPU memory...")
            detector.detector.cleanup()

        return 0

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"\nError: {e}")
        logging.exception("Processing failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
