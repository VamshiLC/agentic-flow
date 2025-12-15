#!/usr/bin/env python3
"""
Test Infrastructure Detection on Video

Usage:
    python test_video.py --video /path/to/video.mp4
    python test_video.py --video /path/to/video.mp4 --fps 1
    python test_video.py --video /path/to/video.mp4 --output output_video.mp4
"""
import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Test Infrastructure Detection on Video")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--fps", type=float, default=1.0, help="Process N frames per second (default: 1)")
    parser.add_argument("--quantization", action="store_true", help="Use 8-bit quantization (saves GPU memory)")
    parser.add_argument("--confidence", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--detect", type=str, help="Comma-separated list of things to detect")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", help="Qwen model to use")
    parser.add_argument("--save-frames", action="store_true", help="Save individual processed frames")
    parser.add_argument("--save-video", action="store_true", help="Save output as video")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Parse custom categories
    categories = None
    if args.detect:
        categories = [c.strip() for c in args.detect.split(",")]

    print("=" * 60)
    print("INFRASTRUCTURE DETECTION - VIDEO MODE")
    print("=" * 60)
    print(f"Video: {args.video}")
    print(f"Process rate: {args.fps} fps")
    if categories:
        print(f"Custom detection: {categories}")
    else:
        print("Detecting: ALL categories")

    # Import and initialize detector
    from agent.detection_agent_hf import InfrastructureDetectionAgentHF

    print("\nLoading detector...")
    detector = InfrastructureDetectionAgentHF(
        model_name=args.model,
        use_quantization=args.quantization,
        sam3_confidence=args.confidence,
        categories=categories,
        debug=False  # Less verbose for video
    )

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video: {args.video}")
        sys.exit(1)

    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / video_fps if video_fps > 0 else 0

    print(f"\nVideo info:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {video_fps:.2f}")
    print(f"  Duration: {duration:.2f}s ({total_frames} frames)")

    # Calculate frame skip
    frame_skip = int(video_fps / args.fps) if args.fps < video_fps else 1
    frames_to_process = total_frames // frame_skip
    print(f"  Processing every {frame_skip} frames (~{frames_to_process} frames total)")

    # Setup video writer if saving video
    out_video = None
    if args.save_video:
        output_video_path = os.path.join(args.output, f"detected_{Path(args.video).stem}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(output_video_path, fourcc, args.fps, (width, height))
        print(f"\nOutput video: {output_video_path}")

    # Process frames
    print(f"\n{'='*60}")
    print("Processing video...")
    print("=" * 60)

    frame_idx = 0
    processed_count = 0
    all_detections = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames based on fps setting
        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        # Convert BGR to RGB for PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        # Run detection
        timestamp = frame_idx / video_fps
        print(f"\n[Frame {frame_idx}/{total_frames}] Time: {timestamp:.2f}s", end=" ")

        try:
            result = detector.detect_infrastructure(pil_image)
            num_detections = result['num_detections']

            if num_detections > 0:
                print(f"✓ {num_detections} detections")
                for det in result['detections']:
                    category = det.get('category', 'unknown')
                    confidence = det.get('confidence', 0)
                    print(f"    - {category}: {confidence:.2f}")
                    all_detections.append({
                        'frame': frame_idx,
                        'timestamp': timestamp,
                        'category': category,
                        'confidence': confidence
                    })
            else:
                print("✗ no detections")

            # Get result image
            if result.get('final_image'):
                result_frame = np.array(result['final_image'])
                result_frame_bgr = cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)

                # Save individual frame if requested
                if args.save_frames:
                    frame_path = os.path.join(args.output, f"frame_{frame_idx:06d}.png")
                    cv2.imwrite(frame_path, result_frame_bgr)

                # Write to video
                if out_video:
                    out_video.write(result_frame_bgr)

        except Exception as e:
            print(f"✗ Error: {e}")

        frame_idx += 1
        processed_count += 1

    # Cleanup
    cap.release()
    if out_video:
        out_video.release()

    # Summary
    print(f"\n{'='*60}")
    print("VIDEO PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Processed {processed_count} frames")
    print(f"Total detections: {len(all_detections)}")

    if all_detections:
        # Count by category
        from collections import Counter
        category_counts = Counter(d['category'] for d in all_detections)
        print("\nDetections by category:")
        for cat, count in category_counts.most_common():
            print(f"  - {cat}: {count}")

    if args.save_video:
        print(f"\nOutput video saved: {output_video_path}")

    print(f"\nResults saved to: {args.output}/")
    print("=" * 60)

    # Cleanup detector
    detector.cleanup()


if __name__ == "__main__":
    main()
