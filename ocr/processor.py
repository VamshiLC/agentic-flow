"""
Image and Video Processing for License Plate OCR

- process_image(): Single image OCR
- process_video(): Video frame OCR
- Output saving (JSON, annotated images/video)
"""

import os
import cv2
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List, Union
from PIL import Image
from tqdm import tqdm
import numpy as np

from .license_plate_agent import LicensePlateOCR
from .utils import draw_plate_detections, create_plate_summary, pil_to_cv2
from .tracker import PlateTracker

logger = logging.getLogger(__name__)


def process_image(
    image_path: str,
    output_dir: str,
    ocr_agent: Optional[LicensePlateOCR] = None,
    save_annotated: bool = True,
    save_json: bool = True,
    use_quantization: bool = False
) -> Dict:
    """
    Process a single image for license plate OCR.

    Args:
        image_path: Path to input image
        output_dir: Output directory for results
        ocr_agent: Optional pre-initialized OCR agent
        save_annotated: Save annotated image
        save_json: Save JSON results
        use_quantization: Use 8-bit quantization

    Returns:
        Detection results dict
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize OCR agent if not provided
    if ocr_agent is None:
        ocr_agent = LicensePlateOCR(use_quantization=use_quantization)

    # Load image
    image = Image.open(image_path).convert('RGB')
    image_name = Path(image_path).stem

    print(f"\nProcessing: {image_path}")

    # Run detection + OCR
    result = ocr_agent.detect_and_read(image)

    # Print results
    print(f"Found {result['num_plates']} plate(s)")
    for i, plate in enumerate(result['plates']):
        print(f"  Plate {i+1}: {plate.get('plate_text', 'UNREADABLE')}")
        print(f"    State: {plate.get('state', 'Unknown')}")
        print(f"    Confidence: {plate.get('ocr_confidence', 0):.2f}")

    # Save annotated image
    if save_annotated and result['num_plates'] > 0:
        frame_rgb = np.array(image)
        annotated = draw_plate_detections(frame_rgb, result['plates'])
        annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

        annotated_path = output_dir / f"{image_name}_plates.jpg"
        cv2.imwrite(str(annotated_path), annotated_bgr)
        print(f"Saved annotated: {annotated_path}")

    # Save JSON results
    if save_json:
        json_result = {
            'image': str(image_path),
            'num_plates': result['num_plates'],
            'plates': result['plates']
        }

        json_path = output_dir / f"{image_name}_plates.json"
        with open(json_path, 'w') as f:
            json.dump(json_result, f, indent=2)
        print(f"Saved JSON: {json_path}")

    return result


def process_video(
    video_path: str,
    output_dir: str,
    target_fps: float = 1.0,
    ocr_agent: Optional[LicensePlateOCR] = None,
    save_frames: bool = True,
    save_video: bool = True,
    save_json: bool = True,
    use_quantization: bool = False,
    enable_tracking: bool = True
) -> Dict:
    """
    Process video for license plate OCR with tracking.

    Args:
        video_path: Path to input video
        output_dir: Output directory for results
        target_fps: Target processing FPS (e.g., 1.0 = 1 frame/second)
        ocr_agent: Optional pre-initialized OCR agent
        save_frames: Save individual annotated frames
        save_video: Save annotated video
        save_json: Save JSON results
        use_quantization: Use 8-bit quantization
        enable_tracking: Enable plate tracking across frames with voting

    Returns:
        Summary results dict
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames_dir = output_dir / "frames"
    if save_frames:
        frames_dir.mkdir(exist_ok=True)

    # Initialize OCR agent if not provided
    if ocr_agent is None:
        ocr_agent = LicensePlateOCR(use_quantization=use_quantization)

    # Initialize plate tracker
    tracker = PlateTracker(
        iou_threshold=0.3,
        min_readings=2,
        max_frames_missing=5
    ) if enable_tracking else None

    # Open video
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n{'='*60}")
    print("LICENSE PLATE OCR - Video Processing")
    print(f"{'='*60}")
    print(f"Video: {video_path}")
    print(f"Total frames: {total_frames}")
    print(f"Video FPS: {video_fps:.1f}")
    print(f"Processing FPS: {target_fps}")
    print(f"Resolution: {width}x{height}")
    print(f"Tracking: {'Enabled' if enable_tracking else 'Disabled'}")
    print(f"{'='*60}\n")

    # Calculate frame interval
    frame_interval = max(1, int(video_fps / target_fps))
    frames_to_process = total_frames // frame_interval

    # Video writer for annotated output
    video_writer = None
    if save_video:
        output_video_path = output_dir / f"{Path(video_path).stem}_plates.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            str(output_video_path),
            fourcc,
            target_fps,
            (width, height)
        )

    # Process frames
    all_results = []
    all_plates = []
    frame_idx = 0
    processed_count = 0

    pbar = tqdm(total=frames_to_process, desc="Processing")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            # Convert to RGB for processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            # Run OCR
            result = ocr_agent.detect_and_read(pil_image)

            # Update tracker if enabled
            if tracker and result['num_plates'] > 0:
                tracked_plates = tracker.update(processed_count, result['plates'])
                result['plates'] = tracked_plates

            # Store results
            frame_result = {
                'frame_index': frame_idx,
                'timestamp': frame_idx / video_fps,
                'num_plates': result['num_plates'],
                'plates': []
            }

            for plate in result['plates']:
                plate_data = {k: v for k, v in plate.items() if k != 'mask'}
                frame_result['plates'].append(plate_data)
                all_plates.append(plate_data)

            all_results.append(frame_result)

            # Draw annotations
            if result['num_plates'] > 0:
                annotated = draw_plate_detections(frame_rgb, result['plates'])
                annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            else:
                annotated_bgr = frame

            # Save frame
            if save_frames and result['num_plates'] > 0:
                frame_path = frames_dir / f"frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(frame_path), annotated_bgr)

            # Write to video
            if video_writer:
                video_writer.write(annotated_bgr)

            processed_count += 1
            pbar.update(1)

            # Update progress bar
            tracked_count = len(tracker.tracked_plates) if tracker else len(all_plates)
            pbar.set_postfix({
                'detections': len(all_plates),
                'tracked': tracked_count
            })

        frame_idx += 1

    pbar.close()
    cap.release()

    if video_writer:
        video_writer.release()

    # Get tracked plates with voting results
    tracked_results = []
    if tracker:
        tracked_results = tracker.get_all_tracked_plates()

    # Create summary
    summary = create_plate_summary(all_plates)
    summary['video'] = str(video_path)
    summary['frames_processed'] = processed_count
    summary['frames_with_plates'] = sum(1 for r in all_results if r['num_plates'] > 0)

    # Add tracking summary
    if tracker and tracked_results:
        summary['tracking_enabled'] = True
        summary['unique_plates_tracked'] = len(tracked_results)
        summary['tracked_plates'] = tracked_results

        # Get final plate texts from tracking (voted results)
        summary['final_plate_texts'] = [t['plate_text'] for t in tracked_results]
    else:
        summary['tracking_enabled'] = False

    # Save JSON results
    if save_json:
        # Full results
        results_path = output_dir / "detections.json"
        with open(results_path, 'w') as f:
            json.dump({
                'summary': summary,
                'frames': all_results,
                'tracked_plates': tracked_results
            }, f, indent=2)

        # Summary only
        summary_path = output_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Frames processed: {processed_count}")
    print(f"Frames with plates: {summary['frames_with_plates']}")
    print(f"Total detections: {summary['total_plates']}")

    if tracker and tracked_results:
        print(f"\n--- TRACKING RESULTS (with voting) ---")
        print(f"Unique plates tracked: {len(tracked_results)}")
        print(f"\nFinal Plate Numbers:")
        for t in tracked_results:
            print(f"  - {t['plate_text']} (confidence: {t['confidence']:.2f}, votes: {t['vote_count']}/{t['total_readings']})")
            if t.get('all_readings'):
                print(f"    All readings: {t['all_readings']}")
    else:
        if summary['plate_texts']:
            print(f"\nDetected plate numbers:")
            for text in set(summary['plate_texts']):
                print(f"  - {text}")

    if save_video:
        print(f"\nOutput video: {output_video_path}")
    if save_json:
        print(f"Results JSON: {results_path}")

    print(f"{'='*60}\n")

    return {
        'summary': summary,
        'results': all_results,
        'tracked_plates': tracked_results
    }


def process_batch_images(
    image_paths: List[str],
    output_dir: str,
    ocr_agent: Optional[LicensePlateOCR] = None,
    use_quantization: bool = False
) -> List[Dict]:
    """
    Process multiple images for license plate OCR.

    Args:
        image_paths: List of image paths
        output_dir: Output directory
        ocr_agent: Optional pre-initialized agent
        use_quantization: Use quantization

    Returns:
        List of results for each image
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize agent once for all images
    if ocr_agent is None:
        ocr_agent = LicensePlateOCR(use_quantization=use_quantization)

    results = []
    all_plates = []

    print(f"\nProcessing {len(image_paths)} images...")

    for image_path in tqdm(image_paths, desc="Processing"):
        result = process_image(
            image_path=image_path,
            output_dir=output_dir,
            ocr_agent=ocr_agent,
            save_annotated=True,
            save_json=False  # Save combined JSON at end
        )
        results.append({
            'image': str(image_path),
            **result
        })

        for plate in result.get('plates', []):
            plate_data = {k: v for k, v in plate.items() if k != 'mask'}
            plate_data['source_image'] = str(image_path)
            all_plates.append(plate_data)

    # Save combined results
    summary = create_plate_summary(all_plates)
    summary['total_images'] = len(image_paths)
    summary['images_with_plates'] = sum(1 for r in results if r.get('num_plates', 0) > 0)

    combined_path = output_dir / "all_results.json"
    with open(combined_path, 'w') as f:
        json.dump({
            'summary': summary,
            'images': results
        }, f, indent=2)

    print(f"\nBatch processing complete!")
    print(f"Total plates found: {summary['total_plates']}")
    print(f"Results saved to: {combined_path}")

    return results
