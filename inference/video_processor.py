"""
Video Processing Pipeline

Batch process entire GoPro videos for infrastructure detection.
"""
import os
import json
import glob
import gc
import shutil
from datetime import datetime
from utils.video_utils import extract_frames, get_video_info, chunk_video
from inference.single_frame import process_single_frame
from utils.output_formatter import save_detection_json, create_detection_summary
from models.sam3_loader import load_sam3_model
from agent.detection_agent_hf import InfrastructureDetectionAgentHF

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def process_video(
    video_path,
    output_dir="output",
    sample_rate=15,
    start_time=None,
    end_time=None,
    debug=False,
    # Chunking parameters
    enable_chunking=False,
    chunk_duration=600.0,
    chunk_overlap=1.0,
    cleanup_chunks=True
):
    """
    Process entire GoPro video for infrastructure detection.

    Args:
        video_path: Path to the input video file
        output_dir: Directory to save all outputs
        sample_rate: Extract every Nth frame (15 = 2 frames per second at 30fps)
        start_time: Optional start time in seconds
        end_time: Optional end time in seconds
        debug: If True, print debug information
        enable_chunking: Enable video chunking for large files (default: False)
        chunk_duration: Duration of each chunk in seconds (default: 600 = 10 min)
        chunk_overlap: Overlap between chunks in seconds (default: 1.0)
        cleanup_chunks: Delete chunk files after processing (default: True)

    Returns:
        dict: Processing results with summary statistics

    Example:
        # Basic processing
        results = process_video(
            "gopro_video.mp4",
            output_dir="results/",
            sample_rate=15  # 2 fps
        )

        # With chunking for large videos
        results = process_video(
            "long_video.mp4",
            output_dir="results/",
            enable_chunking=True,
            chunk_duration=600
        )
    """
    print(f"\n{'='*70}")
    print(f"VIDEO PROCESSING PIPELINE")
    print(f"{'='*70}")
    print(f"Video: {video_path}")
    print(f"Output directory: {output_dir}")
    print(f"Sample rate: every {sample_rate} frames")
    print(f"{'='*70}\n")

    # Validate input
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Check if chunking is enabled
    if enable_chunking:
        return _process_video_with_chunks(
            video_path, output_dir, sample_rate,
            chunk_duration, chunk_overlap, cleanup_chunks, debug
        )

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, "frames")
    detections_dir = os.path.join(output_dir, "detections")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(detections_dir, exist_ok=True)

    # Get video info
    print("STEP 1: Analyzing video...")
    video_info = get_video_info(video_path)
    print(f"  - Resolution: {video_info['width']}x{video_info['height']}")
    print(f"  - FPS: {video_info['fps']:.2f}")
    print(f"  - Duration: {video_info['duration_formatted']}")
    print(f"  - Total frames: {video_info['frame_count']}")

    # Extract frames
    print(f"\nSTEP 2: Extracting frames (sample rate: {sample_rate})...")
    num_frames = extract_frames(
        video_path,
        frames_dir,
        sample_rate=sample_rate,
        start_time=start_time,
        end_time=end_time
    )

    if num_frames == 0:
        print("ERROR: No frames extracted from video")
        return None

    # Load detector once for all frames
    print(f"\nSTEP 3: Loading HF-based detector (Qwen + SAM3)...")
    sam3_processor = load_sam3_model(confidence_threshold=0.3)
    detector = InfrastructureDetectionAgentHF(
        sam3_processor=sam3_processor,
        sam3_confidence=0.3
    )

    # Process each frame
    print(f"\nSTEP 4: Processing {num_frames} frames...")
    print(f"{'='*70}")

    frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    all_detections = []

    for i, frame_path in enumerate(frame_files):
        frame_num = i + 1
        print(f"\n[{frame_num}/{num_frames}] Processing {os.path.basename(frame_path)}")

        # Process frame
        try:
            detections = process_single_frame(
                frame_path,
                output_dir=detections_dir,
                detector=detector,
                save_json=True,
                debug=debug
            )
            all_detections.append(detections)

        except Exception as e:
            print(f"ERROR processing frame {frame_num}: {e}")
            # Add empty detection for failed frames
            all_detections.append({
                "frame_id": os.path.basename(frame_path),
                "detections": [],
                "error": str(e)
            })

    # Create summary
    print(f"\n{'='*70}")
    print("STEP 5: Creating summary...")
    summary = create_detection_summary(all_detections)

    # Add video metadata to summary
    summary["video_info"] = {
        "path": os.path.abspath(video_path),
        "filename": os.path.basename(video_path),
        "duration": video_info["duration"],
        "fps": video_info["fps"],
        "resolution": f"{video_info['width']}x{video_info['height']}",
        "sample_rate": sample_rate
    }
    summary["processing_info"] = {
        "timestamp": datetime.now().isoformat(),
        "frames_processed": num_frames,
        "output_directory": os.path.abspath(output_dir)
    }

    # Save consolidated results
    consolidated_path = os.path.join(output_dir, "video_detections.json")
    save_detection_json({
        "summary": summary,
        "frames": all_detections
    }, consolidated_path)

    # Save summary separately
    summary_path = os.path.join(output_dir, "summary.json")
    save_detection_json(summary, summary_path)

    # Print final summary
    print(f"\n{'='*70}")
    print("PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Total frames processed: {summary['total_frames']}")
    print(f"Frames with issues: {summary['frames_with_issues']}")
    print(f"Total detections: {summary['total_detections']}")
    print(f"\nDetections by category:")
    for category, count in summary["detections_by_category"].items():
        print(f"  - {category}: {count}")
    print(f"\nDetections by severity:")
    for severity, count in summary["detections_by_severity"].items():
        print(f"  - {severity}: {count}")
    print(f"\nOutput files:")
    print(f"  - Consolidated: {consolidated_path}")
    print(f"  - Summary: {summary_path}")
    print(f"  - Individual frames: {detections_dir}/")
    print(f"{'='*70}\n")

    return {
        "summary": summary,
        "all_detections": all_detections,
        "output_dir": output_dir
    }


def process_video_simple(video_path, output_dir="output", sample_rate=15):
    """
    Simplified video processing with default settings.

    Args:
        video_path: Path to the video file
        output_dir: Output directory
        sample_rate: Frame sampling rate (default: 15 = 2 fps at 30fps)

    Returns:
        dict: Processing results
    """
    return process_video(
        video_path,
        output_dir=output_dir,
        sample_rate=sample_rate,
        debug=False
    )


def _process_video_with_chunks(
    video_path, output_dir, sample_rate,
    chunk_duration, chunk_overlap, cleanup_chunks, debug
):
    """
    Process video in chunks to handle large files.

    Args:
        video_path: Path to input video
        output_dir: Output directory
        sample_rate: Frame sampling rate
        chunk_duration: Chunk duration in seconds
        chunk_overlap: Overlap between chunks in seconds
        cleanup_chunks: Whether to delete chunks after processing
        debug: Debug mode

    Returns:
        dict: Aggregated processing results
    """
    print(f"\n{'='*70}")
    print(f"VIDEO CHUNKING MODE ENABLED")
    print(f"{'='*70}\n")

    # Create chunks directory
    chunks_dir = os.path.join(output_dir, "_chunks")
    os.makedirs(chunks_dir, exist_ok=True)

    # STEP 1: Split video into chunks
    print("STEP 1: Splitting video into chunks...")
    chunks_info = chunk_video(
        video_path,
        chunks_dir,
        chunk_duration=chunk_duration,
        overlap_seconds=chunk_overlap
    )

    # STEP 2: Process each chunk
    all_chunk_results = []

    print(f"\n{'='*70}")
    print(f"STEP 2: Processing {len(chunks_info)} chunks...")
    print(f"{'='*70}\n")

    for i, chunk_info in enumerate(chunks_info):
        print(f"\n{'='*70}")
        print(f"Processing chunk {i+1}/{len(chunks_info)}")
        print(f"{'='*70}")

        chunk_output_dir = os.path.join(output_dir, f"chunk_{i:03d}")

        try:
            # Process chunk using existing function (recursive but with enable_chunking=False)
            chunk_results = process_video(
                chunk_info["chunk_path"],
                output_dir=chunk_output_dir,
                sample_rate=sample_rate,
                start_time=None,  # Already chunked
                end_time=None,
                debug=debug,
                enable_chunking=False  # Prevent recursive chunking
            )

            # Adjust timestamps to account for chunk position in original video
            if chunk_results and "all_detections" in chunk_results:
                for frame_detection in chunk_results["all_detections"]:
                    # Add chunk's start time to frame timestamps (if present)
                    if "timestamp" in frame_detection:
                        frame_detection["timestamp"] += chunk_info["start_time"]
                    # Add chunk index for reference
                    frame_detection["chunk_index"] = chunk_info["chunk_index"]

            all_chunk_results.append({
                "chunk_info": chunk_info,
                "results": chunk_results
            })

            # Force garbage collection after each chunk
            gc.collect()

            # If using CUDA, clear cache
            if HAS_TORCH and torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"ERROR processing chunk {i+1}: {e}")
            # Add empty result for failed chunk
            all_chunk_results.append({
                "chunk_info": chunk_info,
                "results": None,
                "error": str(e)
            })
            continue

    # STEP 3: Aggregate results
    print(f"\n{'='*70}")
    print("STEP 3: Aggregating results from all chunks...")
    print(f"{'='*70}\n")

    aggregated_results = _aggregate_chunk_results(all_chunk_results, video_path, output_dir)

    # STEP 4: Cleanup
    if cleanup_chunks:
        print("\nSTEP 4: Cleaning up chunk files...")
        try:
            shutil.rmtree(chunks_dir)
            print(f"  Removed {chunks_dir}")
        except Exception as e:
            print(f"  Warning: Could not remove chunks directory: {e}")

    return aggregated_results


def _aggregate_chunk_results(chunk_results, original_video_path, output_dir):
    """
    Combine results from all chunks into single output.

    Args:
        chunk_results: List of chunk processing results
        original_video_path: Path to original video
        output_dir: Output directory

    Returns:
        dict: Aggregated results
    """
    all_detections = []
    total_detections = 0
    detections_by_category = {}
    detections_by_severity = {}
    failed_chunks = []

    # Merge all chunk detections
    for chunk_result in chunk_results:
        if chunk_result["results"] is None:
            failed_chunks.append(chunk_result["chunk_info"]["chunk_index"])
            continue

        chunk_summary = chunk_result["results"]["summary"]
        chunk_detections = chunk_result["results"]["all_detections"]

        all_detections.extend(chunk_detections)

        # Aggregate counts
        for category, count in chunk_summary.get("detections_by_category", {}).items():
            detections_by_category[category] = detections_by_category.get(category, 0) + count

        for severity, count in chunk_summary.get("detections_by_severity", {}).items():
            detections_by_severity[severity] = detections_by_severity.get(severity, 0) + count

    # Create consolidated summary
    video_info = get_video_info(original_video_path)

    summary = {
        "total_frames": sum(cr["results"]["summary"]["total_frames"] for cr in chunk_results if cr["results"]),
        "total_detections": sum(len(d.get("detections", [])) for d in all_detections),
        "frames_with_issues": sum(1 for d in all_detections if len(d.get("detections", [])) > 0),
        "detections_by_category": detections_by_category,
        "detections_by_severity": detections_by_severity,
        "num_chunks": len(chunk_results),
        "failed_chunks": len(failed_chunks),
        "video_info": {
            "path": os.path.abspath(original_video_path),
            "filename": os.path.basename(original_video_path),
            "duration": video_info["duration"],
            "fps": video_info["fps"],
            "resolution": f"{video_info['width']}x{video_info['height']}"
        },
        "processing_info": {
            "timestamp": datetime.now().isoformat(),
            "chunked_processing": True,
            "num_chunks": len(chunk_results),
            "frames_processed": sum(cr["results"]["summary"]["total_frames"] for cr in chunk_results if cr["results"])
        }
    }

    # Save consolidated results
    consolidated_path = os.path.join(output_dir, "video_detections_chunked.json")
    save_detection_json({
        "summary": summary,
        "frames": all_detections,
        "chunks": [cr["chunk_info"] for cr in chunk_results]
    }, consolidated_path)

    # Save summary separately
    summary_path = os.path.join(output_dir, "summary.json")
    save_detection_json(summary, summary_path)

    # Print results
    print(f"\nConsolidated Results:")
    print(f"  Total chunks processed: {len(chunk_results) - len(failed_chunks)}/{len(chunk_results)}")
    if failed_chunks:
        print(f"  Failed chunks: {failed_chunks}")
    print(f"  Total frames: {summary['total_frames']}")
    print(f"  Total detections: {summary['total_detections']}")
    print(f"  Frames with issues: {summary['frames_with_issues']}")
    print(f"\nDetections by category:")
    for category, count in summary["detections_by_category"].items():
        print(f"  - {category}: {count}")
    print(f"\nDetections by severity:")
    for severity, count in summary["detections_by_severity"].items():
        print(f"  - {severity}: {count}")
    print(f"\nOutput files:")
    print(f"  - Consolidated: {consolidated_path}")
    print(f"  - Summary: {summary_path}")
    print(f"{'='*70}\n")

    return {
        "summary": summary,
        "all_detections": all_detections,
        "output_dir": output_dir,
        "chunk_results": chunk_results
    }


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python video_processor.py <video_path> [output_dir] [sample_rate]")
        sys.exit(1)

    video_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"
    sample_rate = int(sys.argv[3]) if len(sys.argv) > 3 else 15

    results = process_video(video_path, output_dir, sample_rate, debug=True)
