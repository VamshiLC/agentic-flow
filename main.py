#!/usr/bin/env python3
"""
ASH Infrastructure Detection Agent - CLI Entry Point

Command-line interface for processing GoPro videos and images
to detect road infrastructure issues.
"""
import argparse
import sys
import json
from pathlib import Path

from inference.single_frame import process_single_frame
from inference.video_processor import process_video
from models.qwen_loader import validate_server_connection, get_available_models
from config import get_config


def check_vllm_server(config):
    """Check if vLLM server is running"""
    print("Checking vLLM server connection...")
    if not validate_server_connection(config.QWEN_SERVER_URL):
        print("\nERROR: vLLM server is not running!")
        print("\nTo start the vLLM server, run:")
        print(f"vllm serve {config.QWEN_MODEL} \\")
        print("    --tensor-parallel-size 1 \\")
        print("    --allowed-local-media-path / \\")
        print("    --enforce-eager \\")
        print("    --port 8001")
        return False
    return True


def process_image_mode(args, config):
    """Process single image"""
    print(f"\nProcessing image: {args.input}")

    result = process_single_frame(
        args.input,
        output_dir=args.output,
        save_json=True,
        debug=args.debug
    )

    if args.json:
        print("\n" + json.dumps(result, indent=2))

    return 0


def process_video_mode(args, config):
    """Process video"""
    print(f"\nProcessing video: {args.input}")

    results = process_video(
        args.input,
        output_dir=args.output,
        sample_rate=args.sample_rate,
        start_time=args.start_time,
        end_time=args.end_time,
        debug=args.debug,
        # Chunking parameters
        enable_chunking=args.enable_chunking,
        chunk_duration=args.chunk_duration,
        chunk_overlap=args.chunk_overlap,
        cleanup_chunks=not args.keep_chunks
    )

    if args.json and results:
        print("\n" + json.dumps(results["summary"], indent=2))

    return 0


def list_models():
    """List available Qwen3-VL models"""
    print("\nAvailable Qwen3-VL models:")
    print("=" * 70)

    models = get_available_models()
    for model_id, info in models.items():
        recommended = " (RECOMMENDED)" if info.get("recommended") else ""
        print(f"\n{model_id}{recommended}")
        print(f"  Description: {info['description']}")
        print(f"  VRAM required: {info['vram']}")

    print("\n" + "=" * 70)
    return 0


def main():
    """Main CLI entry point"""
    config = get_config()

    parser = argparse.ArgumentParser(
        description="ASH Infrastructure Detection Agent - Detect road infrastructure issues using Qwen3-VL + SAM3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single image
  python main.py --mode image --input gopro_frame.jpg --output ./results

  # Process a video (1 frame per second)
  python main.py --mode video --input gopro_video.mp4 --output ./results --sample-rate 30

  # Process video with time range
  python main.py --mode video --input video.mp4 --start-time 10 --end-time 60

  # List available models
  python main.py --list-models
        """
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["video", "image"],
        help="Processing mode: video or image"
    )

    # Input/output
    parser.add_argument(
        "--input",
        help="Path to input video or image file"
    )
    parser.add_argument(
        "--output",
        default=config.OUTPUT_DIR,
        help=f"Output directory (default: {config.OUTPUT_DIR})"
    )

    # Video processing options
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=config.VIDEO_SAMPLE_RATE,
        help=f"Frame sampling rate for videos (default: {config.VIDEO_SAMPLE_RATE}, i.e., 2 fps at 30fps)"
    )
    parser.add_argument(
        "--start-time",
        type=float,
        help="Start time in seconds (for videos)"
    )
    parser.add_argument(
        "--end-time",
        type=float,
        help="End time in seconds (for videos)"
    )

    # Video chunking options (for large files)
    parser.add_argument(
        "--enable-chunking",
        action="store_true",
        help="Enable video chunking for large files (splits into smaller segments)"
    )
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=600.0,
        help="Duration of each chunk in seconds (default: 600 = 10 minutes)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=float,
        default=1.0,
        help="Overlap between chunks in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--keep-chunks",
        action="store_true",
        help="Keep chunk files after processing (default: delete chunks)"
    )

    # Output options
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON output to console"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with verbose logging"
    )

    # Model options
    parser.add_argument(
        "--server-url",
        default=config.QWEN_SERVER_URL,
        help=f"vLLM server URL (default: {config.QWEN_SERVER_URL})"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=config.SAM3_CONFIDENCE_THRESHOLD,
        help=f"SAM3 confidence threshold (default: {config.SAM3_CONFIDENCE_THRESHOLD})"
    )

    # Utility options
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available Qwen3-VL models"
    )
    parser.add_argument(
        "--check-server",
        action="store_true",
        help="Check if vLLM server is running"
    )

    args = parser.parse_args()

    # Handle utility commands
    if args.list_models:
        return list_models()

    if args.check_server:
        check_vllm_server(config)
        return 0

    # Validate required arguments
    if not args.mode or not args.input:
        parser.print_help()
        print("\nERROR: --mode and --input are required")
        return 1

    # Validate input file exists
    if not Path(args.input).exists():
        print(f"\nERROR: Input file not found: {args.input}")
        return 1

    # Check vLLM server connection
    if not check_vllm_server(config):
        return 1

    # Update config with CLI arguments
    if args.server_url != config.QWEN_SERVER_URL:
        config.QWEN_SERVER_URL = args.server_url
    if args.confidence != config.SAM3_CONFIDENCE_THRESHOLD:
        config.SAM3_CONFIDENCE_THRESHOLD = args.confidence

    # Process based on mode
    try:
        if args.mode == "image":
            return process_image_mode(args, config)
        elif args.mode == "video":
            return process_video_mode(args, config)
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user")
        return 130
    except Exception as e:
        print(f"\nERROR: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
