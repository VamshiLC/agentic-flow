"""
Infrastructure Detection Agent using Qwen3-VL + SAM3
"""
import os
from functools import partial
from sam3.agent.client_llm import send_generate_request
from sam3.agent.client_sam3 import call_sam_service
from sam3.agent.inference import run_single_image_inference


class InfrastructureDetectionAgent:
    """
    Agent that autonomously detects road infrastructure issues using Qwen3-VL and SAM3.

    Based on Meta's official SAM3 Agent pattern:
    https://github.com/facebookresearch/sam3/blob/main/examples/sam3_agent.ipynb
    """

    def __init__(self, sam3_processor, llm_config):
        """
        Initialize the detection agent.

        Args:
            sam3_processor: SAM3 processor instance (Sam3Processor)
            llm_config: Dictionary with LLM configuration
                {
                    "provider": "vllm",
                    "model": "Qwen/Qwen3-VL-4B-Instruct",
                    "server_url": "http://0.0.0.0:8001/v1",
                    "api_key": "DUMMY_API_KEY"
                }
        """
        self.sam3_processor = sam3_processor
        self.llm_config = llm_config

        # Bind SAM3 and LLM clients with partial application
        self.send_generate_request = partial(
            send_generate_request,
            server_url=llm_config["server_url"],
            model=llm_config["model"],
            api_key=llm_config["api_key"]
        )

        self.call_sam_service = partial(
            call_sam_service,
            sam3_processor=sam3_processor
        )

    def process_frame(self, image_path, output_dir="output", debug=True):
        """
        Process a single frame for autonomous infrastructure detection.

        Args:
            image_path: Path to the image file to process
            output_dir: Directory to save output images and results
            debug: If True, print debug information during processing

        Returns:
            output_image_path: Path to the output image with segmentation overlays

        Note:
            The agent operates autonomously - no user prompt is needed.
            The system prompt (in agent/prompts.py) instructs the agent to
            detect all 12 categories of infrastructure issues.
        """
        # Ensure image path is absolute
        image_path = os.path.abspath(image_path)

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Empty prompt - agent decides autonomously based on system prompt
        prompt = ""

        # Run single image inference using SAM3's agent pattern
        output_image_path = run_single_image_inference(
            image_path,
            prompt,
            self.llm_config,
            self.send_generate_request,
            self.call_sam_service,
            debug=debug,
            output_dir=output_dir
        )

        return output_image_path

    def process_multiple_frames(self, image_paths, output_dir="output", debug=False):
        """
        Process multiple frames in batch.

        Args:
            image_paths: List of image paths to process
            output_dir: Directory to save output images and results
            debug: If True, print debug information during processing

        Returns:
            results: List of output image paths
        """
        results = []

        for i, image_path in enumerate(image_paths):
            print(f"Processing frame {i+1}/{len(image_paths)}: {image_path}")

            # Create frame-specific output directory
            frame_output_dir = os.path.join(output_dir, f"frame_{i:06d}")

            try:
                output_path = self.process_frame(
                    image_path,
                    output_dir=frame_output_dir,
                    debug=debug
                )
                results.append(output_path)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append(None)

        return results
