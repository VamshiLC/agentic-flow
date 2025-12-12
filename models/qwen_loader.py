"""
Qwen3-VL Model Configuration

Configures connection to Qwen3-VL-4B-Instruct served via vLLM.
"""
import os


def get_qwen_config(
    model="Qwen/Qwen3-VL-4B-Instruct",
    server_url="http://0.0.0.0:8001/v1",
    api_key=None
):
    """
    Get Qwen3-VL configuration for vLLM server.

    Args:
        model: Hugging Face model ID for Qwen3-VL
        server_url: URL of the vLLM server endpoint
        api_key: API key for authentication (default: DUMMY_API_KEY)

    Returns:
        dict: Configuration dictionary for LLM client

    Notes:
        vLLM server should be started before running the agent:
        ```bash
        vllm serve Qwen/Qwen3-VL-4B-Instruct \
            --tensor-parallel-size 1 \
            --allowed-local-media-path / \
            --enforce-eager \
            --port 8001
        ```
    """
    if api_key is None:
        # Check environment variable, fallback to dummy key
        api_key = os.getenv("LLM_API_KEY", "DUMMY_API_KEY")

    config = {
        "provider": "vllm",
        "model": model,
        "server_url": server_url,
        "api_key": api_key,
        "name": "qwen3-vl-4b"
    }

    return config


def get_available_models():
    """
    Get list of available Qwen3-VL model variants.

    Returns:
        dict: Dictionary of model IDs with descriptions
    """
    return {
        "Qwen/Qwen3-VL-4B-Instruct": {
            "description": "4B parameter vision-language model, good balance of speed and accuracy",
            "vram": "~8GB",
            "recommended": True
        },
        "Qwen/Qwen3-VL-2B-Instruct": {
            "description": "2B parameter model, fastest option",
            "vram": "~6GB",
            "recommended": False
        },
        "Qwen/Qwen3-VL-8B-Instruct": {
            "description": "8B parameter model, best accuracy but slower",
            "vram": "~16GB",
            "recommended": False
        },
        "Qwen/Qwen3-VL-4B-Thinking": {
            "description": "4B model with enhanced reasoning capabilities",
            "vram": "~8GB",
            "recommended": False
        }
    }


def validate_server_connection(server_url="http://0.0.0.0:8001/v1"):
    """
    Validate that the vLLM server is running and accessible.

    Args:
        server_url: URL of the vLLM server

    Returns:
        bool: True if server is accessible, False otherwise
    """
    try:
        import requests
        # Try to access the health endpoint
        health_url = server_url.replace("/v1", "/health")
        response = requests.get(health_url, timeout=5)
        if response.status_code == 200:
            print(f"✓ vLLM server is running at {server_url}")
            return True
        else:
            print(f"✗ vLLM server returned status code {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Cannot connect to vLLM server at {server_url}: {e}")
        print("\nTo start the vLLM server, run:")
        print("vllm serve Qwen/Qwen3-VL-4B-Instruct --tensor-parallel-size 1 --allowed-local-media-path / --enforce-eager --port 8001")
        return False
