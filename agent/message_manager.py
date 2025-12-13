"""
Message Manager for Infrastructure Detection Agent

Handles conversation history management including:
- Building messages with images
- Message pruning to manage context window
- Tracking used prompts

Based on: https://github.com/facebookresearch/sam3/blob/main/sam3/agent/agent_core.py
"""
import io
import base64
from PIL import Image
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Represents a conversation message."""
    role: str  # "system", "user", "assistant"
    content: Any  # str or list of content parts


class MessageManager:
    """
    Manages conversation history for the agentic loop.

    Features:
    - Build messages with text and images
    - Prune history to fit context window
    - Track segment_phrase calls for warnings
    """

    MAX_IMAGES_PER_REQUEST = 2  # Limit images to avoid context overflow

    def __init__(self, system_prompt: str):
        """
        Initialize message manager.

        Args:
            system_prompt: System prompt for the agent
        """
        self.messages: List[Dict[str, Any]] = []
        self.system_prompt = system_prompt
        self.used_prompts: set = set()
        self.segment_phrase_history: List[Dict] = []  # Track all segment calls

        # Add system message
        self._add_system_message()

    def _add_system_message(self):
        """Add system prompt as first message."""
        self.messages.append({
            "role": "system",
            "content": self.system_prompt
        })

    def add_user_message_with_image(
        self,
        image: Image.Image,
        text: str = "Analyze this road image for infrastructure issues."
    ):
        """
        Add user message with image.

        Args:
            image: PIL Image to include
            text: Text prompt
        """
        image_base64 = self._image_to_base64(image)

        content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_base64}"
                }
            },
            {
                "type": "text",
                "text": text
            }
        ]

        self.messages.append({
            "role": "user",
            "content": content
        })

    def add_tool_result_message(
        self,
        tool_name: str,
        result_message: str,
        result_image: Optional[Image.Image] = None,
        result_data: Optional[Dict] = None
    ):
        """
        Add tool result as user message.

        The tool result is formatted as a user message so the LLM
        can see and respond to it.

        Args:
            tool_name: Name of tool that was called
            result_message: Human-readable result message
            result_image: Optional result image (e.g., rendered masks)
            result_data: Optional structured data
        """
        # Build content parts
        content = []

        # Add result image if provided
        if result_image is not None:
            image_base64 = self._image_to_base64(result_image)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_base64}"
                }
            })

        # Add result text
        result_text = f"[Tool Result: {tool_name}]\n{result_message}"
        if result_data:
            result_text += f"\nData: {json.dumps(result_data, default=str)}"

        content.append({
            "type": "text",
            "text": result_text
        })

        self.messages.append({
            "role": "user",
            "content": content
        })

        # Track segment_phrase calls
        if tool_name == "segment_phrase" and result_data:
            prompt = result_data.get("prompt")
            if prompt:
                self.used_prompts.add(prompt.lower())
                self.segment_phrase_history.append({
                    "prompt": prompt,
                    "num_masks": result_data.get("num_masks", 0)
                })

    def add_assistant_message(self, content: str):
        """
        Add assistant response.

        Args:
            content: Full response from LLM (including <think> and <tool> tags)
        """
        self.messages.append({
            "role": "assistant",
            "content": content
        })

    def get_messages(self) -> List[Dict[str, Any]]:
        """
        Get current messages for API call.

        Returns:
            List of message dicts ready for LLM API
        """
        return self.messages

    def get_messages_for_hf(self) -> List[Dict[str, Any]]:
        """
        Get messages formatted for HuggingFace Transformers.

        Converts OpenAI-style messages to HF format.

        Returns:
            List of message dicts for HF processor
        """
        hf_messages = []

        for msg in self.messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                # HF typically prepends system to first user message
                # or handles it separately
                hf_messages.append({
                    "role": "system",
                    "content": content
                })
            elif isinstance(content, str):
                hf_messages.append({
                    "role": role,
                    "content": content
                })
            elif isinstance(content, list):
                # Multi-modal content
                hf_content = []
                for part in content:
                    if part["type"] == "text":
                        hf_content.append({
                            "type": "text",
                            "text": part["text"]
                        })
                    elif part["type"] == "image_url":
                        # Extract base64 and convert to PIL
                        url = part["image_url"]["url"]
                        if url.startswith("data:image"):
                            # Extract base64 part
                            base64_data = url.split(",")[1]
                            image_bytes = base64.b64decode(base64_data)
                            image = Image.open(io.BytesIO(image_bytes))
                            hf_content.append({
                                "type": "image",
                                "image": image
                            })

                hf_messages.append({
                    "role": role,
                    "content": hf_content
                })

        return hf_messages

    def prune_messages(self, keep_latest_segment: bool = True):
        """
        Prune messages to manage context window.

        Strategy (from official SAM3 agent):
        1. Keep system message (index 0)
        2. Keep initial user message with image (index 1)
        3. Keep latest assistant message with segment_phrase call
        4. Add warning about previously used prompts

        Args:
            keep_latest_segment: If True, keep latest segment_phrase result
        """
        if len(self.messages) <= 3:
            return  # Nothing to prune

        # Find latest segment_phrase assistant message
        latest_segment_idx = None
        for i in range(len(self.messages) - 1, -1, -1):
            msg = self.messages[i]
            if msg["role"] == "assistant":
                content = msg["content"]
                if isinstance(content, str) and "segment_phrase" in content:
                    latest_segment_idx = i
                    break

        # Build pruned messages
        pruned = [
            self.messages[0],  # System
            self.messages[1],  # Initial user + image
        ]

        # Add warning about used prompts
        if self.used_prompts:
            warning = self._build_prompt_warning()
            pruned.append({
                "role": "user",
                "content": warning
            })

        # Add latest segment result if found
        if latest_segment_idx is not None and keep_latest_segment:
            # Add the assistant message
            pruned.append(self.messages[latest_segment_idx])

            # Add the tool result that follows it (if exists)
            if latest_segment_idx + 1 < len(self.messages):
                pruned.append(self.messages[latest_segment_idx + 1])

        self.messages = pruned
        logger.debug(f"Pruned messages to {len(self.messages)}")

    def _build_prompt_warning(self) -> str:
        """Build warning message about used prompts."""
        prompts_list = ", ".join(sorted(self.used_prompts))
        return (
            f"[WARNING] The following text_prompts have already been used and "
            f"must NOT be used again: {prompts_list}\n"
            f"Please use different phrases or synonyms."
        )

    def count_images(self) -> int:
        """Count total images in message history."""
        count = 0
        for msg in self.messages:
            content = msg.get("content", [])
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        count += 1
        return count

    def should_prune(self) -> bool:
        """Check if pruning is needed based on image count."""
        return self.count_images() > self.MAX_IMAGES_PER_REQUEST

    def get_used_prompts(self) -> set:
        """Get set of used segment_phrase prompts."""
        return self.used_prompts

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    def to_debug_dict(self) -> List[Dict]:
        """Convert messages to debug-friendly format (no images)."""
        debug_messages = []
        for msg in self.messages:
            debug_msg = {"role": msg["role"]}
            content = msg["content"]

            if isinstance(content, str):
                debug_msg["content"] = content
            elif isinstance(content, list):
                debug_content = []
                for part in content:
                    if part.get("type") == "image_url":
                        debug_content.append({"type": "image", "data": "[IMAGE]"})
                    else:
                        debug_content.append(part)
                debug_msg["content"] = debug_content

            debug_messages.append(debug_msg)

        return debug_messages

    def save_debug_log(self, filepath: str):
        """Save message history to JSONL for debugging."""
        import json
        with open(filepath, 'w') as f:
            for msg in self.to_debug_dict():
                f.write(json.dumps(msg) + "\n")
        logger.debug(f"Saved debug log to {filepath}")
