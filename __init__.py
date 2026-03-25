from .nodes.grok_vision_node import GrokVisionNode
from .nodes.grok_prompt_builder_node import GrokPromptBuilderNode


NODE_CLASS_MAPPINGS = {
    "GrokVisionNode": GrokVisionNode,
    "GrokPromptBuilderNode": GrokPromptBuilderNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GrokVisionNode": "Grok Vision to Prompt",
    "GrokPromptBuilderNode": "Grok Prompt Builder",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
