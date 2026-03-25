from .grok_node import GrokVisionNode, GrokPromptBuilderNode

NODE_CLASS_MAPPINGS = {
    "GrokVisionNode": GrokVisionNode,
    "GrokPromptBuilderNode": GrokPromptBuilderNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GrokVisionNode": "Grok Vision",
    "GrokPromptBuilderNode": "Grok Prompt Builder",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
