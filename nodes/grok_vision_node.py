import numpy as np
from PIL import Image
import base64
import io

from ..utils.grok_client import GrokClient


VISION_MODELS = [
    "grok-4-1-fast-non-reasoning",
    "grok-4-1-fast-reasoning",
]


def tensor_to_base64(image_tensor):
    """Convert ComfyUI IMAGE tensor [B,H,W,C] float32 0-1 to base64 PNG."""
    img_np = (image_tensor[0].numpy() * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np, mode="RGB")
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8"), "image/png"


class GrokVisionNode:
    """Analyze an image with Grok Vision and return a descriptive prompt string."""

    CATEGORY = "Grok API"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "run"

    DEFAULT_SYSTEM_PROMPT = (
        "You are an expert at writing detailed, descriptive prompts for AI image generation datasets. "
        "Analyze the provided image and generate a detailed NSFW prompt describing its content. "
        "Be explicit, precise, and use comma-separated tags and natural language. "
        "Focus on: subject, pose, clothing/nudity, body details, setting, lighting, style, quality tags. "
        "Output ONLY the prompt, no explanations, no preamble."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "xai-...", "multiline": False, "password": True}),
                "image": ("IMAGE",),
                "model": (VISION_MODELS, {"default": VISION_MODELS[0]}),
                "system_prompt": ("STRING", {"default": cls.DEFAULT_SYSTEM_PROMPT, "multiline": True}),
                "user_message": ("STRING", {
                    "default": "Describe this image as a detailed NSFW prompt for dataset",
                    "multiline": True,
                }),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 64}),
            }
        }

    def run(self, api_key, image, model, system_prompt, user_message, temperature, max_tokens):
        if not api_key or api_key == "xai-..." or not api_key.strip():
            return ("[GrokVisionNode Error] Please provide a valid xAI API key",)

        try:
            b64_image, mime_type = tensor_to_base64(image)
        except Exception as e:
            return (f"[GrokVisionNode Error] Image conversion failed: {str(e)}",)

        client = GrokClient(api_key)
        result = client.vision(
            model=model,
            system_prompt=system_prompt,
            user_text=user_message,
            image_base64=b64_image,
            image_mime=mime_type,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (result,)
