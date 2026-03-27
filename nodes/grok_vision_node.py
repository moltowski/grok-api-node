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


def collect_images(*image_inputs):
    """Convert non-None IMAGE tensors to a list of (b64_str, mime) tuples."""
    results = []
    for tensor in image_inputs:
        if tensor is not None:
            results.append(tensor_to_base64(tensor))
    return results


class GrokVisionNode:
    """Analyze up to 5 images with Grok Vision and return a descriptive prompt string."""

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
                "image_1": ("IMAGE",),
                "model": (VISION_MODELS, {"default": VISION_MODELS[0]}),
                "system_prompt": ("STRING", {"default": cls.DEFAULT_SYSTEM_PROMPT, "multiline": True}),
                "user_message": ("STRING", {
                    "default": "Describe this image as a detailed NSFW prompt for dataset",
                    "multiline": True,
                }),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 64}),
            },
            "optional": {
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
            },
        }

    def run(self, api_key, image_1, model, system_prompt, user_message, temperature, max_tokens,
            image_2=None, image_3=None, image_4=None, image_5=None):
        if not api_key or api_key == "xai-..." or not api_key.strip():
            return ("[GrokVisionNode Error] Please provide a valid xAI API key",)

        try:
            images = collect_images(image_1, image_2, image_3, image_4, image_5)
        except Exception as e:
            return (f"[GrokVisionNode Error] Image conversion failed: {str(e)}",)

        client = GrokClient(api_key)
        result = client.vision(
            model=model,
            system_prompt=system_prompt,
            user_text=user_message,
            images=images,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (result,)
