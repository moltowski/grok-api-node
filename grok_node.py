import io
import base64
import requests
import numpy as np
from PIL import Image


XAI_BASE_URL = "https://api.x.ai/v1/chat/completions"


def _call_grok(api_key: str, model: str, messages: list, temperature: float, max_tokens: int) -> str:
    resp = requests.post(
        XAI_BASE_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _tensor_to_base64_jpeg(image_tensor) -> str:
    """Convert a ComfyUI IMAGE tensor [B,H,W,C] or [H,W,C] float32 0-1 to base64 JPEG."""
    t = image_tensor
    if len(t.shape) == 4:
        t = t[0]
    np_img = (t.cpu().numpy() * 255).astype(np.uint8)
    pil_img = Image.fromarray(np_img)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class GrokVisionNode:
    """Analyze an image with Grok Vision and return a descriptive prompt string."""

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "run"
    CATEGORY = "Grok API"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "password": True}),
                "image": ("IMAGE",),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "You are an expert at writing detailed NSFW dataset prompts. Analyze the image and produce a single, highly detailed prompt describing it.",
                }),
                "user_message": ("STRING", {
                    "multiline": True,
                    "default": "Describe this image as a detailed dataset prompt.",
                }),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 64}),
            }
        }

    def run(self, api_key, image, system_prompt, user_message, temperature, max_tokens):
        if not api_key.strip():
            raise ValueError("GrokVisionNode: api_key is required")

        b64 = _tensor_to_base64_jpeg(image)

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}",
                            "detail": "high",
                        },
                    },
                    {"type": "text", "text": user_message},
                ],
            },
        ]

        result = _call_grok(api_key, "grok-2-vision-1212", messages, temperature, max_tokens)
        return (result,)


class GrokPromptBuilderNode:
    """Generate N prompts from example lists + a guide instruction using Grok."""

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("prompts_text", "prompt_count")
    FUNCTION = "run"
    CATEGORY = "Grok API"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "password": True}),
                "guide_prompt": ("STRING", {
                    "multiline": True,
                    "default": "Generate prompts in the same style and format as the examples provided.",
                }),
                "examples_1": ("STRING", {"multiline": True, "default": ""}),
                "num_prompts": ("INT", {"default": 10, "min": 1, "max": 50, "step": 1}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 2048, "min": 64, "max": 16384, "step": 64}),
                "output_separator": ("STRING", {"default": "\\n---\\n"}),
            },
            "optional": {
                "examples_2": ("STRING", {"multiline": True, "default": ""}),
                "examples_3": ("STRING", {"multiline": True, "default": ""}),
                "examples_4": ("STRING", {"multiline": True, "default": ""}),
                "examples_5": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    def run(
        self,
        api_key,
        guide_prompt,
        examples_1,
        num_prompts,
        temperature,
        max_tokens,
        output_separator,
        examples_2="",
        examples_3="",
        examples_4="",
        examples_5="",
    ):
        if not api_key.strip():
            raise ValueError("GrokPromptBuilderNode: api_key is required")

        # Collect non-empty example lists
        all_examples = [e for e in [examples_1, examples_2, examples_3, examples_4, examples_5] if e.strip()]
        if not all_examples:
            raise ValueError("GrokPromptBuilderNode: at least one example list is required")

        # Build separator — convert escape sequences so \n becomes an actual newline
        sep = output_separator.replace("\\n", "\n").replace("\\t", "\t")

        # Build the example block
        example_block = ""
        for i, ex in enumerate(all_examples, 1):
            example_block += f"--- Example List {i} ---\n{ex.strip()}\n\n"

        system_content = (
            "You are an expert prompt writer. "
            "You will study the provided example prompts and generate new prompts that match their style, "
            "format, vocabulary, and tone exactly.\n\n"
            f"Instructions: {guide_prompt}"
        )

        user_content = (
            f"Here are example prompts to use as style reference:\n\n{example_block}"
            f"Generate exactly {num_prompts} new prompts in the same style.\n"
            f"Separate each prompt with: {repr(sep)}\n"
            "Output only the prompts — no numbering, no preamble, no explanation."
        )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

        result = _call_grok(api_key, "grok-3-mini-beta", messages, temperature, max_tokens)

        # Count prompts based on separator
        parts = [p.strip() for p in result.split(sep) if p.strip()]
        prompt_count = len(parts)

        return (result, prompt_count)
