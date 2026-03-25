from ..utils.grok_client import GrokClient


PROMPT_MODELS = [
    "grok-4-1-fast-non-reasoning",
    "grok-4-1-fast-reasoning",
]


class GrokPromptBuilderNode:
    """Generate N prompts from example lists + a guide instruction using Grok."""

    CATEGORY = "Grok API"
    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("prompts_text", "prompt_count")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "xai-...", "multiline": False, "password": True}),
                "model": (PROMPT_MODELS, {"default": PROMPT_MODELS[0]}),
                "guide_prompt": ("STRING", {
                    "default": "Generate photorealistic portrait prompts with detailed descriptions",
                    "multiline": True,
                }),
                "examples_1": ("STRING", {"default": "", "multiline": True}),
                "num_prompts": ("INT", {"default": 10, "min": 1, "max": 50, "step": 1}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 2048, "min": 256, "max": 8192, "step": 128}),
                "output_separator": ("STRING", {"default": "\n---\n", "multiline": False}),
            },
            "optional": {
                "examples_2": ("STRING", {"default": "", "multiline": True}),
                "examples_3": ("STRING", {"default": "", "multiline": True}),
                "examples_4": ("STRING", {"default": "", "multiline": True}),
                "examples_5": ("STRING", {"default": "", "multiline": True}),
            },
        }

    def build_messages(self, guide, examples_lists, num_prompts):
        all_examples = []
        for lst in examples_lists:
            if lst and lst.strip():
                lines = [line.strip() for line in lst.strip().split("\n") if line.strip()]
                all_examples.extend(lines)

        system = (
            "You are an expert prompt engineer for AI image generation. "
            "Your task is to generate new prompts that match the style, vocabulary, and structure "
            "of the provided examples. "
            "Output ONLY the prompts, one per line, no numbering, no explanations."
        )

        examples_section = ""
        if all_examples:
            examples_section = "Here are example prompts to use as style reference:\n\n"
            examples_section += "\n".join(f"- {ex}" for ex in all_examples)
            examples_section += "\n\n"

        user = (
            f"{examples_section}"
            f"Guide/theme for the new prompts:\n{guide}\n\n"
            f"Generate exactly {num_prompts} new prompts following the same style as the examples above.\n"
            "Output only the prompts, one per line."
        )

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def run(self, api_key, model, guide_prompt, examples_1, num_prompts, temperature,
            max_tokens, output_separator, examples_2="", examples_3="",
            examples_4="", examples_5=""):

        if not api_key or api_key == "xai-..." or not api_key.strip():
            return ("[GrokPromptBuilderNode Error] Please provide a valid xAI API key", 0)

        if not guide_prompt or not guide_prompt.strip():
            return ("[GrokPromptBuilderNode Error] Please provide a guide prompt", 0)

        examples_lists = [examples_1, examples_2, examples_3, examples_4, examples_5]
        messages = self.build_messages(guide_prompt, examples_lists, num_prompts)

        client = GrokClient(api_key)
        result = client.chat(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)

        if result.startswith("[Grok"):
            return (result, 0)

        # Parse lines and strip common list prefixes
        lines = [line.strip() for line in result.strip().split("\n") if line.strip()]
        clean_prompts = []
        for line in lines:
            if len(line) > 2:
                if line[0].isdigit() and line[1] in [".", ")", ":"]:
                    line = line[2:].strip()
                elif line[0] in ["-", "*", "•"]:
                    line = line[1:].strip()
            if line:
                clean_prompts.append(line)

        prompts_text = output_separator.join(clean_prompts)
        return (prompts_text, len(clean_prompts))
