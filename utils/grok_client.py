import requests
import json
from typing import Dict, List, Any


class GrokClient:
    """Client for interacting with xAI Grok API (OpenAI-compatible format)."""

    BASE_URL = "https://api.x.ai/v1/chat/completions"

    def __init__(self, api_key: str):
        self.api_key = api_key.strip()
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def chat(self, model: str, messages: List[Dict[str, Any]],
             temperature: float = 0.7, max_tokens: int = 1024) -> str:
        try:
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            response = requests.post(
                self.BASE_URL,
                headers=self.headers,
                json=payload,
                timeout=120,
            )

            if response.status_code != 200:
                try:
                    error_msg = response.json().get("error", {}).get("message", response.text[:200])
                except Exception:
                    error_msg = response.text[:200]
                return f"[Grok API Error {response.status_code}] {error_msg}"

            try:
                return response.json()["choices"][0]["message"]["content"]
            except (KeyError, IndexError, json.JSONDecodeError) as e:
                return f"[Grok Response Parse Error] {str(e)} | Response: {response.text[:200]}"

        except requests.exceptions.Timeout:
            return "[Grok API Error] Request timeout (120s exceeded)"
        except requests.exceptions.ConnectionError:
            return "[Grok API Error] Connection failed"
        except Exception as e:
            return f"[Grok API Error] {type(e).__name__}: {str(e)}"

    def vision(self, model: str, system_prompt: str, user_text: str,
               images: list, temperature: float = 0.7, max_tokens: int = 1024) -> str:
        image_blocks = [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
            for b64, mime in images
        ]
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [{"type": "text", "text": user_text}] + image_blocks,
            },
        ]
        return self.chat(model, messages, temperature, max_tokens)
