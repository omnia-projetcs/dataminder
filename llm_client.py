"""
Unified LLM client abstraction for Dataminder.

Supports two providers:
  - ollama : Local Ollama server (default, existing behavior)
  - vllm   : Remote vLLM server via OpenAI-compatible API
"""

import ollama
import requests


class LLMClient:
    """Thread-safe LLM client that delegates to either Ollama or vLLM."""

    def __init__(self, provider="ollama", vllm_url="http://localhost:8000", vllm_api_key=""):
        self.provider = provider.lower()
        self.vllm_url = vllm_url.rstrip("/")
        self.vllm_api_key = vllm_api_key

        if self.provider not in ("ollama", "vllm"):
            raise ValueError(f"Unknown provider '{self.provider}'. Use 'ollama' or 'vllm'.")

    def chat(self, model, messages, keep_alive=None):
        """
        Send a chat completion request and return the assistant's response text.

        Args:
            model: Model name/identifier.
            messages: List of dicts with 'role' and 'content' keys.
            keep_alive: Ollama-specific parameter (ignored for vLLM).

        Returns:
            The response text (str).
        """
        if self.provider == "ollama":
            return self._chat_ollama(model, messages, keep_alive)
        else:
            return self._chat_vllm(model, messages)

    def unload_model(self, model):
        """Unload the model from VRAM. Only relevant for Ollama."""
        if self.provider != "ollama":
            return
        try:
            ollama.chat(model=model, messages=[
                {'role': 'user', 'content': '.'}
            ], keep_alive=0)
        except Exception:
            pass

    def _chat_ollama(self, model, messages, keep_alive):
        kwargs = {"model": model, "messages": messages}
        if keep_alive is not None:
            kwargs["keep_alive"] = keep_alive
        response = ollama.chat(**kwargs)
        return response['message']['content']

    def _chat_vllm(self, model, messages):
        url = f"{self.vllm_url}/v1/chat/completions"

        headers = {"Content-Type": "application/json"}
        if self.vllm_api_key:
            headers["Authorization"] = f"Bearer {self.vllm_api_key}"

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": 8192,
            "temperature": 0.7,
        }

        resp = requests.post(url, json=payload, headers=headers, timeout=300)
        resp.raise_for_status()

        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def __repr__(self):
        if self.provider == "ollama":
            return "LLMClient(provider=ollama)"
        return f"LLMClient(provider=vllm, url={self.vllm_url})"
