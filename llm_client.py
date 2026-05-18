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
        self._vllm_endpoint = None  # auto-detected on first call: "chat" or "completions"

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
        headers = {"Content-Type": "application/json"}
        if self.vllm_api_key:
            headers["Authorization"] = f"Bearer {self.vllm_api_key}"

        # Auto-detect the working endpoint on first call
        if self._vllm_endpoint is None:
            self._vllm_endpoint = self._detect_vllm_endpoint(headers)
            print(f"[vLLM] Using endpoint: {self._vllm_endpoint}")

        if self._vllm_endpoint == "chat":
            return self._vllm_chat_completions(model, messages, headers)
        else:
            return self._vllm_completions(model, messages, headers)

    def _detect_vllm_endpoint(self, headers):
        """Probe the server to find which endpoint is available."""
        # Try /v1/chat/completions first
        chat_url = f"{self.vllm_url}/v1/chat/completions"
        try:
            # Send a minimal probe request — we check for 404 specifically
            probe = requests.post(chat_url, json={"model": "probe", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}, headers=headers, timeout=10)
            if probe.status_code != 404:
                return "chat"
        except requests.exceptions.ConnectionError:
            pass

        # Try /v1/completions
        comp_url = f"{self.vllm_url}/v1/completions"
        try:
            probe = requests.post(comp_url, json={"model": "probe", "prompt": "hi", "max_tokens": 1}, headers=headers, timeout=10)
            if probe.status_code != 404:
                return "completions"
        except requests.exceptions.ConnectionError:
            pass

        # Default to chat and let it fail with a clear error
        print("[vLLM] WARNING: Could not detect a working endpoint. Defaulting to /v1/chat/completions.")
        return "chat"

    def _vllm_chat_completions(self, model, messages, headers):
        """Call the OpenAI-compatible /v1/chat/completions endpoint."""
        url = f"{self.vllm_url}/v1/chat/completions"
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

    def _vllm_completions(self, model, messages, headers):
        """Call the /v1/completions endpoint, formatting chat messages into a single prompt."""
        url = f"{self.vllm_url}/v1/completions"
        prompt = self._messages_to_prompt(messages)
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": 8192,
            "temperature": 0.7,
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["text"]

    @staticmethod
    def _messages_to_prompt(messages):
        """Convert a list of chat messages to a single text prompt for the completions API."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        parts.append("Assistant:")
        return "\n\n".join(parts)

    def __repr__(self):
        if self.provider == "ollama":
            return "LLMClient(provider=ollama)"
        endpoint = self._vllm_endpoint or "auto"
        return f"LLMClient(provider=vllm, url={self.vllm_url}, endpoint={endpoint})"
