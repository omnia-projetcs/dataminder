"""
Unified LLM client abstraction for Dataminder.

Supports two providers:
  - ollama : Local Ollama server (default, existing behavior)
  - vllm   : Remote vLLM server via OpenAI-compatible API
"""

import threading
import time
import requests
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError


# Default timeout per LLM call (seconds) and retry settings
DEFAULT_TIMEOUT = 300   # 5 minutes
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 5  # seconds — retry delays: 5s, 10s, 20s


class LLMClient:
    """Thread-safe LLM client that delegates to either Ollama or vLLM."""

    def __init__(self, provider="ollama", vllm_url="http://localhost:8000", vllm_api_key="", timeout=None):
        self.provider = provider.lower()
        self.vllm_url = vllm_url.rstrip("/")
        self.vllm_api_key = vllm_api_key
        self.timeout = timeout if timeout is not None else DEFAULT_TIMEOUT
        self._vllm_endpoint = None  # "chat" or "completions"
        self._vllm_model = None     # auto-detected from /v1/models
        self._detect_lock = threading.Lock()

        if self.provider not in ("ollama", "vllm"):
            raise ValueError(f"Unknown provider '{self.provider}'. Use 'ollama' or 'vllm'.")

    def chat(self, model, messages, keep_alive=None):
        """
        Send a chat completion request and return the assistant's response text.
        Includes automatic retry with exponential backoff on failure/timeout.

        Args:
            model: Model name/identifier.
            messages: List of dicts with 'role' and 'content' keys.
            keep_alive: Ollama-specific parameter (ignored for vLLM).

        Returns:
            The response text (str).

        Raises:
            TimeoutError: If the LLM call exceeds the timeout after all retries.
            Exception: If the LLM call fails after all retries.
        """
        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                if self.provider == "ollama":
                    return self._chat_ollama(model, messages, keep_alive)
                else:
                    return self._chat_vllm(model, messages)
            except (TimeoutError, FuturesTimeoutError) as e:
                last_error = e
                if attempt < MAX_RETRIES:
                    delay = RETRY_BACKOFF_BASE * (2 ** (attempt - 1))
                    print(f"  [TIMEOUT] LLM call timed out after {self.timeout}s (attempt {attempt}/{MAX_RETRIES}). Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    print(f"  [TIMEOUT] LLM call timed out after {self.timeout}s (attempt {attempt}/{MAX_RETRIES}). Giving up.")
            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES:
                    delay = RETRY_BACKOFF_BASE * (2 ** (attempt - 1))
                    print(f"  [RETRY] LLM call failed: {e} (attempt {attempt}/{MAX_RETRIES}). Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    print(f"  [RETRY] LLM call failed: {e} (attempt {attempt}/{MAX_RETRIES}). Giving up.")

        raise last_error

    def unload_model(self, model):
        """Unload the model from VRAM. Only relevant for Ollama."""
        if self.provider != "ollama":
            return
        try:
            import ollama
            ollama.chat(model=model, messages=[
                {'role': 'user', 'content': '.'}
            ], keep_alive=0)
        except Exception:
            pass

    def _chat_ollama(self, model, messages, keep_alive):
        """Call Ollama with a thread-based timeout wrapper.

        The ollama Python library has no native timeout, so we run the blocking
        call inside a thread and enforce a deadline via future.result(timeout=...).
        """
        import ollama as ollama_lib

        def _blocking_call():
            kwargs = {"model": model, "messages": messages}
            if keep_alive is not None:
                kwargs["keep_alive"] = keep_alive
            response = ollama_lib.chat(**kwargs)
            return response['message']['content']

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_blocking_call)
            try:
                return future.result(timeout=self.timeout)
            except FuturesTimeoutError:
                raise TimeoutError(
                    f"Ollama chat call exceeded {self.timeout}s timeout"
                )

    def _chat_vllm(self, model, messages):
        headers = {"Content-Type": "application/json"}
        if self.vllm_api_key:
            headers["Authorization"] = f"Bearer {self.vllm_api_key}"

        # Ensure endpoint + model detection is done once, thread-safe
        if self._vllm_endpoint is None:
            with self._detect_lock:
                # Double-check after acquiring lock
                if self._vllm_endpoint is None:
                    self._detect_model(headers)
                    self._detect_endpoint(self._vllm_model or model, headers)

        # Use auto-detected model if available
        resolved_model = self._vllm_model or model

        # Use the detected endpoint
        if self._vllm_endpoint == "completions":
            return self._call_completions(resolved_model, messages, headers)
        else:
            return self._call_chat(resolved_model, messages, headers)

    def _detect_model(self, headers):
        """Query /v1/models to auto-detect the loaded model on the vLLM server."""
        url = f"{self.vllm_url}/v1/models"
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                models = data.get("data", [])
                if models:
                    self._vllm_model = models[0].get("id", None)
                    print(f"[vLLM] Auto-detected model: {self._vllm_model}")
                else:
                    print("[vLLM] /v1/models returned empty list, using --model value.")
            else:
                print(f"[vLLM] /v1/models returned {resp.status_code}, using --model value.")
        except Exception as e:
            print(f"[vLLM] Could not query /v1/models ({e}), using --model value.")

    def _detect_endpoint(self, model, headers):
        """Detect which endpoint the vLLM server supports using a lightweight probe.

        Sends a minimal request (max_tokens=1) to /v1/chat/completions.
        If it returns 404, fall back to /v1/completions.
        The probe response is discarded since it's a throwaway check.
        """
        chat_url = f"{self.vllm_url}/v1/chat/completions"
        probe_payload = {
            "model": model,
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 1,
        }

        try:
            resp = requests.post(chat_url, json=probe_payload, headers=headers, timeout=30)
        except requests.RequestException as e:
            print(f"[vLLM] Could not probe /v1/chat/completions ({e}), assuming /v1/completions")
            self._vllm_endpoint = "completions"
            return

        if resp.status_code == 404:
            print(f"[vLLM] /v1/chat/completions not found (404), switching to /v1/completions")
            self._vllm_endpoint = "completions"
        else:
            self._vllm_endpoint = "chat"
            print(f"[vLLM] Using endpoint: /v1/chat/completions")

    def _call_chat(self, model, messages, headers):
        """Call the OpenAI-compatible /v1/chat/completions endpoint."""
        url = f"{self.vllm_url}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": 8192,
            "temperature": 0.7,
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def _call_completions(self, model, messages, headers):
        """Call the /v1/completions endpoint, formatting chat messages into a single prompt."""
        url = f"{self.vllm_url}/v1/completions"
        prompt = self._messages_to_prompt(messages)
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": 8192,
            "temperature": 0.7,
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
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
            return f"LLMClient(provider=ollama, timeout={self.timeout}s)"
        endpoint = self._vllm_endpoint or "auto"
        model = self._vllm_model or "auto"
        return f"LLMClient(provider=vllm, url={self.vllm_url}, endpoint={endpoint}, model={model}, timeout={self.timeout}s)"
