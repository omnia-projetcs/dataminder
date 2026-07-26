"""
Unified LLM client abstraction for Dataminder.

Supports two providers:
  - ollama : Local Ollama server (default, existing behavior)
  - vllm   : Remote vLLM server via OpenAI-compatible API
"""

import os
import threading
import time

import requests
from requests.adapters import HTTPAdapter


# Default timeout per LLM call (seconds) and retry settings
DEFAULT_TIMEOUT = 300   # 5 minutes
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 5  # seconds — retry delays: 5s, 10s, 20s
DEFAULT_TEMPERATURE = 0.0


class LLMClient:
    """Thread-safe LLM client that delegates to either Ollama or vLLM."""

    def __init__(
        self,
        provider="ollama",
        vllm_url="http://localhost:8000",
        vllm_api_key="",
        timeout=None,
        ollama_url=None,
        max_pool_connections=32,
        temperature=DEFAULT_TEMPERATURE,
        seed=None,
    ):
        self.provider = provider.lower()
        self.vllm_url = vllm_url.rstrip("/")
        self.vllm_api_key = vllm_api_key
        self.timeout = timeout if timeout is not None else DEFAULT_TIMEOUT
        self.ollama_url = self._normalize_ollama_url(ollama_url)
        self.max_pool_connections = max_pool_connections
        self.temperature = float(temperature)
        self.seed = int(seed) if seed is not None else None
        self._thread_local = threading.local()
        self._vllm_endpoint = None  # "chat" or "completions"
        self._vllm_model = None     # auto-detected from /v1/models
        self._detect_lock = threading.Lock()

        if self.provider not in ("ollama", "vllm"):
            raise ValueError(f"Unknown provider '{self.provider}'. Use 'ollama' or 'vllm'.")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2")

    def generation_config(self):
        """Return non-secret settings that affect generated output."""
        return {
            "provider": self.provider,
            "endpoint": (
                self.ollama_url if self.provider == "ollama" else self.vllm_url
            ),
            "temperature": self.temperature,
            "seed": self.seed,
        }

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
            except (TimeoutError, requests.Timeout) as e:
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
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": "."}],
                "keep_alive": 0,
                "stream": False,
            }
            self._session().post(f"{self.ollama_url}/api/chat", json=payload, timeout=30)
        except Exception:
            pass

    @staticmethod
    def _normalize_ollama_url(ollama_url):
        """Return a base Ollama URL without API path suffixes."""
        base_url = ollama_url or os.environ.get("OLLAMA_HOST") or "http://localhost:11434"
        base_url = base_url.rstrip("/")
        if base_url.endswith("/api"):
            base_url = base_url[:-4]
        return base_url

    def _session(self):
        """Return a thread-local HTTP session with a shared connection-pool size.

        requests.Session is not guaranteed to be thread-safe, so each worker
        thread gets its own session while still benefiting from HTTP keep-alive
        and adapter-level connection pooling.
        """
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = requests.Session()
            adapter = HTTPAdapter(
                pool_connections=self.max_pool_connections,
                pool_maxsize=self.max_pool_connections,
            )
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            self._thread_local.session = session
        return session

    def _chat_ollama(self, model, messages, keep_alive):
        """Call Ollama's HTTP API with native request timeouts.

        Using the HTTP API directly avoids spawning a nested one-off executor for
        every call. This is cheaper under --threads and lets each model worker
        reuse its thread-local HTTP connection to Ollama.
        """
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        payload["options"] = {"temperature": self.temperature}
        if self.seed is not None:
            payload["options"]["seed"] = self.seed

        resp = self._session().post(
            f"{self.ollama_url}/api/chat",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "")

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
            resp = self._session().get(url, headers=headers, timeout=30)
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
            resp = self._session().post(chat_url, json=probe_payload, headers=headers, timeout=30)
        except requests.RequestException as e:
            print(f"[vLLM] Could not probe /v1/chat/completions ({e}), assuming /v1/completions")
            self._vllm_endpoint = "completions"
            return

        if resp.status_code == 404:
            print("[vLLM] /v1/chat/completions not found (404), switching to /v1/completions")
            self._vllm_endpoint = "completions"
        else:
            self._vllm_endpoint = "chat"
            print("[vLLM] Using endpoint: /v1/chat/completions")

    def _call_chat(self, model, messages, headers):
        """Call the OpenAI-compatible /v1/chat/completions endpoint."""
        url = f"{self.vllm_url}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": 8192,
            "temperature": self.temperature,
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        resp = self._session().post(url, json=payload, headers=headers, timeout=self.timeout)
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
            "temperature": self.temperature,
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        resp = self._session().post(url, json=payload, headers=headers, timeout=self.timeout)
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
            return (
                f"LLMClient(provider=ollama, url={self.ollama_url}, "
                f"temperature={self.temperature}, seed={self.seed}, "
                f"timeout={self.timeout}s)"
            )
        endpoint = self._vllm_endpoint or "auto"
        model = self._vllm_model or "auto"
        return (
            f"LLMClient(provider=vllm, url={self.vllm_url}, endpoint={endpoint}, "
            f"model={model}, temperature={self.temperature}, seed={self.seed}, "
            f"timeout={self.timeout}s)"
        )
