import unittest
from unittest.mock import patch

from llm_client import LLMClient


class _Response:
    status_code = 200

    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self._data


class _Session:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


class LLMClientGenerationTests(unittest.TestCase):
    def test_ollama_payload_includes_reproducibility_settings(self):
        client = LLMClient(
            provider="ollama",
            temperature=0.25,
            seed=42,
        )
        session = _Session(_Response({"message": {"content": "answer"}}))

        with patch.object(client, "_session", return_value=session):
            result = client._chat_ollama(
                "model",
                [{"role": "user", "content": "question"}],
                None,
            )

        self.assertEqual(result, "answer")
        payload = session.calls[0][1]["json"]
        self.assertEqual(payload["options"], {"temperature": 0.25, "seed": 42})

    def test_vllm_payload_and_public_config_do_not_expose_api_key(self):
        client = LLMClient(
            provider="vllm",
            vllm_api_key="top-secret",
            temperature=0.0,
            seed=7,
        )
        session = _Session(
            _Response({"choices": [{"message": {"content": "answer"}}]})
        )

        with patch.object(client, "_session", return_value=session):
            result = client._call_chat(
                "model",
                [{"role": "user", "content": "question"}],
                {"Authorization": "Bearer top-secret"},
            )

        self.assertEqual(result, "answer")
        payload = session.calls[0][1]["json"]
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(payload["seed"], 7)
        self.assertNotIn("top-secret", str(client.generation_config()))

    def test_temperature_must_be_in_supported_range(self):
        with self.assertRaises(ValueError):
            LLMClient(temperature=2.1)


if __name__ == "__main__":
    unittest.main()
