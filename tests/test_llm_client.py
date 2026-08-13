import unittest
from unittest.mock import patch

import requests

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

    def test_client_http_errors_are_not_retried(self):
        client = LLMClient(provider="ollama")
        response = requests.Response()
        response.status_code = 400
        error = requests.HTTPError("bad request")
        error.response = response
        with patch.object(client, "_chat_ollama", side_effect=error) as chat_mock:
            with patch("llm_client.time.sleep") as sleep_mock:
                with self.assertRaises(requests.HTTPError):
                    client.chat("model", [{"role": "user", "content": "question"}])
        self.assertEqual(chat_mock.call_count, 1)
        sleep_mock.assert_not_called()

    def test_vllm_uses_requested_model_not_autodetected(self):
        client = LLMClient(provider="vllm")
        client._vllm_endpoint = "chat"
        client._vllm_model = "server-default"
        session = _Session(
            _Response({"choices": [{"message": {"content": "ok"}}]})
        )
        with patch.object(client, "_session", return_value=session):
            result = client._chat_vllm(
                "user-model",
                [{"role": "user", "content": "question"}],
            )
        self.assertEqual(result, "ok")
        self.assertEqual(session.calls[0][1]["json"]["model"], "user-model")

    def test_close_invalidates_pooled_sessions(self):
        client = LLMClient(provider="ollama")
        first = client._session()
        second = client._session()
        self.assertIs(first, second)
        client.close()
        third = client._session()
        self.assertIsNot(first, third)
        self.assertEqual(len(client._sessions), 1)

    def test_empty_ollama_response_is_retried_then_fails(self):
        client = LLMClient(provider="ollama")
        session = _Session(_Response({"message": {"content": "   "}}))
        with patch.object(client, "_session", return_value=session):
            with patch("llm_client.time.sleep"):
                with self.assertRaises(RuntimeError):
                    client.chat("model", [{"role": "user", "content": "question"}])
        self.assertEqual(len(session.calls), 3)

    def test_vllm_missing_choices_is_an_error(self):
        client = LLMClient(provider="vllm")
        client._vllm_endpoint = "chat"
        session = _Session(_Response({"choices": []}))
        with patch.object(client, "_session", return_value=session):
            with self.assertRaises(RuntimeError):
                client._chat_vllm("user-model", [{"role": "user", "content": "q"}])


if __name__ == "__main__":
    unittest.main()
