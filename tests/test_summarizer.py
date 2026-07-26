import unittest

from summarizer import SummarizationError, summarize_text


class _FailingLLM:
    def chat(self, **kwargs):
        raise TimeoutError("offline")

    def __repr__(self):
        return "FailingLLM"


class SummarizerTests(unittest.TestCase):
    def test_failures_are_not_returned_as_markdown(self):
        with self.assertRaises(SummarizationError):
            summarize_text(
                "Useful source content",
                model_name="test-model",
                llm_client=_FailingLLM(),
            )


if __name__ == "__main__":
    unittest.main()
