import unittest

from summarizer import SummarizationError, summarize_text


class _FailingLLM:
    def chat(self, **kwargs):
        raise TimeoutError("offline")

    def __repr__(self):
        return "FailingLLM"


class _EmptyLLM:
    def chat(self, **kwargs):
        return "Okay, here is a breakdown.\n\n"

    def __repr__(self):
        return "EmptyLLM"


class SummarizerTests(unittest.TestCase):
    def test_failures_are_not_returned_as_markdown(self):
        with self.assertRaises(SummarizationError):
            summarize_text(
                "Useful source content",
                model_name="test-model",
                llm_client=_FailingLLM(),
            )

    def test_empty_cleaned_output_is_an_error(self):
        with self.assertRaises(SummarizationError):
            summarize_text(
                "Useful source content",
                model_name="test-model",
                llm_client=_EmptyLLM(),
            )

    def test_empty_input_is_an_error(self):
        with self.assertRaises(SummarizationError):
            summarize_text("   ")


if __name__ == "__main__":
    unittest.main()
