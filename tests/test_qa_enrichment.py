import json
import os
import tempfile
import unittest

from qa_generator import (
    _enrich_after_dedup,
    _extract_balanced_json_array,
    _try_parse_json,
    clean_dataset,
)


class _EnrichingLLM:
    seed = 7

    def chat(self, **kwargs):
        return '{"input": "auth.log: failed login", "output": "Brute-force attempt"}'


class EnrichmentReturnTests(unittest.TestCase):
    def test_enrichment_returns_the_updated_dataset(self):
        qa_list = [
            {
                "instruction": "What is TLS?",
                "input": "",
                "output": "Transport Layer Security protects data in transit.",
            }
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = f"{temp_dir}/dataset_qa_enriched.json"
            result = _enrich_after_dedup(
                qa_list,
                ratio=1.0,
                model_name="test-model",
                llm_client=_EnrichingLLM(),
                output_path=output_path,
            )

        self.assertIs(result, qa_list)
        self.assertEqual(result[0]["input"], "auth.log: failed login")
        self.assertEqual(result[0]["output"], "Brute-force attempt")
        self.assertTrue(result[0]["_meta"]["enriched"])
        self.assertEqual(result[0]["_meta"]["domain"], "cyber")

    def test_finance_domain_uses_finance_prompt(self):
        captured = {}

        class _CaptureLLM:
            seed = 1

            def chat(self, **kwargs):
                captured["system"] = kwargs["messages"][0]["content"]
                return '{"input": "10-K excerpt", "output": "Credit risk note"}'

        qa_list = [
            {
                "instruction": "What is duration?",
                "input": "",
                "output": "Duration measures interest-rate sensitivity.",
            }
        ]
        result = _enrich_after_dedup(
            qa_list,
            ratio=1.0,
            model_name="test-model",
            llm_client=_CaptureLLM(),
            domain="finance",
        )
        self.assertIn("finance and markets", captured["system"])
        self.assertNotIn("MITRE", captured["system"])
        self.assertEqual(result[0]["_meta"]["domain"], "finance")

    def test_clean_dataset_writes_jsonl(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = os.path.join(temp_dir, "dataset_qa.json")
            with open(source, "w", encoding="utf-8") as output:
                json.dump(
                    [
                        {
                            "instruction": "What is TLS?",
                            "input": "",
                            "output": "Transport Layer Security.",
                            "_meta": {"source": {"document": "a.pdf"}},
                        },
                        {"instruction": "", "output": "drop me"},
                    ],
                    output,
                )
            cleaned = clean_dataset(source)
            self.assertTrue(cleaned.endswith("_cleaned.jsonl"))
            with open(cleaned, encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["instruction"], "What is TLS?")
            self.assertNotIn("_meta", rows[0])

    def test_balanced_array_stops_at_first_complete_list(self):
        content = 'prefix [{"question": "Q?", "answer": "A]"}] leftover ]'
        parsed = _try_parse_json(content)
        self.assertEqual(parsed, [{"question": "Q?", "answer": "A]"}])
        self.assertIsNotNone(_extract_balanced_json_array(content))


if __name__ == "__main__":
    unittest.main()
