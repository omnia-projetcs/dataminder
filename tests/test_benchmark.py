import json
import os
import tempfile
import unittest

from benchmark import benchmark_corpus, load_corpus, main


class BenchmarkTests(unittest.TestCase):
    def _write_corpus(self, temp_dir):
        document_path = os.path.join(temp_dir, "source.txt")
        reference_path = os.path.join(temp_dir, "reference.txt")
        manifest_path = os.path.join(temp_dir, "corpus.jsonl")
        text = "Deterministic extraction keeps provenance and stable identifiers."
        with open(document_path, "w", encoding="utf-8") as output:
            output.write(text)
        with open(reference_path, "w", encoding="utf-8") as output:
            output.write(text)
        case = {
            "id": "plain-text",
            "path": "source.txt",
            "required_phrases": ["stable identifiers"],
            "forbidden_phrases": ["missing phrase"],
            "min_chars": 40,
            "expected_block_types": ["text"],
            "reference_text": "reference.txt",
        }
        with open(manifest_path, "w", encoding="utf-8") as output:
            output.write(json.dumps(case) + "\n")
        return manifest_path

    def test_native_benchmark_scores_expected_text(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cases = load_corpus(self._write_corpus(temp_dir))
            report = benchmark_corpus(cases, ["native"])

            self.assertEqual(report["results"][0]["status"], "success")
            self.assertEqual(report["results"][0]["score"], 1.0)
            self.assertEqual(report["summary"]["native"]["mean_score"], 1.0)

    def test_cli_quality_gate_writes_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = self._write_corpus(temp_dir)
            report_path = os.path.join(temp_dir, "report.json")

            exit_code = main(
                [
                    "--corpus",
                    manifest,
                    "--parsers",
                    "native",
                    "--min-score",
                    "1.0",
                    "--output",
                    report_path,
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue(os.path.exists(report_path))


if __name__ == "__main__":
    unittest.main()
