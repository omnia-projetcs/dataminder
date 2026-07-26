import json
import os
import tempfile
import unittest

from main import process_documents


class _NoopLLM:
    provider = "ollama"
    ollama_url = "http://127.0.0.1:11434"

    def unload_model(self, model):
        return None


class PipelineIntegrationTests(unittest.TestCase):
    def test_level_zero_writes_structured_outputs_and_reprocesses_changes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = os.path.join(temp_dir, "source")
            dest_dir = os.path.join(temp_dir, "export")
            first_dir = os.path.join(source_dir, "one")
            second_dir = os.path.join(source_dir, "two")
            os.makedirs(first_dir)
            os.makedirs(second_dir)
            first = os.path.join(first_dir, "report.txt")
            second = os.path.join(second_dir, "report.txt")
            with open(first, "w", encoding="utf-8") as output:
                output.write("# First\n\nAlpha")
            with open(second, "w", encoding="utf-8") as output:
                output.write("# Second\n\nBeta")

            first_report = process_documents(
                source_dir,
                dest_dir,
                model_name="unused",
                level=0,
                llm_client=_NoopLLM(),
            )
            self.assertEqual(first_report["summary"]["success"], 2)
            self.assertEqual(first_report["summary"]["skipped"], 0)
            report_path = os.path.join(
                dest_dir, ".dataminder-last-run.json"
            )
            self.assertTrue(os.path.exists(report_path))

            first_markdown = os.path.join(dest_dir, "one", "report.txt.md")
            second_markdown = os.path.join(dest_dir, "two", "report.txt.md")
            self.assertTrue(os.path.exists(first_markdown))
            self.assertTrue(os.path.exists(second_markdown))
            self.assertTrue(os.path.exists(os.path.join(dest_dir, "one", "report.txt.chunks.jsonl")))
            self.assertTrue(os.path.exists(os.path.join(dest_dir, "one", "report.txt.document.json")))

            manifest_path = os.path.join(dest_dir, ".dataminder-manifest.json")
            with open(manifest_path, "r", encoding="utf-8") as source:
                before = json.load(source)
            before_hash = before["documents"]["one/report.txt"]["source_sha256"]

            with open(first, "w", encoding="utf-8") as output:
                output.write("# First\n\nAlpha changed")
            second_report = process_documents(
                source_dir,
                dest_dir,
                model_name="unused",
                level=0,
                llm_client=_NoopLLM(),
            )
            self.assertEqual(second_report["summary"]["success"], 1)
            self.assertEqual(second_report["summary"]["skipped"], 1)
            self.assertEqual(second_report["summary"]["failed"], 0)

            with open(manifest_path, "r", encoding="utf-8") as source:
                after = json.load(source)
            after_hash = after["documents"]["one/report.txt"]["source_sha256"]
            self.assertNotEqual(before_hash, after_hash)
            with open(first_markdown, "r", encoding="utf-8") as source:
                self.assertIn("Alpha changed", source.read())
            with open(report_path, "r", encoding="utf-8") as source:
                stored_report = json.load(source)
            self.assertEqual(stored_report["summary"], second_report["summary"])


if __name__ == "__main__":
    unittest.main()
