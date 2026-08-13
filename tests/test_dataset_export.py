import json
import os
import tempfile
import unittest

from dataset_export import clean_dataset, format_hf_text, prepare_hf_dataset
from extractor import configure_ocr, extraction_config


class DatasetExportTests(unittest.TestCase):
    def test_prepare_hf_accepts_json_array_and_jsonl(self):
        records = [
            {
                "instruction": "What is TLS?",
                "input": "",
                "output": "Transport Layer Security.",
            },
            {
                "instruction": "Name the handshake",
                "input": "TCP",
                "output": "Three-way handshake.",
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            array_path = os.path.join(temp_dir, "dataset_qa_cleaned.json")
            jsonl_path = os.path.join(temp_dir, "dataset_qa_cleaned.jsonl")
            with open(array_path, "w", encoding="utf-8") as output:
                json.dump(records, output)
            with open(jsonl_path, "w", encoding="utf-8") as output:
                for record in records:
                    output.write(json.dumps(record) + "\n")

            array_export = prepare_hf_dataset(array_path)
            jsonl_export = prepare_hf_dataset(jsonl_path)
            self.assertTrue(array_export.endswith("_hf.jsonl"))
            self.assertTrue(jsonl_export.endswith("_hf.jsonl"))

            with open(array_export, encoding="utf-8") as handle:
                array_rows = [json.loads(line) for line in handle if line.strip()]
            with open(jsonl_export, encoding="utf-8") as handle:
                jsonl_rows = [json.loads(line) for line in handle if line.strip()]

            self.assertEqual(len(array_rows), 2)
            self.assertEqual(array_rows, jsonl_rows)
            self.assertIn("### Instruction:\nWhat is TLS?", array_rows[0]["text"])
            self.assertIn("### Input:\nTCP", array_rows[1]["text"])

    def test_clean_dataset_is_reexported_from_qa_generator(self):
        from qa_generator import clean_dataset as reexported

        self.assertIs(reexported, clean_dataset)

    def test_hf_text_omits_empty_input_section(self):
        self.assertEqual(
            format_hf_text("Q", "", "A"),
            "### Instruction:\nQ\n\n### Response:\nA",
        )

    def test_iter_alpaca_records_skips_utf8_bom(self):
        from dataset_export import iter_alpaca_records

        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "bom.json")
            payload = [
                {
                    "instruction": "What is TLS?",
                    "input": "",
                    "output": "Transport Layer Security.",
                }
            ]
            with open(path, "w", encoding="utf-8-sig") as output:
                json.dump(payload, output)
            rows = list(iter_alpaca_records(path))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["instruction"], "What is TLS?")

    def test_clean_dataset_accepts_jsonl_input(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = os.path.join(temp_dir, "pairs.jsonl")
            with open(source, "w", encoding="utf-8") as output:
                output.write(
                    json.dumps(
                        {
                            "instruction": "What is TLS?",
                            "input": "",
                            "output": "Transport Layer Security.",
                        }
                    )
                    + "\n"
                )
                output.write(json.dumps({"instruction": "", "output": "drop"}) + "\n")
            cleaned = clean_dataset(source)
            with open(cleaned, encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["instruction"], "What is TLS?")


class ExtractionSettingsTests(unittest.TestCase):
    def tearDown(self):
        configure_ocr()

    def test_extraction_config_tracks_ocr_settings(self):
        configure_ocr(engine="paddleocr", device="gpu", lang="fr", dpi=150, max_pages=3)
        config = extraction_config()
        self.assertEqual(config["ocr_engine"], "paddleocr")
        self.assertEqual(config["ocr_device"], "gpu")
        self.assertEqual(config["ocr_lang"], "fr")
        self.assertEqual(config["ocr_dpi"], 150)
        self.assertEqual(config["ocr_max_pages"], 3)


if __name__ == "__main__":
    unittest.main()
