import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from main import build_model_data_products, default_finance_source


class ModelDataPipelineTests(unittest.TestCase):
    def test_builds_separate_grounded_products_without_an_llm(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cyber_source = root / "sources" / "cyber"
            finance_source = root / "sources" / "finance"
            (cyber_source / "PENTEST").mkdir(parents=True)
            finance_source.mkdir(parents=True)

            cyber_text = (
                "# Incident response\n\n"
                + (
                    "Incident responders preserve evidence, record timestamps, "
                    "validate indicators, contain affected systems, and document "
                    "every remediation decision before restoring service. "
                )
                * 45
            )
            finance_text = (
                "# Portfolio risk\n\n"
                + (
                    "Portfolio risk analysis compares expected return, volatility, "
                    "correlation, liquidity, and concentration before a documented "
                    "allocation decision is approved. "
                )
                * 45
            )
            (cyber_source / "PENTEST" / "incident-guide.md").write_text(
                cyber_text,
                encoding="utf-8",
            )
            (finance_source / "portfolio-guide.md").write_text(
                finance_text,
                encoding="utf-8",
            )

            rag_dir = root / "rag"
            training_dir = root / "training"
            colab_dir = root / "colab"
            report = build_model_data_products(
                cyber_source=cyber_source,
                finance_source=finance_source,
                rag_dir=rag_dir,
                training_dir=training_dir,
                colab_dir=colab_dir,
                chunk_max_chars=900,
                chunk_overlap_chars=90,
                max_chunks_per_document=20,
            )

            self.assertEqual(
                set(report["domains"]),
                {"cyber", "finance"},
            )
            self.assertEqual(
                report["policies"]["generated_summaries"],
                "excluded",
            )
            self.assertEqual(report["policies"]["generated_qa"], "excluded")

            for domain in ("cyber", "finance"):
                database = rag_dir / f"{domain}_rag.sqlite"
                training = training_dir / f"{domain}_model_enrichment.jsonl"
                colab = colab_dir / f"{domain}_colab_messages.jsonl"
                self.assertTrue(database.is_file())
                self.assertTrue(training.is_file())
                self.assertTrue(colab.is_file())

                connection = sqlite3.connect(database)
                try:
                    self.assertEqual(
                        connection.execute(
                            "PRAGMA integrity_check"
                        ).fetchone()[0],
                        "ok",
                    )
                    self.assertGreater(
                        connection.execute(
                            "SELECT count(*) FROM chunks"
                        ).fetchone()[0],
                        0,
                    )
                    self.assertEqual(
                        connection.execute(
                            "SELECT count(DISTINCT domain) FROM documents"
                        ).fetchone()[0],
                        1,
                    )
                    self.assertEqual(
                        connection.execute(
                            "SELECT min(domain) FROM documents"
                        ).fetchone()[0],
                        domain,
                    )
                finally:
                    connection.close()

                training_rows = [
                    json.loads(line)
                    for line in training.read_text(encoding="utf-8").splitlines()
                ]
                colab_rows = [
                    json.loads(line)
                    for line in colab.read_text(encoding="utf-8").splitlines()
                ]
                self.assertGreater(len(training_rows), 0)
                self.assertEqual(len(colab_rows), len(training_rows))
                self.assertTrue(
                    all(row["domain"] == domain for row in training_rows)
                )
                self.assertTrue(
                    all(
                        [message["role"] for message in row["messages"]]
                        == ["user", "assistant"]
                        for row in colab_rows
                    )
                )
                self.assertTrue(
                    all(
                        row["metadata"]["domain"] == domain
                        for row in colab_rows
                    )
                )


    def test_chunk_windows_prefers_paragraph_breaks(self):
        from scripts.build_rag_databases import TextUnit, chunk_windows

        paragraph = "Alpha sentence. " * 20
        text = paragraph.strip() + "\n\n" + ("Beta sentence. " * 20)
        parts = chunk_windows(
            TextUnit(text=text, method="test"),
            max_chars=len(paragraph) + 40,
            overlap_chars=10,
        )
        self.assertGreater(len(parts), 1)
        self.assertTrue(parts[0].text.startswith("Alpha"))
        self.assertNotIn("Beta sentence", parts[0].text)

    def test_gold_jsonl_skips_invalid_lines(self):
        from scripts.build_gold_datasets import iter_jsonl_objects
        import collections

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "raw.jsonl"
            path.write_text(
                "{not json\n"
                + json.dumps({"question": "ok", "answer": "yes"})
                + "\n",
                encoding="utf-8",
            )
            rejected = collections.Counter()
            rows = list(iter_jsonl_objects(path, rejected))
            self.assertEqual(rejected["invalid_json"], 1)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0][1]["question"], "ok")

    def test_plain_extract_reads_legacy_encoding(self):
        from scripts.build_rag_databases import extract_plain

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "note.txt"
            path.write_bytes("café résumé".encode("cp1252"))
            document = extract_plain(path)
            self.assertIn("café", document.units[0].text)

    def test_export_database_requires_existing_file(self):
        from scripts.export_model_enrichment import export_database

        with tempfile.TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / "missing.sqlite"
            with self.assertRaises(FileNotFoundError):
                export_database(
                    database=missing,
                    output=Path(temp_dir) / "out.jsonl",
                    expected_domain="cyber",
                    min_quality=0.8,
                    min_chars=400,
                    min_words=50,
                    max_chunks_per_document=10,
                )

    def test_colab_export_reports_invalid_json_line(self):
        from scripts.export_colab_messages import convert_file

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "cyber_model_enrichment.jsonl"
            source.write_text("{bad\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, ":1: invalid JSON"):
                convert_file(
                    source=source,
                    output=root / "out.jsonl",
                    expected_domain="cyber",
                )

    def test_colab_export_requires_source_file(self):
        from scripts.export_colab_messages import convert_file

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaises(FileNotFoundError):
                convert_file(
                    source=root / "missing.jsonl",
                    output=root / "out.jsonl",
                    expected_domain="cyber",
                )

    def test_finance_source_falls_back_to_local_corpus(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            local = root / "data" / "datas-finance"
            local.mkdir(parents=True)
            self.assertEqual(default_finance_source(root), str(local))


if __name__ == "__main__":
    unittest.main()
