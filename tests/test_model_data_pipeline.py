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


    def test_finance_source_falls_back_to_local_corpus(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            local = root / "data" / "datas-finance"
            local.mkdir(parents=True)
            self.assertEqual(default_finance_source(root), str(local))


if __name__ == "__main__":
    unittest.main()
