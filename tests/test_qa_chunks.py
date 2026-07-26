import json
import os
import tempfile
import unittest
from unittest.mock import patch

from qa_generator import generate_qa_dataset


class _NoopLLM:
    def unload_model(self, model):
        return None


class _ConfiguredLLM(_NoopLLM):
    def __init__(self, temperature):
        self.temperature = temperature

    def generation_config(self):
        return {
            "provider": "ollama",
            "endpoint": "http://localhost:11434",
            "temperature": self.temperature,
            "seed": 11,
        }


class QAChunkInputTests(unittest.TestCase):
    def test_chunk_provenance_reaches_final_dataset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = os.path.join(temp_dir, "export")
            dest_dir = os.path.join(temp_dir, "results")
            os.makedirs(os.path.join(source_dir, "nested"))
            chunk_path = os.path.join(
                source_dir, "nested", "manual.pdf.chunks.jsonl"
            )
            chunk = {
                "id": "sha256:chunk",
                "document_id": "sha256:document",
                "source_path": "nested/manual.pdf",
                "text": (
                    "TLS 1.3 removes obsolete cipher suites and reduces the "
                    "number of handshake round trips."
                ),
                "pages": [4],
                "block_ids": ["sha256:document/page/4/block/2"],
                "heading_path": ["Transport security"],
            }
            with open(chunk_path, "w", encoding="utf-8") as output:
                output.write(json.dumps(chunk) + "\n")

            generated = [
                {
                    "question": "What handshake improvement does TLS 1.3 provide?",
                    "answer": "TLS 1.3 reduces the number of handshake round trips.",
                }
            ]
            updated = [
                {
                    "question": "What handshake improvement does TLS 1.3 provide?",
                    "answer": "TLS 1.3 completes its standard handshake in one round trip.",
                }
            ]
            with patch(
                "qa_generator.generate_qa_from_text",
                side_effect=[generated, updated],
            ) as generate_mock:
                generate_qa_dataset(
                    source_dir,
                    dest_dir,
                    "unused",
                    llm_client=_NoopLLM(),
                    input_format="chunks",
                )

                # An unchanged second pass establishes the file sentinel without
                # calling the LLM again.
                generate_qa_dataset(
                    source_dir,
                    dest_dir,
                    "unused",
                    llm_client=_NoopLLM(),
                    input_format="chunks",
                )
                self.assertEqual(generate_mock.call_count, 1)

                chunk["id"] = "sha256:changed-chunk"
                chunk["text"] += " The standard handshake takes one round trip."
                with open(chunk_path, "w", encoding="utf-8") as output:
                    output.write(json.dumps(chunk) + "\n")
                generate_qa_dataset(
                    source_dir,
                    dest_dir,
                    "unused",
                    llm_client=_NoopLLM(),
                    input_format="chunks",
                )
                self.assertEqual(generate_mock.call_count, 2)

            with open(
                os.path.join(dest_dir, "dataset_qa.json"),
                "r",
                encoding="utf-8",
            ) as source:
                dataset = json.load(source)

            self.assertEqual(len(dataset), 1)
            provenance = dataset[0]["_meta"]["source"]
            self.assertEqual(provenance["chunk_id"], "sha256:changed-chunk")
            self.assertEqual(provenance["document_id"], "sha256:document")
            self.assertEqual(provenance["pages"], [4])
            self.assertEqual(provenance["document"], "nested/manual.pdf")
            self.assertEqual(
                dataset[0]["output"],
                "TLS 1.3 completes its standard handshake in one round trip.",
            )
            self.assertTrue(
                os.path.exists(
                    os.path.join(dest_dir, "dataset_qa_raw_chunks.jsonl")
                )
            )

    def test_generation_configuration_invalidates_qa_resume_state(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = os.path.join(temp_dir, "summaries")
            dest_dir = os.path.join(temp_dir, "results")
            os.makedirs(source_dir)
            with open(
                os.path.join(source_dir, "network.md"),
                "w",
                encoding="utf-8",
            ) as output:
                output.write(
                    "A TCP connection begins with a three-way handshake. "
                    "The peers exchange SYN, SYN-ACK, and ACK segments."
                )

            generated = [
                {
                    "question": "How does a TCP connection begin?",
                    "answer": "It begins with a three-way handshake.",
                }
            ]
            regenerated = [
                {
                    "question": "Which segments establish a TCP connection?",
                    "answer": "SYN, SYN-ACK, and ACK establish the connection.",
                }
            ]
            with patch(
                "qa_generator.generate_qa_from_text",
                side_effect=[generated, regenerated],
            ) as generate_mock:
                generate_qa_dataset(
                    source_dir,
                    dest_dir,
                    "model",
                    llm_client=_ConfiguredLLM(0.0),
                    input_format="summaries",
                )
                generate_qa_dataset(
                    source_dir,
                    dest_dir,
                    "model",
                    llm_client=_ConfiguredLLM(0.5),
                    input_format="summaries",
                )

            self.assertEqual(generate_mock.call_count, 2)
            with open(
                os.path.join(dest_dir, "dataset_qa.json"),
                "r",
                encoding="utf-8",
            ) as source:
                dataset = json.load(source)
            self.assertEqual(len(dataset), 1)
            self.assertEqual(
                dataset[0]["output"],
                "SYN, SYN-ACK, and ACK establish the connection.",
            )


if __name__ == "__main__":
    unittest.main()
