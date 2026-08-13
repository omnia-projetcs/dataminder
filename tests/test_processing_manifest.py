import os
import tempfile
import unittest

from processing_manifest import (
    ProcessingManifest,
    atomic_text_writer,
    atomic_write_text,
    output_paths,
    relative_source_id,
    stable_fingerprint,
)


class ProcessingManifestTests(unittest.TestCase):
    def test_nested_same_name_sources_do_not_collide(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = os.path.join(temp_dir, "source")
            dest_dir = os.path.join(temp_dir, "export")
            first = os.path.join(source_dir, "client-a", "report.pdf")
            second = os.path.join(source_dir, "client-b", "report.docx")

            first_outputs = output_paths(first, source_dir, dest_dir)
            second_outputs = output_paths(second, source_dir, dest_dir)

            self.assertNotEqual(first_outputs["markdown"], second_outputs["markdown"])
            self.assertTrue(first_outputs["markdown"].endswith("client-a/report.pdf.md"))
            self.assertTrue(second_outputs["markdown"].endswith("client-b/report.docx.md"))
            self.assertEqual(relative_source_id(first, source_dir), "client-a/report.pdf")

            same_dir_pdf = os.path.join(source_dir, "report.pdf")
            same_dir_docx = os.path.join(source_dir, "report.docx")
            self.assertNotEqual(
                output_paths(same_dir_pdf, source_dir, dest_dir)["markdown"],
                output_paths(same_dir_docx, source_dir, dest_dir)["markdown"],
            )

    def test_manifest_requires_hash_config_and_all_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = os.path.join(temp_dir, "source")
            dest_dir = os.path.join(temp_dir, "export")
            source = os.path.join(source_dir, "manual.txt")
            os.makedirs(source_dir)
            with open(source, "w", encoding="utf-8") as output:
                output.write("manual")

            outputs = output_paths(source, source_dir, dest_dir)
            for path in outputs.values():
                atomic_write_text(path, "result")

            manifest = ProcessingManifest(dest_dir)
            fingerprint = stable_fingerprint({"model": "local", "level": 0})
            manifest.mark_success(
                "manual.txt",
                source_sha256="abc",
                pipeline_fingerprint=fingerprint,
                outputs=outputs,
                document_id="sha256:abc",
            )

            reloaded = ProcessingManifest(dest_dir)
            self.assertTrue(
                reloaded.is_current(
                    "manual.txt",
                    source_sha256="abc",
                    pipeline_fingerprint=fingerprint,
                    outputs=outputs,
                )
            )
            self.assertFalse(
                reloaded.is_current(
                    "manual.txt",
                    source_sha256="changed",
                    pipeline_fingerprint=fingerprint,
                    outputs=outputs,
                )
            )
            os.unlink(outputs["chunks"])
            self.assertFalse(
                reloaded.is_current(
                    "manual.txt",
                    source_sha256="abc",
                    pipeline_fingerprint=fingerprint,
                    outputs=outputs,
                )
            )

    def test_manifest_rejects_empty_output_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dest_dir = os.path.join(temp_dir, "export")
            os.makedirs(dest_dir)
            outputs = {
                "markdown": os.path.join(dest_dir, "doc.md"),
                "chunks": os.path.join(dest_dir, "doc.chunks.jsonl"),
                "document": os.path.join(dest_dir, "doc.document.json"),
            }
            for path in outputs.values():
                atomic_write_text(path, "ok")
            empty_path = outputs["markdown"]
            with open(empty_path, "w", encoding="utf-8"):
                pass

            manifest = ProcessingManifest(dest_dir)
            fingerprint = stable_fingerprint({"model": "local"})
            manifest.mark_success(
                "doc.txt",
                source_sha256="abc",
                pipeline_fingerprint=fingerprint,
                outputs=outputs,
                document_id="sha256:abc",
            )
            self.assertFalse(
                manifest.is_current(
                    "doc.txt",
                    source_sha256="abc",
                    pipeline_fingerprint=fingerprint,
                    outputs=outputs,
                )
            )

    def test_atomic_text_writer_discards_partial_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "out.jsonl")
            atomic_write_text(path, "keep\n")
            with self.assertRaises(RuntimeError):
                with atomic_text_writer(path) as handle:
                    handle.write("partial\n")
                    raise RuntimeError("boom")
            with open(path, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "keep\n")


if __name__ == "__main__":
    unittest.main()
