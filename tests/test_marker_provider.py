import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from document_ir import create_document
from document_parser import extract_document
from marker_provider import document_from_marker_output


class MarkerProviderTests(unittest.TestCase):
    def test_marker_chunks_map_to_document_ir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = os.path.join(temp_dir, "paper.pdf")
            with open(source, "wb") as output:
                output.write(b"%PDF-test")

            heading_id = "/page/0/SectionHeader/0"
            rendered = SimpleNamespace(
                blocks=[
                    SimpleNamespace(
                        id=heading_id,
                        block_type="SectionHeader",
                        html="<h2>Methods</h2>",
                        page=0,
                        bbox=[0, 10, 100, 30],
                        section_hierarchy={2: heading_id},
                    ),
                    SimpleNamespace(
                        id="/page/0/Text/1",
                        block_type="Text",
                        html="<p>Measured result.</p>",
                        page=0,
                        bbox=[0, 40, 100, 60],
                        section_hierarchy={2: heading_id},
                    ),
                ],
                metadata={"page_stats": [{"page_id": 0}]},
            )

            document = document_from_marker_output(
                source,
                rendered,
                source_sha256="abc",
                mode="fast",
            )

            self.assertEqual(document.metadata["parser"], "marker")
            self.assertEqual(document.blocks[0].block_type, "section_header")
            self.assertEqual(document.blocks[1].heading_path, ["Methods"])
            self.assertEqual(document.blocks[1].page, 1)
            self.assertEqual(document.blocks[1].bbox, [0.0, 40.0, 100.0, 60.0])
            self.assertEqual(document.blocks[1].extraction_method, "marker-fast")

    def test_auto_falls_back_to_native_with_diagnostic(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = os.path.join(temp_dir, "paper.pdf")
            with open(source, "wb") as output:
                output.write(b"%PDF-test")
            native_document = create_document(source, source_sha256="abc")

            with (
                patch("document_parser.is_marker_available", return_value=True),
                patch(
                    "document_parser.extract_document_with_marker",
                    side_effect=RuntimeError("inference unavailable"),
                ),
                patch(
                    "document_parser.extract_document_native",
                    return_value=native_document,
                ),
            ):
                result = extract_document(
                    source,
                    parser="auto",
                    source_sha256="abc",
                )

            self.assertEqual(result.metadata["parser"], "native-fallback")
            self.assertEqual(result.diagnostics[0]["code"], "marker_fallback")

    def test_marker_is_rejected_for_audio(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = os.path.join(temp_dir, "clip.mp3")
            with open(source, "wb") as output:
                output.write(b"audio")
            with self.assertRaises(ValueError):
                extract_document(source, parser="marker", source_sha256="abc")


if __name__ == "__main__":
    unittest.main()
