import json
import os
import tempfile
import unittest

from document_ir import (
    blocks_from_text,
    build_chunks,
    chunks_to_jsonl,
    create_document,
    sha256_file,
)


class DocumentIRTests(unittest.TestCase):
    def test_blocks_preserve_headings_tables_and_code(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = os.path.join(temp_dir, "guide.md")
            with open(source, "w", encoding="utf-8") as output:
                output.write("placeholder")
            document = create_document(source)
            document.blocks = blocks_from_text(
                "# Install\n\nUse the package.\n\n"
                "## Matrix\n\n| A | B |\n|---|---|\n"
                "\n```bash\nrun --safe\n```\n",
                document.id,
                extraction_method="markdown",
            )

            types = [block.block_type for block in document.blocks]
            self.assertEqual(
                types,
                ["section_header", "text", "section_header", "table", "code"],
            )
            self.assertEqual(document.blocks[-1].heading_path, ["Install", "Matrix"])
            self.assertEqual(len({block.id for block in document.blocks}), 5)

    def test_chunks_are_traceable_and_bounded(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = os.path.join(temp_dir, "long.md")
            content = "# Topic\n\n" + ("A sentence with context. " * 80)
            with open(source, "w", encoding="utf-8") as output:
                output.write(content)
            document = create_document(source)
            document.metadata["source_relative_path"] = "nested/long.md"
            document.blocks = blocks_from_text(
                content,
                document.id,
                extraction_method="markdown",
            )

            chunks = build_chunks(document, max_chars=300, overlap_chars=30)

            self.assertGreater(len(chunks), 1)
            self.assertTrue(all(chunk["char_count"] <= 300 for chunk in chunks))
            self.assertTrue(all(chunk["document_id"] == document.id for chunk in chunks))
            self.assertTrue(all(chunk["block_ids"] for chunk in chunks))
            self.assertTrue(all(chunk["source_path"] == "nested/long.md" for chunk in chunks))

            serialized = chunks_to_jsonl(chunks)
            restored = [json.loads(line) for line in serialized.splitlines()]
            self.assertEqual([item["id"] for item in restored], [item["id"] for item in chunks])

    def test_sha256_is_content_based(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            first = os.path.join(temp_dir, "first.txt")
            second = os.path.join(temp_dir, "second.txt")
            for path in (first, second):
                with open(path, "w", encoding="utf-8") as output:
                    output.write("same")
            self.assertEqual(sha256_file(first), sha256_file(second))


if __name__ == "__main__":
    unittest.main()
