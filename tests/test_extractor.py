import os
import tempfile
import unittest

from extractor import extract_text_from_html, extract_text_from_txt


class ExtractorTextTests(unittest.TestCase):
    def test_reads_utf8_and_legacy_western_encodings(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            utf8_path = os.path.join(temp_dir, "utf8.txt")
            latin_path = os.path.join(temp_dir, "latin.txt")
            with open(utf8_path, "w", encoding="utf-8") as output:
                output.write("clé publique")
            with open(latin_path, "wb") as output:
                output.write("café résumé".encode("cp1252"))

            self.assertIn("clé", extract_text_from_txt(utf8_path))
            self.assertIn("café", extract_text_from_txt(latin_path))
            self.assertIn("résumé", extract_text_from_txt(latin_path))

    def test_html_drops_scripts_and_styles(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "page.html")
            with open(path, "w", encoding="utf-8") as output:
                output.write(
                    "<html><head><style>body{color:red}</style>"
                    "<script>alert('x')</script></head>"
                    "<body><h1>Visible</h1><p>Body text</p></body></html>"
                )
            text = extract_text_from_html(path)
            self.assertIn("Visible", text)
            self.assertIn("Body text", text)
            self.assertNotIn("alert", text)
            self.assertNotIn("color:red", text)


if __name__ == "__main__":
    unittest.main()
