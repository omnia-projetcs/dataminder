# Dataminder parser benchmarks

Benchmark manifests use one JSON object per line:

```json
{
  "id": "html-layout",
  "path": "fixtures/sample.html",
  "required_phrases": ["TLS 1.3", "one round trip"],
  "forbidden_phrases": ["navigation noise"],
  "min_chars": 80,
  "expected_pages": [],
  "expected_block_types": ["text"],
  "reference_text": "fixtures/sample.reference.txt"
}
```

Paths are resolved relative to the manifest. Only `id` and `path` are needed;
the score averages the expectations supplied by each case.

Run the native quality gate:

```bash
python benchmark.py \
  --corpus benchmarks/corpus.example.jsonl \
  --parsers native \
  --min-score 0.90 \
  --output benchmark-report.json
```

After installing `requirements-marker.txt`, compare backends:

```bash
python benchmark.py \
  --corpus /path/to/your/corpus.jsonl \
  --parsers native,marker,auto \
  --marker-mode fast \
  --output benchmark-report.json
```

Use representative private documents locally: digital and scanned PDFs, mixed
PDFs, multi-column pages, tables, equations, Office files, and the languages
used in production. Do not commit confidential documents or reference text.
