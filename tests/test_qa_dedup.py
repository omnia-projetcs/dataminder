import json
import os
import subprocess
import sys
import unittest

from qa_generator import _get_ngrams, _lsh_buckets, _minhash_signature


class QADeduplicationTests(unittest.TestCase):
    def test_minhash_is_independent_from_python_hash_seed(self):
        code = (
            "import json;"
            "from qa_generator import _get_ngrams,_minhash_signature,_lsh_buckets;"
            "s=_minhash_signature(_get_ngrams('deterministic question'),8);"
            "print(json.dumps([s,_lsh_buckets(s,2)]))"
        )
        outputs = []
        for seed in ("1", "98765"):
            environment = dict(os.environ)
            environment["PYTHONHASHSEED"] = seed
            completed = subprocess.run(
                [sys.executable, "-c", code],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )
            outputs.append(json.loads(completed.stdout))
        self.assertEqual(outputs[0], outputs[1])

    def test_band_hashes_are_stable_in_process(self):
        signature = _minhash_signature(_get_ngrams("stable source"), 16)
        self.assertEqual(
            _lsh_buckets(signature, 4),
            _lsh_buckets(signature, 4),
        )


if __name__ == "__main__":
    unittest.main()
