"""Unit tests for the gateway's token truncation. Run: uv run python -m unittest discover tests"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("EMBEDDING_MODELS", "gemma")

from server import truncate_tokens  # noqa: E402

BOS, EOS = 2, 1


class TruncateTokens(unittest.TestCase):
    def test_fitting_input_is_unchanged(self):
        toks = [BOS, 10, 11, 12, EOS]
        self.assertEqual(truncate_tokens(toks, 5, [BOS], [EOS]), toks)

    def test_keeps_start_and_end_tokens_and_the_text_start(self):
        toks = [BOS] + list(range(100, 110)) + [EOS]
        self.assertEqual(truncate_tokens(toks, 5, [BOS], [EOS]), [BOS, 100, 101, 102, EOS])

    def test_suffix_only_model(self):  # qwen: no start token, pools the final EOS
        toks = list(range(100, 110)) + [151643]
        self.assertEqual(truncate_tokens(toks, 4, [], [151643]), [100, 101, 102, 151643])

    def test_result_never_exceeds_the_context(self):
        toks = [BOS] + list(range(100, 3100)) + [EOS]
        cut = truncate_tokens(toks, 2048, [BOS], [EOS])
        self.assertEqual(len(cut), 2048)
        self.assertEqual((cut[0], cut[-1]), (BOS, EOS))


if __name__ == "__main__":
    unittest.main()
