"""Deterministic CTC/mapping regressions; no model download or training required."""

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "grainspeech"))

from preprocessing.ctc import (
    AlignmentError, character_error_rate, hanzi_pinyin, syllable_intervals, viterbi_spans,
)
from preprocess_baker import AlignedBakerPreprocessor
import tgt


class AlignmentTests(unittest.TestCase):
    def test_numbered_pinyin_and_explicit_erhua(self):
        chars, phones, groups = hanzi_pinyin(
            "\u95e8\u513f\uff0c\u513f\u7ae5\u3002", "menr2 er2 tong2",
        )
        self.assertEqual("".join(chars), "\u95e8\u513f\u513f\u7ae5")
        self.assertEqual(phones, ["menr2", "er2", "tong2"])
        self.assertEqual(groups, [(0, 2), (2, 3), (3, 4)])
        with self.assertRaisesRegex(AlignmentError, "Erhua"):
            hanzi_pinyin("\u95e8", "menr2")
        with self.assertRaisesRegex(AlignmentError, "non-Hanzi"):
            hanzi_pinyin("\uff30", "P IY1")
        with self.assertRaises(AlignmentError):
            hanzi_pinyin("\u4f60\u597d", "ni3")

    def test_repeated_labels_require_blank(self):
        emissions = np.log(np.array([
            [0.01, 0.99], [0.99, 0.01], [0.01, 0.99],
        ], dtype=np.float32))
        spans = viterbi_spans(emissions, [1, 1], 0)
        self.assertEqual([span[:2] for span in spans], [(0, 0), (2, 2)])
        self.assertGreater(min(span[2] for span in spans), 0.98)
        with self.assertRaisesRegex(AlignmentError, "Too few"):
            viterbi_spans(emissions[:2], [1, 1], 0)
        with self.assertRaisesRegex(AlignmentError, "blank token"):
            viterbi_spans(emissions, [0], 0)

    def test_complete_path_and_midpoint_policy(self):
        emissions = np.log(np.array([
            [0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.9, 0.05, 0.05],
            [0.9, 0.05, 0.05], [0.05, 0.05, 0.9], [0.9, 0.05, 0.05],
        ], dtype=np.float32))
        spans = viterbi_spans(emissions, [1, 2], 0)
        self.assertEqual([span[:2] for span in spans], [(1, 1), (4, 4)])
        intervals = syllable_intervals(
            spans, ["ni3", "hao3"], [(0, 1), (1, 2)], 0.02, 0.0125, 0.13,
        )
        self.assertAlmostEqual(intervals[0]["end"], 0.0625)
        self.assertEqual(intervals[0]["end"], intervals[1]["start"])
        self.assertEqual(intervals[0]["start"], 0)
        self.assertEqual(intervals[-1]["end"], 0.13)
        self.assertEqual(character_error_rate("abc", "adc"), 1 / 3)
        self.assertEqual(character_error_rate("abc", "abc"), 0)

    def test_mel_duration_accounting(self):
        extractor = object.__new__(AlignedBakerPreprocessor)
        extractor.sample_rate, extractor.hop_length = 22050, 256
        tier = tgt.IntervalTier()
        tier.add_interval(tgt.Interval(0, 0.2, "ni3"))
        tier.add_interval(tgt.Interval(0.2, 0.5, "hao3"))
        phones, durations, start, end = extractor._get_alignment(tier)
        self.assertEqual(phones, ["ni3", "hao3"])
        self.assertEqual(sum(durations), round((end - start) * 22050 / 256))
        bad = tgt.IntervalTier()
        bad.add_interval(tgt.Interval(0, 0.001, "ni3"))
        with self.assertRaisesRegex(AlignmentError, "shorter"):
            extractor._get_alignment(bad)


if __name__ == "__main__":
    unittest.main()
