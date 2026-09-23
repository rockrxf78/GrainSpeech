"""Small synthetic regressions for the original Baker Parquet importer."""

import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import yaml

from prepare_baker import ROOT, decode_record, prepare


class BakerImportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="grainspeech-baker-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.corpus = self.root / "source"
        self.corpus.mkdir()
        self.output = self.root / "prepared"
        with (ROOT / "configs/LJSpeech/preprocess.yaml").open() as stream:
            self.config = yaml.safe_load(stream)
        self.paths = []
        for index, rate in enumerate((16000, 48000)):
            buffer = io.BytesIO()
            time = np.arange(rate // 5) / rate
            waveform = 0.4 * np.sin(2 * np.pi * 200 * time)
            sf.write(buffer, waveform, rate, format="WAV", subtype="PCM_16")
            row = {
                "file_name": index + 1,
                "text": "\u4f60\u597d#2\u3002",
                "text_pinyin": "ni3 hao3",
                "audio": {"bytes": buffer.getvalue(), "path": "unused.wav"},
            }
            path = self.corpus / f"train-{index:05d}-of-00002.parquet"
            pq.write_table(pa.Table.from_pylist([row]), path)
            self.paths.append(path)

    def test_full_import_preserves_annotations_and_source(self):
        hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in self.paths]
        summary = prepare(self.corpus, self.output, self.config, mel_samples=1)
        self.assertEqual(summary["prepared_rows"], 2)
        self.assertFalse(summary["is_subset"])
        self.assertFalse(summary["training_ready"])
        self.assertEqual(summary["source_sampling_rates"], [16000, 48000])
        records = [
            json.loads(line)
            for line in (self.output / "manifest.jsonl").read_text().splitlines()
        ]
        self.assertEqual(len({record["id"] for record in records}), 2)
        for index, record in enumerate(records):
            self.assertEqual(record["source_file_name"], index + 1)
            self.assertEqual(record["source_audio_path"], "unused.wav")
            self.assertEqual(record["text"], "\u4f60\u597d\u3002")
            self.assertEqual(record["text_original"], "\u4f60\u597d#2\u3002")
            self.assertEqual(record["text_pinyin"], "ni3 hao3")
            waveform, rate = sf.read(self.output / record["wav"])
            self.assertEqual(rate, 22050)
            self.assertEqual(len(waveform), record["samples"])
            self.assertLessEqual(float(np.max(np.abs(waveform))), 0.951)
            self.assertGreater(float(np.max(np.abs(waveform))), 0.94)
        mel = np.load(self.output / records[0]["mel"])
        self.assertEqual(mel.shape[1], 80)
        self.assertTrue(np.isfinite(mel).all())
        self.assertNotIn("mel", records[1])
        self.assertEqual(
            hashes, [hashlib.sha256(path.read_bytes()).hexdigest() for path in self.paths],
        )
        with self.assertRaisesRegex(FileExistsError, "not empty"):
            prepare(self.corpus, self.output, self.config)

    def test_subset_and_source_overlap(self):
        summary = prepare(self.corpus, self.output, self.config, limit=1)
        self.assertTrue(summary["is_subset"])
        self.assertEqual(summary["prepared_rows"], 1)
        self.assertEqual(summary["source_rows"], 2)
        with self.assertRaisesRegex(ValueError, "overlap"):
            prepare(self.corpus, self.corpus / "output", self.config)

    def test_invalid_inputs_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            prepare(self.corpus, self.output, self.config, limit=0)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            prepare(self.corpus, self.output, self.config, mel_samples=-1)
        with self.assertRaisesRegex(ValueError, "text_pinyin"):
            decode_record({"text": "text", "text_pinyin": ""}, "bad-row", 22050)
        with self.assertRaisesRegex(ValueError, "audio bytes"):
            decode_record(
                {"text": "text", "text_pinyin": "ni3", "audio": {"bytes": None}},
                "bad-row", 22050,
            )
        for path in self.paths:
            pq.write_table(pa.table({"text": ["missing columns"]}), path)
        with self.assertRaisesRegex(ValueError, "missing columns"):
            prepare(self.corpus, self.output, self.config)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
