"""Compare single-graph A16W8 and FP32 exports, including frame-index regressions.

Requires requirements-export.txt and both exported TFLite files. Saves the two
README sentences' predicted Mels and comparison.json to --output-dir.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "grainspeech"))
from text import text_to_sequence


class ExportRunner:
    def __init__(self, path):
        self.interpreter = tf.lite.Interpreter(
            model_path=str(path),
            experimental_preserve_all_tensors=True,
            experimental_op_resolver_type=(
                tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
            ),
        )
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.details = {
            detail["index"]: detail
            for detail in self.interpreter.get_tensor_details()
        }
        self.ops = self.interpreter._get_ops_details()
        self.capacity = int(self.input["shape"][1])

    def tensor(self, index):
        value = self.interpreter.get_tensor(int(index))
        scale, zero_point = self.details[int(index)]["quantization"]
        if scale:
            return (value.astype(np.float32) - zero_point) * scale
        return value

    def run(self, ids):
        if len(ids) > self.capacity:
            raise ValueError(f"{len(ids)} phonemes exceed capacity {self.capacity}")
        phoneme = np.zeros(self.input["shape"], dtype=self.input["dtype"])
        phoneme[0, :len(ids)] = ids
        self.interpreter.set_tensor(self.input["index"], phoneme)
        self.interpreter.invoke()
        outputs = [
            self.tensor(detail["index"])
            for detail in self.interpreter.get_output_details()
        ]
        mel = next(value for value in outputs if value.ndim == 3)
        length_value = next(value for value in outputs if value.shape == (1,))[0]
        if not np.isfinite(mel).all() or not np.isfinite(length_value):
            raise AssertionError("Non-finite TFLite output")
        length = int(np.rint(length_value))
        assert 0 <= length <= mel.shape[1], length
        np.testing.assert_array_equal(mel[:, length:], 0)

        # Only preserved buffers are valid for intermediate-tensor inspection.
        lookup = next(op for op in self.ops if op["op_name"] == "EMBEDDING_LOOKUP")
        lookup_ids = self.tensor(lookup["inputs"][0])
        np.testing.assert_array_equal(lookup_ids.reshape(phoneme.shape), phoneme)
        table = self.tensor(lookup["inputs"][1])
        np.testing.assert_array_equal(
            self.tensor(lookup["outputs"][0]), table[lookup_ids]
        )

        cumulative = next(op for op in self.ops if op["op_name"] == "CUMSUM")
        durations = np.rint(self.tensor(cumulative["inputs"][0])).astype(np.int32)
        assert np.all(durations >= 0)
        np.testing.assert_array_equal(durations[:, len(ids):], 0)
        expected_length = min(int(durations.sum()), mel.shape[1])
        assert length == expected_length, (length, expected_length)

        expansion = next(
            op for op in self.ops
            if op["op_name"] == "GATHER"
            and list(self.details[int(op["outputs"][0])]["shape"]) == list(mel.shape)
        )
        owners = self.tensor(expansion["inputs"][1]).reshape(-1)
        expected_owners = np.minimum(
            np.searchsorted(np.cumsum(durations), np.arange(mel.shape[1]), side="right"),
            self.capacity - 1,
        )
        np.testing.assert_array_equal(owners, expected_owners)
        return mel[0, :length].copy()

    def check_a16w8(self):
        count = 0
        for op in self.ops:
            if op["op_name"] in {"CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED"}:
                activation = self.details[int(op["inputs"][0])]
                weight = self.details[int(op["inputs"][1])]
                assert activation["dtype"] == np.int16, activation["name"]
                assert weight["dtype"] == np.int8, weight["name"]
                count += 1
        assert count > 0, "No A16W8 convolution/linear operators found"
        return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "outputs/grainspeech_a16w8.tflite")
    parser.add_argument("--reference", type=Path, default=ROOT / "outputs/grainspeech_fp32_reference.tflite")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/a16w8_comparison")
    args = parser.parse_args()
    reference = ExportRunner(args.reference)
    quantized = ExportRunner(args.model)
    assert reference.capacity == quantized.capacity
    quantized_ops = quantized.check_a16w8()
    manifest = json.loads((ROOT / "assets/examples/manifest.json").read_text())
    examples = manifest["examples"]
    cases = [
        (example["id"], text_to_sequence(example["arpabet"], ["english_cleaners"]))
        for example in examples
    ]
    rng = np.random.default_rng(317)
    for length in sorted({0, 1, 2, 8, 32, 64, quantized.capacity}):
        if length <= quantized.capacity:
            cases.append((f"random-{length}", rng.integers(1, 74, size=length).tolist()))
    if quantized.capacity >= 73:
        cases.append(("all-symbols", list(range(1, 74))))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    example_ids = {example["id"] for example in examples}
    for name, ids in cases:
        fp32 = reference.run(ids)
        a16w8 = quantized.run(ids)
        assert abs(len(fp32) - len(a16w8)) <= max(2, np.ceil(0.05 * len(fp32))), (
            name, len(fp32), len(a16w8)
        )
        common_length = min(len(fp32), len(a16w8))
        mae = (
            float(np.mean(np.abs(fp32[:common_length] - a16w8[:common_length])))
            if common_length else 0.0
        )
        record = {
            "id": name,
            "fp32_frames": len(fp32),
            "a16w8_frames": len(a16w8),
            "common_frame_mel_mae": mae,
        }
        if name in example_ids:
            record["text"] = next(e["text"] for e in examples if e["id"] == name)
            for label, mel in (("fp32", fp32), ("a16w8", a16w8)):
                np.save(args.output_dir / f"{name}_{label}.npy", mel)
            # The README arrays came from the original PyTorch checkpoint.
            published = np.load(ROOT / "assets/examples" / f"{name}.npy")
            np.testing.assert_allclose(fp32, published, atol=1e-4, rtol=1e-4)
        records.append(record)
        print(f"{name}: FP32={len(fp32)}, A16W8={len(a16w8)} frames, Mel MAE={mae:.5f}")

    report = {
        "model": str(args.model),
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "reference": str(args.reference),
        "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "tensorflow": tf.__version__,
        "a16w8_conv_linear_ops": quantized_ops,
        "sampling_rate": manifest["sampling_rate"],
        "hop_length": manifest["hop_length"],
        "mel_mae_note": "Unaligned common-frame MAE includes differences in phoneme timing.",
        "cases": records,
    }
    path = args.output_dir / "comparison.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
