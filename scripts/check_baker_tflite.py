"""Compare Chinese FP32/A16W8 exports and render matched listening samples."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np
from scipy.io import wavfile

from export_tflite import calibration_inputs
from infer_tflite import ROOT, AcousticModel, main as infer_main
from text.vocabulary import pinyin_to_phones


SAMPLES = {
    "welcome": "ni2 hao3 huan1 ying2 shi3 yong4 zhong1 wen2 yu3 yin1 he2 cheng2",
    "weather": "jin1 tian1 de5 tian1 qi4 hen2 hao3",
    "training": "wo3 men5 zheng4 zai4 xun4 lian4 yi2 ge4 xiao3 xing2 zhong1 wen2 yu3 yin1 mo2 xing2",
}


def expect_error(function, fragment):
    try:
        function()
    except ValueError as error:
        assert fragment in str(error), str(error)
    else:
        raise AssertionError(f"Expected ValueError containing {fragment!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp32", type=Path, default=ROOT / "outputs/grainspeech_baker_fp32.tflite")
    parser.add_argument("--quantized", type=Path, default=ROOT / "outputs/grainspeech_baker_a16w8.tflite")
    parser.add_argument("--reference-dir", type=Path, default=ROOT / "outputs/baker_trained_mels")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/baker_a16w8_audio")
    args = parser.parse_args()
    models = {kind: AcousticModel(path) for kind, path in (("fp32", args.fp32), ("a16w8", args.quantized))}
    quantized = models["a16w8"]
    assert quantized.language == "zh" and quantized.vocab_size > 74
    assert quantized.frontend == models["fp32"].frontend
    assert quantized.metadata["checkpoint_sha256"] == models["fp32"].metadata["checkpoint_sha256"]
    assert quantized.metadata["calibration"]["maximum_token_id"] > 73
    expect_error(
        lambda: AcousticModel(args.quantized, metadata_path=args.fp32.with_suffix(".json")),
        "hash mismatch",
    )
    expect_error(lambda: quantized.predict([quantized.vocab_size]), "integer phoneme IDs")
    expect_error(lambda: pinyin_to_phones("notapinyin1", quantized.frontend["lexicon"]), "Unknown")
    with tempfile.TemporaryDirectory(prefix="baker-calibration-check-") as directory:
        path = Path(directory) / "train.txt"
        token = quantized.frontend["symbols"][-1]
        path.write_text(f"short|Baker|{{{token}}}|example\n", encoding="utf-8")
        arrays, information = calibration_inputs(path, quantized.frontend, 8, 8)
        assert len(arrays) == information["samples"] == 1
        assert arrays[0][0][0, 0] == quantized.vocab_size - 1
        np.testing.assert_array_equal(arrays[0][0][0, 1:], 0)
        path.write_text("bad|Baker|{unknown}|example\n")
        expect_error(lambda: calibration_inputs(path, quantized.frontend, 8, 1), "Unknown")

    details = {detail["index"]: detail for detail in quantized.interpreter.get_tensor_details()}
    operators = Counter()
    for operation in quantized.interpreter._get_ops_details():
        if operation["op_name"] in ("CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED"):
            activation = details[operation["inputs"][0]]["dtype"]
            weight = details[operation["inputs"][1]]["dtype"]
            assert activation == np.int16 and weight == np.int8
            operators[operation["op_name"]] += 1
    assert operators
    results = []
    for name, pinyin in SAMPLES.items():
        mels = {}
        wave_files = {}
        for kind, model_path in (("fp32", args.fp32), ("a16w8", args.quantized)):
            mel_path = args.output_dir / "mel" / f"{name}_{kind}.npy"
            wav_path = args.output_dir / f"{name}_{kind}.wav"
            argv = [
                "infer_tflite.py", "--acoustic-model", str(model_path),
                "--pinyin", pinyin, "--save-mel", str(mel_path), "--output", str(wav_path),
            ]
            with patch.object(sys, "argv", argv):
                infer_main()
            mel = np.load(mel_path)
            rate, pcm = wavfile.read(wav_path)
            assert rate == 22050 and pcm.dtype == np.int16 and pcm.ndim == 1
            assert len(pcm) == len(mel) * 256 and np.any(pcm != 0)
            mels[kind] = mel
            wave_files[kind] = str(wav_path)
        reference = np.load(args.reference_dir / f"{name}_best.npy")
        np.testing.assert_allclose(mels["fp32"], reference, atol=1e-4, rtol=1e-5)
        assert abs(len(mels["a16w8"]) - len(reference)) <= max(2, round(len(reference) * 0.05))
        common = min(len(mels["fp32"]), len(mels["a16w8"]))
        error = float(np.mean(np.abs(mels["fp32"][:common] - mels["a16w8"][:common])))
        result = {
            "id": name, "pinyin": pinyin, "fp32_frames": len(reference),
            "a16w8_frames": len(mels["a16w8"]), "common_frame_mel_mae": error,
            "audio": wave_files,
        }
        results.append(result)
        print(f"{name}: FP32={len(reference)}, A16W8={len(mels['a16w8'])} frames; MAE={error:.6f}")
    report = {
        "acoustic_fp32_sha256": models["fp32"].metadata["tflite_sha256"],
        "acoustic_a16w8_sha256": quantized.metadata["tflite_sha256"],
        "a16w8_operators": dict(operators), "samples": results,
        "vocoder": "A16W8 HiFi-GAN for both comparisons",
    }
    (args.output_dir / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    assert "torch" not in sys.modules, "TFLite inference must not import PyTorch"


if __name__ == "__main__":
    main()
