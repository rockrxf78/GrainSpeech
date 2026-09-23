"""Check end-to-end TFLite inference using the exported models and README Mels."""

import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import warnings

import numpy as np
from scipy.io import wavfile

from infer_tflite import (
    ROOT, AcousticModel, main as infer_main, parse_args, parse_phonemes,
    text_to_arpabet,
)


def expect_value_error(function, message):
    try:
        function()
    except ValueError as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"Expected ValueError containing {message!r}")


def main():
    examples = json.loads((ROOT / "assets/examples/manifest.json").read_text())["examples"]
    for example in examples:
        assert text_to_arpabet(example["text"]) == parse_phonemes(example["arpabet"])
    expect_value_error(lambda: parse_phonemes(""), "No phonemes")
    expect_value_error(lambda: parse_phonemes("NOT_A_PHONE"), "Unsupported phonemes")

    model = AcousticModel(ROOT / "outputs/grainspeech_a16w8.tflite")
    expect_value_error(lambda: model.predict([]), "No phoneme IDs")
    expect_value_error(lambda: model.predict([74]), "[1, 73]")
    expect_value_error(
        lambda: model.predict([1] * (model.max_phonemes + 1)), "phonemes"
    )
    long_sequence = [1 + index % 73 for index in range(model.max_phonemes)]
    expect_value_error(lambda: model.predict(long_sequence), "may be truncated")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        clipped = model.predict(long_sequence, allow_truncation=True)
    assert clipped.shape == (model.max_mel_frames, 80)
    assert any("may be truncated" in str(warning.message) for warning in caught)

    for argv in (
        ["--text", "Hello", "--phonemes", "HH AH0 L OW1"],
        ["--text", "Hello", "--threads", "0"],
    ):
        with patch.object(sys, "argv", ["infer_tflite.py"] + argv):
            try:
                parse_args()
            except SystemExit as error:
                assert error.code == 2
            else:
                raise AssertionError(f"Invalid arguments accepted: {argv}")

    with tempfile.TemporaryDirectory(prefix="grainspeech-tflite-e2e-") as directory:
        output_dir = Path(directory)
        for index, example in enumerate(examples):
            name = example["id"]
            output = output_dir / f"{name}.wav"
            saved_mel = output_dir / f"{name}.NPY"
            source = (
                ["--text", example["text"]] if index == 0
                else ["--phonemes", example["arpabet"]]
            )
            argv = [
                "infer_tflite.py", *source,
                "--acoustic-model", str(ROOT / "outputs/grainspeech_a16w8.tflite"),
                "--vocoder-model", str(ROOT / "outputs/hifigan_a16w8.tflite"),
                "--vocoder-metadata", str(ROOT / "outputs/hifigan_a16w8.json"),
                "--threads", "2", "--save-mel", str(saved_mel), "--output", str(output),
            ]
            with patch.object(sys, "argv", argv):
                infer_main()
            mel = np.load(saved_mel)
            previous = np.load(ROOT / "outputs/a16w8_comparison" / f"{name}_a16w8.npy")
            np.testing.assert_array_equal(mel, previous)
            rate, pcm = wavfile.read(output)
            assert rate == 22050 and pcm.dtype == np.int16
            assert pcm.shape == (len(mel) * 256,) and np.any(pcm != 0)
            print(f"{name}: end-to-end WAV and saved Mel OK")
    assert "torch" not in sys.modules, "The TFLite pipeline must not import PyTorch"
    print("Text, phoneme, capacity, CLI, and no-PyTorch checks complete")


if __name__ == "__main__":
    main()
