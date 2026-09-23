"""Check vocoder export with synthetic networks, without any pretrained files.

These deliberately small configurations are test fixtures, not a reconstruction
of the repository checkpoint's configuration or an assessment of its speech.
"""

import hashlib
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
import warnings

import numpy as np
import torch

from export_hifigan_tflite import (
    StaticHiFiGAN, calibration_windows, context_frames, convert_model,
    inspect_model, load_generator, pytorch_reference, run_interpreter, validate_config,
)
from hifigan import AttrDict, Generator
from infer_hifigan_tflite import main as infer_main, synthesize


def check_variant(kind):
    config = {
        "resblock": kind,
        "upsample_rates": [2, 3],
        "upsample_kernel_sizes": [4, 7],
        "upsample_initial_channel": 16,
        "resblock_kernel_sizes": [3, 5],
        "resblock_dilation_sizes": (
            [[1, 2, 3], [2, 3, 4]] if kind == "1" else [[1, 2], [2, 3]]
        ),
        "sampling_rate": 16000,
        "hop_size": 6,
    }
    validate_config(config)
    torch.manual_seed(819 + int(kind))
    generator = Generator(AttrDict(config)).eval()
    # Exercise checkpoint loading and folding of weight normalization.
    with tempfile.TemporaryDirectory(prefix="synthetic-hifigan-") as directory:
        checkpoint = Path(directory) / "weights.pt"
        torch.save({"generator": generator.state_dict()}, checkpoint)
        normalized = load_generator(checkpoint, config)
    frames = 64
    module = StaticHiFiGAN(normalized.state_dict(), config, frames)
    rng = np.random.default_rng(135 + int(kind))
    mel = rng.normal(-3, 2, (frames * 2 + 9, 80)).astype(np.float32)
    shorter = rng.normal(-4, 2, (7, 80)).astype(np.float32)
    representative = lambda: calibration_windows([mel, shorter], frames, 12)
    sample = next(representative())[0]
    original = pytorch_reference(generator, sample)
    reference = pytorch_reference(normalized, sample)
    np.testing.assert_allclose(original, reference, atol=1e-6, rtol=1e-5)
    np.testing.assert_allclose(module(sample).numpy(), reference, atol=1e-5, rtol=1e-4)

    metadata = {
        "input": {"shape": [1, frames, 80]},
        "output": {"shape": [1, frames * 6]},
        "context_frames": context_frames(config),
        "hop_length": 6,
    }
    assert sum(metadata["context_frames"].values()) < frames
    for float32 in (True, False):
        model = convert_model(module, representative, float32)
        interpreter, operators = inspect_model(model)
        predicted = run_interpreter(interpreter, sample)
        if float32:
            np.testing.assert_allclose(predicted, reference, atol=1e-5, rtol=1e-4)
        else:
            assert operators["CONV_2D int16/int8"] > 0
            assert operators["TRANSPOSE_CONV int16/int8"] == 2
            # An intentionally loose bound for random synthetic weights, not speech quality.
            assert float(np.mean(np.abs(predicted - reference))) < 0.01
        for length in (frames, frames + 1, frames * 2 + 9):
            waveform = synthesize(mel[:length], interpreter, metadata)
            assert waveform.shape == (length * 6,)
            assert np.isfinite(waveform).all()
            if float32:
                complete = pytorch_reference(normalized, mel[None, :length])[0]
                np.testing.assert_allclose(waveform, complete, atol=1e-5, rtol=1e-4)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            short_audio = synthesize(shorter, interpreter, metadata)
        assert any("Padding" in str(warning.message) for warning in caught)
        assert short_audio.shape == (len(shorter) * 6,)
        if float32:
            padded = np.zeros((1, frames, 80), np.float32)
            padded[0, :len(shorter)] = shorter
            expected = pytorch_reference(normalized, padded)[0]
            np.testing.assert_allclose(short_audio, expected[:len(short_audio)], atol=1e-5, rtol=1e-4)
        precision = "FP32" if float32 else "A16W8"
        print(f"Synthetic ResBlock{kind} {precision}: parity and chunk boundaries OK; {operators}")
        if not float32:
            from scipy.io import wavfile

            with tempfile.TemporaryDirectory(prefix="synthetic-hifigan-wav-") as directory:
                root = Path(directory)
                model_path = root / "toy.tflite"
                model_path.write_bytes(model)
                model_path.with_suffix(".json").write_text(json.dumps({
                    **metadata,
                    "sampling_rate": config["sampling_rate"],
                    "tflite_sha256": hashlib.sha256(model).hexdigest(),
                }))
                mel_path = root / "toy-mel.npy"
                np.save(mel_path, mel)
                args = [
                    "infer_hifigan_tflite.py", "--model", str(model_path),
                    "--mel", str(mel_path), "--output-dir", str(root / "audio"),
                ]
                with patch.object(sys, "argv", args):
                    infer_main()
                rate, pcm = wavfile.read(root / "audio/toy-mel.wav")
                assert rate == config["sampling_rate"]
                assert pcm.dtype == np.int16 and pcm.shape == (len(mel) * 6,)


def check_mixed_runtime_reference():
    # This larger synthetic shape exposed the ARM mixed-runtime Conv1d crash.
    config = {
        "resblock": "2",
        "upsample_rates": [3, 4, 3],
        "upsample_kernel_sizes": [7, 8, 7],
        "upsample_initial_channel": 96,
        "resblock_kernel_sizes": [3, 9],
        "resblock_dilation_sizes": [[1, 2], [2, 4]],
        "sampling_rate": 16000,
    }
    torch.manual_seed(812)
    generator = Generator(AttrDict(config)).eval()
    generator.remove_weight_norm()
    module = StaticHiFiGAN(generator.state_dict(), config, 64)
    mel = np.random.default_rng(42).normal(-3, 2, (1, 64, 80)).astype(np.float32)
    original_backend = torch.backends.mkldnn.enabled
    reference = pytorch_reference(generator, mel)
    assert torch.backends.mkldnn.enabled == original_backend
    np.testing.assert_allclose(module(mel).numpy(), reference, atol=1e-5, rtol=1e-4)
    print("Larger mixed-runtime reference: parity OK; PyTorch backend restored")


def main():
    torch.set_num_threads(1)
    check_mixed_runtime_reference()
    check_variant("1")
    check_variant("2")
    print("Synthetic checks complete; pretrained checkpoint conversion must be run separately.")


if __name__ == "__main__":
    main()
