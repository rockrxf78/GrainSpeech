"""Synthesize WAVs from saved Mel arrays using an exported TFLite HiFi-GAN.

Long utterances use overlapping Mel windows, discard boundary context, and
concatenate only the valid center samples. No PyTorch vocoder is loaded.
"""

import argparse
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np

from export_hifigan_tflite import ROOT, inspect_model, load_mel, run_interpreter


def synthesize(mel, interpreter, metadata):
    frames = int(metadata["input"]["shape"][1])
    hop = metadata["hop_length"]
    total = len(mel)
    if total <= frames:
        if total < frames:
            warnings.warn(
                f"Padding {total} Mel frames to {frames}; the final boundary can "
                "differ from an unpadded, variable-length PyTorch vocoder. "
                "Export with --mel-frames equal to this utterance length for exact boundaries.",
                stacklevel=2,
            )
        window = np.zeros((1, frames, 80), dtype=np.float32)
        window[0, :total] = mel
        return run_interpreter(interpreter, window)[0, :total * hop]

    left = metadata["context_frames"]["left"]
    right = metadata["context_frames"]["right"]
    if frames <= left + right:
        raise ValueError(
            f"Window size {frames} cannot fit left/right contexts {left}/{right}; "
            f"re-export with --mel-frames greater than {left + right}"
        )
    pieces = []
    position = 0
    while position < total:
        start = max(0, position - left)
        if start + frames >= total:
            # Anchor the final window to the genuine utterance boundary.
            start = total - frames
            end = total
        else:
            end = start + frames - right
        window = np.ascontiguousarray(mel[None, start:start + frames], dtype=np.float32)
        audio = run_interpreter(interpreter, window)[0]
        pieces.append(audio[(position - start) * hop:(end - start) * hop])
        position = end
    waveform = np.concatenate(pieces)
    if len(waveform) != total * hop:
        raise RuntimeError("Chunk assembly changed the waveform length")
    return waveform


def load_vocoder(model_path, metadata_path=None, num_threads=1):
    metadata_path = metadata_path or model_path.with_suffix(".json")
    with metadata_path.open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    model = model_path.read_bytes()
    if hashlib.sha256(model).hexdigest() != metadata["tflite_sha256"]:
        raise ValueError("TFLite/metadata hash mismatch; use the matching export JSON")
    interpreter, _ = inspect_model(model, num_threads=num_threads)
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("Expected a single-input, single-output vocoder")
    if inputs[0]["dtype"] != np.float32 or outputs[0]["dtype"] != np.float32:
        raise ValueError("Expected float32 vocoder interfaces")
    if (
        list(inputs[0]["shape"]) != metadata["input"]["shape"]
        or list(outputs[0]["shape"]) != metadata["output"]["shape"]
    ):
        raise ValueError("TFLite/metadata shape mismatch")
    frames = metadata["input"]["shape"][1]
    hop = metadata["hop_length"]
    sample_rate = metadata["sampling_rate"]
    if (
        type(frames) is not int or frames <= 0
        or type(hop) is not int or hop <= 0
        or type(sample_rate) is not int or sample_rate <= 0
        or metadata["input"]["shape"] != [1, frames, 80]
        or metadata["output"]["shape"] != [1, frames * hop]
    ):
        raise ValueError("Invalid vocoder dimensions, hop length, or sample rate")
    if any(
        type(metadata["context_frames"][side]) is not int
        or metadata["context_frames"][side] < 0
        for side in ("left", "right")
    ):
        raise ValueError("Context sizes must be nonnegative integers")
    return interpreter, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "outputs/hifigan_a16w8.tflite")
    parser.add_argument("--metadata", type=Path, help="Default: JSON adjacent to the TFLite model")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--mel", type=Path, nargs="+", help="One or more [frames, 80] .npy files")
    source.add_argument("--mel-dir", type=Path, default=ROOT / "outputs/a16w8_comparison")
    parser.add_argument("--output-dir", type=Path, help="Default: outputs/<model-stem>_audio")
    args = parser.parse_args()
    interpreter, metadata = load_vocoder(args.model, args.metadata)
    sample_rate = metadata["sampling_rate"]
    paths = args.mel if args.mel else sorted(args.mel_dir.glob("*.npy"))
    if not paths:
        raise FileNotFoundError("No Mel .npy files found")
    if len({path.stem for path in paths}) != len(paths):
        raise ValueError("Mel filenames must have unique stems to avoid WAV collisions")
    output_dir = args.output_dir or ROOT / "outputs" / f"{args.model.stem}_audio"
    output_dir.mkdir(parents=True, exist_ok=True)
    from scipy.io import wavfile

    for path in paths:
        waveform = synthesize(load_mel(path), interpreter, metadata)
        pcm = np.clip(waveform * 32768.0, -32768, 32767).astype(np.int16)
        output = output_dir / f"{path.stem}.wav"
        wavfile.write(output, sample_rate, pcm)
        print(f"Wrote {output} ({len(pcm) / sample_rate:.3f} s, {sample_rate} Hz)")


if __name__ == "__main__":
    main()
