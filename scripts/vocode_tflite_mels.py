"""Generate paired A16W8/FP32 WAVs from check_tflite_export.py's saved Mels.

Run separately from TensorFlow inference. Both variants use the same FP32
HiFi-GAN vocoder; this script does not re-run either acoustic model.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mel-dir", type=Path, default=ROOT / "outputs/a16w8_comparison",
        help="Directory containing comparison.json and saved Mel arrays",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/a16w8_audio",
    )
    parser.add_argument(
        "--hifigan-checkpoint", type=Path,
        default=ROOT / "hifigan/LJ_V2/generator_v2",
        help="HiFi-GAN checkpoint with its config.json in the same directory",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def load_mel(path, expected_frames):
    import numpy as np

    mel = np.load(path, allow_pickle=False)
    if expected_frames <= 0 or mel.shape != (expected_frames, 80):
        raise ValueError(
            f"{path}: expected nonempty [{expected_frames}, 80] Mel, got {mel.shape}"
        )
    if not np.issubdtype(mel.dtype, np.floating) or not np.isfinite(mel).all():
        raise ValueError(f"{path}: expected finite, already-dequantized floating-point Mel")
    return np.ascontiguousarray(mel.T, dtype=np.float32)


def main():
    args = parse_args()
    import numpy as np
    import torch
    from scipy.io import wavfile

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    report_path = args.mel_dir / "comparison.json"
    with report_path.open(encoding="utf-8") as stream:
        report = json.load(stream)
    sample_rate = report["sampling_rate"]
    hop_length = report["hop_length"]
    for name, value in (("sampling_rate", sample_rate), ("hop_length", hop_length)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{report_path}: {name} must be a positive integer")

    cases = [case for case in report["cases"] if "text" in case]
    if not cases:
        raise ValueError(f"{report_path}: no speech examples found")
    inputs = []
    for case in cases:
        name = case["id"]
        if not name or Path(name).name != name or name in (".", ".."):
            raise ValueError(f"{report_path}: invalid example ID {name!r}")
        for variant in ("a16w8", "fp32"):
            path = args.mel_dir / f"{name}_{variant}.npy"
            inputs.append((case, variant, path, load_mel(path, case[f"{variant}_frames"])))

    # Reuse the repository loader. Only this step reads the vocoder configuration.
    sys.path.insert(0, str(ROOT / "grainspeech"))
    from model_l1_ssim_gvar import get_hifigan

    torch.set_num_threads(1)
    vocoder = get_hifigan(
        str(args.hifigan_checkpoint), infer_device=args.device,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for case, variant, path, mel in inputs:
        with torch.inference_mode():
            audio = vocoder(torch.from_numpy(mel).unsqueeze(0).to(args.device))
        expected_samples = mel.shape[1] * hop_length
        if tuple(audio.shape) != (1, 1, expected_samples):
            raise RuntimeError(
                f"{path}: vocoder returned {tuple(audio.shape)}, "
                f"expected (1, 1, {expected_samples}); check the vocoder hop length"
            )
        waveform = audio[0, 0].float().cpu().numpy()
        if not np.isfinite(waveform).all():
            raise RuntimeError(f"{path}: vocoder produced non-finite samples")
        pcm = np.clip(waveform * 32768.0, -32768, 32767).astype(np.int16)
        output = args.output_dir / f"{path.stem}.wav"
        wavfile.write(output, sample_rate, pcm)
        print(
            f"{variant.upper()}: {case['text']}\n"
            f"Wrote {output} ({len(pcm) / sample_rate:.3f} s, {sample_rate} Hz)",
            flush=True,
        )


if __name__ == "__main__":
    main()
