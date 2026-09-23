"""Generate Chinese mels from tone-marked pinyin syllables (no vocoder).

Supply numbered tones explicitly; no Chinese normalization or sandhi is inferred.
The Baker CTC frontend aligns syllable units, not individual phones. Run
infer_hifigan_tflite.py separately for WAV synthesis; do not mix the runtimes.
"""

import argparse
from contextlib import nullcontext
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "grainspeech"))
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from model_l1_ssim_gvar import GrainSpeech
from text.vocabulary import pinyin_to_phones, vocabulary_from_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Trusted Chinese acoustic checkpoint")
    parser.add_argument("--pinyin", required=True,
                        help='Tone-marked pinyin syllables and sp pauses, e.g. "ni3 sp hao3"')
    parser.add_argument("--output", type=Path, required=True, help="Output .npy mel array")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--disable-cudnn", action="store_true")
    args = parser.parse_args()
    if args.output.suffix != ".npy":
        parser.error("--output must end in .npy")
    model = GrainSpeech.load_from_checkpoint(
        args.checkpoint, map_location="cpu", weights_only=False, hifigan_checkpoint=None,
    )
    config = model.hparams.preprocess_config
    text = config["preprocessing"]["text"]
    if text.get("language") != "zh":
        raise ValueError("Expected a Chinese checkpoint with embedded frontend metadata")
    vocabulary = vocabulary_from_config(config)
    phones = pinyin_to_phones(args.pinyin, text["lexicon"])
    ids = vocabulary.encode(phones)
    device = torch.device(args.device)
    model.to(device).eval()
    batch = {
        "phoneme": torch.tensor([ids], dtype=torch.long, device=device),
        "phoneme_mask": torch.zeros((1, len(ids)), dtype=torch.bool, device=device),
    }
    backend = torch.backends.cudnn.flags(enabled=False) if args.disable_cudnn else nullcontext()
    with torch.inference_mode(), backend:
        mel, lengths, _ = model.phoneme2mel(batch, train=False)
    mel = mel[0, :int(lengths[0])].float().cpu().numpy()
    if mel.ndim != 2 or mel.shape[1] != 80 or not len(mel) or not np.isfinite(mel).all():
        raise ValueError("Model produced an invalid mel array")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, mel, allow_pickle=False)
    print(f"Saved {mel.shape} mel array to {args.output}; quality depends on training.")


if __name__ == "__main__":
    main()
