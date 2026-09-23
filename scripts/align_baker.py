"""Align Baker Hanzi to audio with pretrained Mandarin CTC, retaining supplied tones.

The acoustic units are pinyin SYLLABLES, not guessed initial/final splits.
Rejected examples are reported explicitly. No uniform-duration fallback is used.
"""

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import random
import sys

os.environ.setdefault("USE_TF", "0")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "grainspeech"))

import librosa
import numpy as np
import soundfile as sf
import tgt
import torch

from preprocessing.ctc import (
    AlignmentError, character_error_rate, hanzi_pinyin, syllable_intervals, viterbi_spans,
)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/Baker")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/Baker-alignments")
    parser.add_argument("--model", default="wbbbbb/wav2vec2-large-chinese-zh-cn")
    parser.add_argument("--revision", default="b654b5f3df69725df32808de454a1b865c829e4c")
    parser.add_argument("--safetensors", action=argparse.BooleanOptionalAction, default=True,
                        help="Require safetensors weights; use --no-safetensors for a trusted .bin model")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "external/alignment-models")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--disable-cudnn", action="store_true",
                        help="Use native PyTorch CUDA kernels if installed cuDNN libraries conflict")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int, help="Use the first N recordings")
    selection.add_argument("--sample-size", type=int, help="Use a reproducible random corpus sample")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-cer", type=float, default=0.35)
    parser.add_argument("--min-confidence", type=float, default=0.01)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.sample_size is not None and args.sample_size <= 0:
        parser.error("--sample-size must be positive")
    if not 0 <= args.max_cer <= 1 or not 0 <= args.min_confidence <= 1:
        parser.error("Confidence and CER thresholds must lie in [0, 1]")
    dataset, output = args.dataset.resolve(), args.output_dir.resolve()
    if output == dataset or output in dataset.parents:
        parser.error("Alignment output must not overwrite the prepared dataset")
    manifest_path = dataset / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Dataset must contain nonempty, unique recording IDs")
    prepared = {}
    vocabulary = set()
    for row in rows:
        name = row["id"]
        if Path(name).name != name or name in (".", ".."):
            raise ValueError(f"Unsafe recording ID: {name!r}")
        try:
            value = hanzi_pinyin(row["text"], row["text_pinyin"])
        except AlignmentError as error:
            prepared[name] = str(error)
        else:
            prepared[name] = value
            vocabulary.update(value[1])
    if not vocabulary:
        raise ValueError("No supported Hanzi/pinyin recordings")

    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    options = {"revision": args.revision, "cache_dir": str(args.cache_dir)}
    processor = Wav2Vec2Processor.from_pretrained(args.model, **options)
    model, loading = Wav2Vec2ForCTC.from_pretrained(
        args.model, weights_only=True, use_safetensors=args.safetensors,
        attn_implementation="eager", output_loading_info=True, **options,
    )
    if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"Alignment model did not load exactly: {loading}")
    model.eval().to(device)
    tokenizer = processor.tokenizer
    token_ids = tokenizer.get_vocab()
    sampling_rate = processor.feature_extractor.sampling_rate
    stride = 1
    receptive_field = 1
    for kernel, step in zip(model.config.conv_kernel, model.config.conv_stride):
        receptive_field += (kernel - 1) * stride
        stride *= step
    settings = {
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "model": args.model, "revision": model.config._commit_hash,
        "limit": args.limit, "max_cer": args.max_cer, "min_confidence": args.min_confidence,
        "sample_size": args.sample_size, "seed": args.seed,
        "safetensors": args.safetensors,
        "method": "pretrained_mandarin_ctc_pinyin_syllables_v1",
        "boundary_policy": "midpoint_between_CTC_spans; outer_edges_at_recording_bounds",
        "ctc_sampling_rate": sampling_rate,
        "ctc_seconds_per_frame": stride / sampling_rate,
    }
    if output.exists() and any(output.iterdir()):
        if not args.resume:
            raise FileExistsError(f"Choose a fresh output directory or use --resume: {output}")
        if json.loads((output / "run.json").read_text()) != settings:
            raise ValueError("Cannot resume with changed source, model, subset, or thresholds")
    output.mkdir(parents=True, exist_ok=True)
    (output / "records").mkdir(exist_ok=True)
    (output / "TextGrid" / "Baker").mkdir(parents=True, exist_ok=True)
    write_json(output / "run.json", settings)
    write_json(output / "frontend.json", {
        "language": "zh", "unit": "tone_marked_pinyin_syllable",
        "symbols": ["_", "sp", "sil"] + sorted(vocabulary),
        "lexicon": {syllable: [syllable] for syllable in sorted(vocabulary)},
        "text_cleaners": [], "max_length": 4096,
    })
    selected = (
        random.Random(args.seed).sample(rows, min(args.sample_size, len(rows)))
        if args.sample_size is not None else rows[:args.limit] if args.limit else rows
    )
    accepted = rejected = 0
    for index, row in enumerate(selected, start=1):
        name = row["id"]
        record_path = output / "records" / f"{name}.json"
        textgrid_path = output / "TextGrid/Baker" / f"{name}.TextGrid"
        if record_path.exists():
            result = json.loads(record_path.read_text(encoding="utf-8"))
            if result["status"] == "accepted" and not textgrid_path.is_file():
                raise FileNotFoundError(f"Missing TextGrid for completed record: {name}")
        else:
            result = {"id": name, "status": "rejected"}
            try:
                item = prepared[name]
                if isinstance(item, str):
                    raise AlignmentError(item)
                characters, syllables, groups = item
                unknown = sorted(set(characters) - token_ids.keys())
                if unknown:
                    raise AlignmentError(f"Characters outside CTC vocabulary: {unknown}")
                wav_path = (dataset / row["wav"]).resolve()
                if dataset not in wav_path.parents:
                    raise ValueError(f"Recording lies outside prepared dataset: {wav_path}")
                waveform, rate = sf.read(wav_path, dtype="float32")
                if waveform.ndim != 1 or not len(waveform) or not np.isfinite(waveform).all():
                    raise AlignmentError("Expected finite nonempty mono audio")
                duration = len(waveform) / rate
                audio = librosa.resample(waveform, orig_sr=rate, target_sr=sampling_rate)
                inputs = processor(
                    audio, sampling_rate=sampling_rate, return_tensors="pt",
                ).to(device)
                backend = torch.backends.cudnn.flags(enabled=False) if args.disable_cudnn else nullcontext()
                with torch.inference_mode(), backend:
                    logits = model(**inputs).logits[0].float()
                    log_probs = logits.log_softmax(-1).cpu().numpy()
                hypothesis = tokenizer.decode(logits.argmax(-1).cpu().tolist()).replace(" ", "")
                cer = character_error_rate("".join(characters), hypothesis)
                result.update(
                    hypothesis=hypothesis, cer=cer, hanzi="".join(characters),
                    syllables=syllables, syllable_character_ranges=groups,
                )
                if cer > args.max_cer:
                    raise AlignmentError(f"Greedy ASR CER {cer:.3f} exceeds {args.max_cer}")
                spans = viterbi_spans(
                    log_probs, [token_ids[character] for character in characters],
                    model.config.pad_token_id,
                )
                confidence = min(span[2] for span in spans)
                result.update(
                    character_ctc_spans=spans, min_character_confidence=confidence,
                )
                if confidence < args.min_confidence:
                    raise AlignmentError(f"Minimum character confidence {confidence:.5f} too low")
                intervals = syllable_intervals(
                    spans, syllables, groups, stride / sampling_rate,
                    (receptive_field - 1) / (2 * sampling_rate), duration,
                )
                grid = tgt.TextGrid()
                tier = tgt.IntervalTier(start_time=0, end_time=duration, name="phones")
                for interval in intervals:
                    tier.add_interval(tgt.Interval(
                        interval["start"], interval["end"], interval["phone"],
                    ))
                grid.add_tier(tier)
                tgt.io.write_to_file(grid, str(textgrid_path), format="long")
                result.update(
                    status="accepted", intervals=intervals, cer=cer,
                    min_character_confidence=confidence, hypothesis=hypothesis,
                )
            except AlignmentError as error:
                result["reason"] = str(error)
            write_json(record_path, result)
        if result["status"] == "accepted":
            accepted += 1
        else:
            rejected += 1
            print(f"Rejected {name}: {result['reason']}", flush=True)
        if index % 25 == 0 or index == len(selected):
            print(f"{index}/{len(selected)}: accepted={accepted}, rejected={rejected}", flush=True)
    write_json(output / "summary.json", {
        **settings, "selected": len(selected), "accepted": accepted, "rejected": rejected,
        "selected_ids": [row["id"] for row in selected],
        "automatic_alignment": True,
        "note": "Review boundaries before full training; CTC syllable estimates are not ground truth.",
    })
    if accepted < 2:
        raise RuntimeError("Fewer than two accepted recordings; do not start training")


if __name__ == "__main__":
    main()
