"""Build GrainSpeech features from accepted, source-indexed Baker CTC alignments."""

import argparse
from contextlib import nullcontext
import copy
import json
from pathlib import Path
import random
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "grainspeech"))

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import yaml

from preprocessing.ctc import AlignmentError
from preprocessing.ljspeech import LJSpeechPreprocessor


class AlignedBakerPreprocessor(LJSpeechPreprocessor):
    def _get_alignment(self, tier):
        intervals = tier.intervals
        if not intervals:
            raise AlignmentError("Empty alignment")
        start = intervals[0].start_time
        ends = []
        phones = []
        previous = start
        for interval in intervals:
            if abs(interval.start_time - previous) > 1e-5 or interval.end_time <= previous:
                raise AlignmentError("Alignment has gaps, overlaps, or nonpositive intervals")
            phones.append(interval.text)
            ends.append(round((interval.end_time - start) * self.sample_rate / self.hop_length))
            previous = interval.end_time
        durations = np.diff([0] + ends).tolist()
        if any(duration <= 0 for duration in durations):
            raise AlignmentError("A syllable is shorter than one Mel frame")
        return phones, durations, start, previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/Baker")
    parser.add_argument("--alignments", type=Path, default=ROOT / "data/Baker-alignments")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/Baker-features")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--disable-cudnn", action="store_true",
                        help="Use native PyTorch CUDA kernels if installed cuDNN libraries conflict")
    parser.add_argument("--val-size", type=int)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    dataset, alignments, output = args.dataset.resolve(), args.alignments.resolve(), args.output_dir.resolve()
    for source in (dataset, alignments):
        if output == source or output in source.parents or source in output.parents:
            parser.error("Feature output must be separate from the prepared data and alignments")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a fresh feature output directory: {output}")
    summary = json.loads((alignments / "summary.json").read_text())
    source_bytes = (dataset / "manifest.jsonl").read_bytes()
    import hashlib

    if hashlib.sha256(source_bytes).hexdigest() != summary["manifest_sha256"]:
        raise ValueError("Alignment source manifest does not match this dataset")
    if summary["accepted"] < 2:
        raise ValueError("At least two accepted recordings are required")
    rows = [json.loads(line) for line in source_bytes.decode("utf-8").splitlines()]
    if "selected_ids" in summary:
        by_id = {row["id"]: row for row in rows}
        rows = [by_id[name] for name in summary["selected_ids"]]
    elif summary["limit"] is not None:
        rows = rows[:summary["limit"]]
    source_config = json.loads((dataset / "dataset.json").read_text())
    frontend = json.loads((alignments / "frontend.json").read_text(encoding="utf-8"))
    with (ROOT / "configs/LJSpeech/preprocess.yaml").open() as stream:
        config = copy.deepcopy(yaml.safe_load(stream))
    config["dataset"] = "Baker"
    config["path"] = {
        "corpus_path": str(dataset), "raw_path": str(dataset / "raw"),
        "preprocessed_path": str(output),
    }
    config["preprocessing"]["text"] = frontend
    config["preprocessing"]["audio"]["sampling_rate"] = source_config["sampling_rate"]
    for key in ("stft", "mel"):
        config["preprocessing"][key] = source_config["mel_preprocessing"][key]
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    extractor = AlignedBakerPreprocessor(config, torch.device(device), speaker="Baker")
    for directory in ("duration", "mel", "pitch", "energy", "TextGrid/Baker"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    metadata = {}
    rejections = []
    for index, row in enumerate(rows, start=1):
        name = row["id"]
        alignment = json.loads(
            (alignments / "records" / f"{name}.json").read_text(encoding="utf-8")
        )
        if alignment["status"] != "accepted":
            rejections.append({"id": name, "reason": alignment["reason"]})
            continue
        if any(character in row["text"] for character in "|\n\r"):
            raise ValueError(f"{name}: transcript cannot be represented in training metadata")
        expected = [entry["phone"] for entry in alignment["intervals"]]
        if any(phone not in frontend["symbols"] for phone in expected):
            raise ValueError(f"{name}: alignment phone outside frontend vocabulary")
        original = alignments / "TextGrid/Baker" / f"{name}.TextGrid"
        shutil.copy2(original, output / "TextGrid/Baker" / original.name)
        try:
            backend = torch.backends.cudnn.flags(enabled=False) if args.disable_cudnn else nullcontext()
            with backend:
                result = extractor.process_utterance(name)
            if result is None:
                raise AlignmentError("No usable voiced pitch frames")
            text_row = result[0]
            phones = text_row.split("|")[2].strip("{}").split()
            if phones != expected:
                raise ValueError(f"{name}: TextGrid and accepted alignment record disagree")
            duration = np.load(output / "duration" / f"Baker-duration-{name}.npy")
            mel = np.load(output / "mel" / f"Baker-mel-{name}.npy")
            if len(duration) != len(phones) or duration.sum() != len(mel):
                raise ValueError(f"{name}: duration/phone/Mel length mismatch")
            for feature in ("pitch", "energy", "mel"):
                value = np.load(output / feature / f"Baker-{feature}-{name}.npy")
                if not np.isfinite(value).all():
                    raise AlignmentError(f"Non-finite {feature}")
            metadata[name] = text_row
        except AlignmentError as error:
            rejections.append({"id": name, "reason": str(error)})
            print(f"Rejected feature record {name}: {error}", flush=True)
        if index % 100 == 0:
            print(f"Extracted {index}/{len(rows)} selected recordings", flush=True)
    (output / "rejected.json").write_text(
        json.dumps(rejections, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    if len(metadata) < 2:
        raise RuntimeError("Fewer than two usable feature records; do not train")
    names = list(metadata)
    random.Random(args.seed).shuffle(names)
    val_size = args.val_size if args.val_size is not None else min(256, max(1, len(names) // 20))
    if not 1 <= val_size < len(names):
        raise ValueError("--val-size must leave at least one training and validation recording")
    validation, training = names[:val_size], names[val_size:]
    stats = {}
    for feature in ("pitch", "energy"):
        scaler = StandardScaler()
        for name in training:
            values = np.load(output / feature / f"Baker-{feature}-{name}.npy")
            filtered = extractor._remove_outlier(values)
            if len(filtered):
                scaler.partial_fit(filtered.reshape(-1, 1))
        if not hasattr(scaler, "mean_"):
            raise RuntimeError(f"No nonconstant training values for {feature} statistics")
        mean, std = float(scaler.mean_[0]), float(scaler.scale_[0])
        extractor._normalize_files(feature, names, mean, std)
        training_values = [
            np.load(output / feature / f"Baker-{feature}-{name}.npy") for name in training
        ]
        stats[feature] = [
            min(float(value.min()) for value in training_values),
            max(float(value.max()) for value in training_values), mean, std,
        ]
    config["preprocessing"]["val_size"] = val_size
    for split, selected in (("train", training), ("val", validation)):
        (output / f"{split}.txt").write_text(
            "".join(metadata[name] + "\n" for name in selected), encoding="utf-8",
        )
    (output / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    (output / "speakers.json").write_text('{"Baker": 0}\n')
    (output / "preprocess.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8",
    )
    (output / "dataset.json").write_text(json.dumps({
        "training_ready": True, "train": len(training), "validation": len(validation),
        "rejected": len(rejections), "seed": args.seed,
        "alignment": summary, "statistics_fit": "training_split_only",
        "unit": "tone_marked_pinyin_syllable",
    }, indent=2) + "\n")
    print(f"Ready: train={len(training)}, val={len(validation)}, rejected={len(rejections)}")
    print(f"Training configuration: {output / 'preprocess.yaml'}")


if __name__ == "__main__":
    main()
