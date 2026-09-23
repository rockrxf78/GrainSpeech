"""Import original Baker Parquet recordings without using MOSS codec tokens.

Creates WAV/lab files, a source-indexed JSONL manifest, and optional reference
Mels. This is corpus preparation, not phoneme alignment or Chinese training.
"""

import argparse
import io
import json
from pathlib import Path
import re
import sys

import librosa
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "grainspeech"))


def shard_rows(paths):
    for path in paths:
        parquet = pq.ParquetFile(path)
        columns = ["text", "text_pinyin", "audio"]
        if "file_name" in parquet.schema_arrow.names:
            columns.append("file_name")
        index = 0
        for batch in parquet.iter_batches(
            batch_size=1, columns=columns, use_threads=False,
        ):
            for row in batch.to_pylist():
                yield path, index, row
                index += 1


def decode_record(row, name, sample_rate):
    for key in ("text", "text_pinyin"):
        if not isinstance(row[key], str) or not row[key].strip():
            raise ValueError(f"{name}: missing {key}")
    audio = row["audio"]
    if not isinstance(audio, dict) or not audio.get("bytes"):
        raise ValueError(f"{name}: expected embedded original audio bytes")
    waveform, source_rate = sf.read(
        io.BytesIO(audio["bytes"]), dtype="float32", always_2d=True,
    )
    if not len(waveform) or not np.isfinite(waveform).all():
        raise ValueError(f"{name}: empty or non-finite recording")
    channels = waveform.shape[1]
    waveform = waveform.mean(axis=1)
    if source_rate != sample_rate:
        waveform = librosa.resample(
            waveform, orig_sr=source_rate, target_sr=sample_rate,
        )
    peak = float(np.max(np.abs(waveform)))
    if peak <= 0:
        raise ValueError(f"{name}: silent recording")
    return waveform * (0.95 / peak), source_rate, channels


def prepare(corpus, output, config, limit=None, mel_samples=0):
    corpus = Path(corpus).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    if mel_samples < 0:
        raise ValueError("mel_samples must be nonnegative")
    if output == corpus or corpus in output.parents or output in corpus.parents:
        raise ValueError("Output must not overlap the source corpus")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output is not empty; choose a new directory: {output}")
    paths = sorted(corpus.glob("train-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No train-*.parquet files under {corpus}")
    source_rows = 0
    for path in paths:
        parquet = pq.ParquetFile(path)
        missing = {"text", "text_pinyin", "audio"} - set(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        source_rows += parquet.metadata.num_rows
    if not source_rows:
        raise ValueError("The source corpus has no rows")

    preprocessing = config["preprocessing"]
    sample_rate = preprocessing["audio"]["sampling_rate"]
    if type(sample_rate) is not int or sample_rate <= 0:
        raise ValueError("sampling_rate must be a positive integer")
    stft = None
    if mel_samples:
        from preprocessing.audio import TacotronSTFT, get_mel_from_wav

        parameters = preprocessing["stft"]
        mel = preprocessing["mel"]
        stft = TacotronSTFT(
            parameters["filter_length"], parameters["hop_length"],
            parameters["win_length"], mel["n_mel_channels"], sample_rate,
            mel["mel_fmin"], mel["mel_fmax"],
        )

    wav_dir = output / "raw" / "Baker"
    wav_dir.mkdir(parents=True, exist_ok=True)
    if mel_samples:
        (output / "mel").mkdir()
    count = 0
    total_samples = 0
    source_rates = set()
    with (output / "manifest.jsonl").open("w", encoding="utf-8") as manifest:
        for path, index, row in shard_rows(paths):
            name = f"{path.stem}-{index:05d}"
            waveform, source_rate, channels = decode_record(row, name, sample_rate)
            source_rates.add(source_rate)
            wav_path = wav_dir / f"{name}.wav"
            lab_path = wav_dir / f"{name}.lab"
            sf.write(wav_path, waveform, sample_rate, subtype="PCM_16")
            text = re.sub(r"#[1-4]", "", row["text"]).strip()
            if not text:
                raise ValueError(f"{name}: transcript is empty after removing prosody marks")
            lab_path.write_text(text + "\n", encoding="utf-8")
            record = {
                "id": name,
                "speaker": "Baker",
                "source_shard": str(path),
                "source_row": index,
                "source_file_name": row.get("file_name"),
                "source_audio_path": row["audio"].get("path"),
                "source_sampling_rate": source_rate,
                "source_channels": channels,
                "text": text,
                "text_original": row["text"],
                "text_pinyin": row["text_pinyin"],
                "wav": str(wav_path.relative_to(output)),
                "lab": str(lab_path.relative_to(output)),
                "sampling_rate": sample_rate,
                "samples": len(waveform),
            }
            if count < mel_samples:
                # Extract from the saved PCM, exactly as later feature preparation will.
                pcm, _ = sf.read(wav_path, dtype="float32")
                mel, _ = get_mel_from_wav(pcm, stft)
                mel_path = output / "mel" / f"Baker-mel-{name}.npy"
                np.save(mel_path, mel.T)
                record["mel"] = str(mel_path.relative_to(output))
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
            total_samples += len(waveform)
            count += 1
            if count % 100 == 0:
                print(f"Prepared {count} recordings", flush=True)
            if limit is not None and count >= limit:
                break

    summary = {
        "source_corpus": str(corpus),
        "source_shards": len(paths),
        "source_rows": source_rows,
        "prepared_rows": count,
        "is_subset": count < source_rows,
        "hours": total_samples / sample_rate / 3600,
        "source_sampling_rates": sorted(source_rates),
        "sampling_rate": sample_rate,
        "peak_normalization": 0.95,
        "reference_mels": min(mel_samples, count),
        "mel_preprocessing": {
            "stft": preprocessing["stft"],
            "mel": preprocessing["mel"],
            "compression": "natural_log_clamp_1e-5",
        },
        "alignment_status": "required; no duration labels generated",
        "training_ready": False,
    }
    (output / "dataset.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )
    print(f"Prepared {count}/{source_rows} recordings at {sample_rate} Hz: {output}")
    print("Chinese phoneme alignment and frontend/model adaptation are still required.")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-dir", type=Path, default=Path.home() / "datasets/baker/data",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/Baker")
    parser.add_argument(
        "--preprocess-config", type=Path, default=ROOT / "configs/LJSpeech/preprocess.yaml",
        help="Audio/Mel settings only; does not enable the English training pipeline",
    )
    parser.add_argument("--limit", type=int, help="Prepare a subset instead of the full corpus")
    parser.add_argument("--mel-samples", type=int, default=0, help="Reference Mels for vocoder listening")
    args = parser.parse_args()
    with args.preprocess_config.expanduser().open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    prepare(args.corpus_dir, args.output_dir, config, args.limit, args.mel_samples)


if __name__ == "__main__":
    main()
