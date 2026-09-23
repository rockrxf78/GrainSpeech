"""Synthetic Chinese/English model regressions; no corpus or vocoder needed."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "grainspeech"))
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml

from datamodule import LJSpeechDataModule, LJSpeechDataset
from layers.networks import Encoder, PhonemeEncoder
from model_l1_ssim_gvar import GrainSpeech
from text import text_to_sequence
from text.symbols import symbols
from text.vocabulary import PhoneVocabulary, normalize_pinyin, pinyin_to_phones


def expect_error(function, fragment):
    try:
        function()
    except (ValueError, RuntimeError) as error:
        assert fragment in str(error), str(error)
    else:
        raise AssertionError(f"Expected an error containing {fragment!r}")


def fixture(directory):
    with (ROOT / "configs/LJSpeech/preprocess.yaml").open() as stream:
        config = yaml.safe_load(stream)
    config["dataset"] = "Baker"
    config["path"]["preprocessed_path"] = str(directory)
    syllables = ["ni3", "hao3", "ka2", "er2", "er3", "menr2", "lv5"]
    config["preprocessing"]["text"].update({
        "language": "zh", "text_cleaners": [], "unit": "tone_marked_pinyin_syllable",
        "symbols": ["_", "sp", "sil"] + syllables,
        "lexicon": {syllable: [syllable] for syllable in syllables},
    })
    for name in ("mel", "pitch", "energy", "duration"):
        (directory / name).mkdir()
    (directory / "stats.json").write_text(
        json.dumps({"pitch": [-2, 2, 100, 20], "energy": [-2, 2, 1, 1]}))
    (directory / "speakers.json").write_text('{"Baker": 0}')
    records = [("one", ["menr2"], [1]), ("two", ["ni3", "hao3"], [2, 1])]
    generator = np.random.default_rng(17)
    for name, phones, durations in records:
        for feature, values in {
            "mel": generator.normal(size=(sum(durations), 80)).astype(np.float32),
            "pitch": np.zeros(len(phones), dtype=np.float32),
            "energy": np.ones(len(phones), dtype=np.float32),
            "duration": np.array(durations, dtype=np.int64),
        }.items():
            np.save(directory / feature / f"Baker-{feature}-{name}.npy", values)
    lines = "".join(f"{name}|Baker|{{{' '.join(phones)}}}|你好\n"
                    for name, phones, _ in records)
    for split in ("train", "val"):
        (directory / f"{split}.txt").write_text(lines, encoding="utf-8")
    (directory / "preprocess.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    return config


def main():
    torch.set_num_threads(1)
    assert normalize_pinyin(" LÜ0 ") == "lv5"
    assert normalize_pinyin("lu:4") == "lv4"
    for syllable in ("ka2", "er2", "er3", "menr2"):
        assert normalize_pinyin(syllable) == syllable
        assert pinyin_to_phones(syllable, {syllable: [syllable]}) == [syllable]
    expect_error(lambda: normalize_pinyin("ni"), "numbered pinyin")
    expect_error(lambda: pinyin_to_phones("er2", {}), "Unknown pinyin")
    assert pinyin_to_phones("ni3 sp hao3", {
        "ni3": ["n", "i3"], "hao3": ["x", "ɑu3"],
    }) == ["n", "i3", "sp", "x", "ɑu3"]
    assert pinyin_to_phones("sp", {}) == ["sp"]
    syllable_lexicon = {"ni3": ["ni3"], "hao3": ["hao3"]}
    syllable_vocab = PhoneVocabulary(["_", "sp", "sil", "ni3", "hao3"])
    assert syllable_vocab.encode(
        pinyin_to_phones("ni3 sp hao3", syllable_lexicon)
    ) == [3, 1, 4]
    expect_error(lambda: pinyin_to_phones("sil", {}), "numbered pinyin")
    expect_error(lambda: PhoneVocabulary(["n", "_"]), "index 0")
    expect_error(lambda: PhoneVocabulary(["_", "n", "n"]), "duplicate")
    vocabulary = PhoneVocabulary(["_", "n", "i3"])
    assert vocabulary.encode("{n i3}") == [1, 2]
    expect_error(lambda: vocabulary.encode("{unknown}"), "Unknown")
    expect_error(lambda: vocabulary.encode("_"), "padding")
    assert Encoder().embed.num_embeddings == len(symbols) + 1
    assert text_to_sequence("{HH AH0}", []) == [
        symbols.index("@HH"), symbols.index("@AH0")]
    assert PhonemeEncoder(vocab_size=3).encoder.embed.num_embeddings == 3

    directory = ROOT / "outputs" / f"chinese-model-check-{uuid.uuid4().hex}"
    directory.mkdir(parents=True)
    run_name = directory.name
    log_directory = ROOT / "lightning_logs" / run_name
    try:
        config = fixture(directory)
        dataset = LJSpeechDataset("train.txt", config)
        data = LJSpeechDataModule(config, batch_size=2, num_workers=0)
        with patch("model_l1_ssim_gvar.get_hifigan",
                   side_effect=AssertionError("Acoustic-only mode loaded a vocoder")):
            model = GrainSpeech(config, hifigan_checkpoint=None)
        assert model.phoneme2mel.encoder.encoder.embed.num_embeddings == len(
            config["preprocessing"]["text"]["symbols"])
        for indexes in ([0], [1], [0, 1]):
            x, y = data.collate_fn([dataset[index] for index in indexes])
            model.train()
            prediction = model(x)
            assert prediction["mel"].shape == y["mel"].shape
            assert prediction["pitch"].shape == (*x["phoneme"].shape, 1)
            losses = model.loss(prediction, y, x)
            assert all(torch.isfinite(loss) for loss in losses)
            sum(losses).backward()
            model.zero_grad(set_to_none=True)
            model.eval()
            with torch.no_grad():
                mel, lengths, _ = model.phoneme2mel(x)
            assert mel.shape[0] == len(indexes) and (lengths > 0).all()
            for index, length in enumerate(lengths):
                assert not mel[index, int(length):].count_nonzero()
            expect_error(lambda: model.predict_step(x), "requires a vocoder")
        # Padding must be respected even in a single-utterance batch.
        single = {"phoneme": torch.tensor([[1, 0]]),
                  "phoneme_mask": torch.tensor([[False, True]])}
        with torch.no_grad():
            mel, lengths, durations = model.phoneme2mel(single)
        assert int(lengths[0]) == int(durations[0, 0, 0].round())
        if torch.cuda.is_available():
            gpu_batch = {name: value.cuda() for name, value in single.items()}
            with torch.no_grad():
                model.cuda().phoneme2mel(gpu_batch)
            gpu_x = {name: value.cuda() if isinstance(value, torch.Tensor) else value
                     for name, value in x.items()}
            gpu_y = {"mel": y["mel"].cuda()}
            model.train()
            gpu_losses = model.loss(model(gpu_x), gpu_y, gpu_x)
            assert all(torch.isfinite(loss) for loss in gpu_losses)
            sum(gpu_losses).backward()
            model.zero_grad(set_to_none=True)
            model.cpu()
        bad_path = directory / "duration/Baker-duration-one.npy"
        np.save(bad_path, np.array([2]))
        expect_error(lambda: dataset[0], "duration sum")
        np.save(bad_path, np.array([1]))
        pitch_path = directory / "pitch/Baker-pitch-one.npy"
        np.save(pitch_path, np.array([np.nan]))
        expect_error(lambda: dataset[0], "non-finite")
        np.save(pitch_path, np.array([0, 0], dtype=np.float32))
        expect_error(lambda: dataset[0], "lengths differ")
        np.save(pitch_path, np.array([0], dtype=np.float32))
        original_text = dataset.text[0]
        dataset.text[0] = "{not-a-phone}"
        expect_error(lambda: dataset[0], "Unknown")
        dataset.text[0] = original_text

        environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        train = [
            sys.executable, "scripts/train_baker.py",
            "--preprocess-config", str(directory / "preprocess.yaml"),
            "--run-name", run_name, "--batch-size", "1", "--workers", "0",
            "--epochs", "2", "--accelerator", "cpu", "--max-steps", "1",
        ]
        subprocess.run(train, cwd=ROOT, env=environment, check=True)
        checkpoint = log_directory / "checkpoints/last.ckpt"
        assert checkpoint.is_file()
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert saved["global_step"] == 1
        assert saved["hyper_parameters"]["preprocess_config"] == config
        assert saved["hyper_parameters"]["training_config"]["seed"] == 1234
        previous_metrics = {
            path: path.read_bytes() for path in log_directory.glob("version_*/metrics.csv")
        }
        assert previous_metrics
        duplicate = subprocess.run(train, cwd=ROOT, env=environment,
                                   capture_output=True, text=True)
        assert duplicate.returncode != 0 and "Run already has checkpoints" in duplicate.stderr
        train[train.index("--max-steps") + 1] = "2"
        subprocess.run(train + ["--checkpoint", str(checkpoint)],
                       cwd=ROOT, env=environment, check=True)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert saved["global_step"] == 2
        assert all(path.read_bytes() == content for path, content in previous_metrics.items())
        assert any(path.name != "last.ckpt"
                   for path in checkpoint.parent.glob("*.ckpt"))
        config_path = directory / "preprocess.yaml"
        changed = yaml.safe_load(config_path.read_text())
        changed["preprocessing"]["text"]["symbols"][3:5] = ["hao3", "ni3"]
        config_path.write_text(yaml.safe_dump(changed, allow_unicode=True))
        rejected = subprocess.run(
            train + ["--checkpoint", str(checkpoint)], cwd=ROOT, env=environment,
            capture_output=True, text=True,
        )
        assert rejected.returncode != 0 and "Cannot change vocabulary" in rejected.stderr
        config_path.write_text(yaml.safe_dump(config, allow_unicode=True))
        stats_path = directory / "stats.json"
        previous_stats = stats_path.read_text()
        altered_stats = json.loads(previous_stats)
        altered_stats["pitch"][2] += 1
        stats_path.write_text(json.dumps(altered_stats))
        rejected = subprocess.run(train + ["--checkpoint", str(checkpoint)],
                                  cwd=ROOT, env=environment, capture_output=True, text=True)
        assert rejected.returncode != 0 and "normalization statistics" in rejected.stderr
        stats_path.write_text(previous_stats)
        validation_path = directory / "val.txt"
        validation_rows = validation_path.read_text()
        validation_path.write_text("")
        rejected = subprocess.run(train, cwd=ROOT, env=environment,
                                  capture_output=True, text=True)
        assert rejected.returncode != 0 and "Both train.txt and val.txt" in rejected.stderr
        validation_path.write_text(validation_rows)
        # Checkpoint inference must not depend on the original feature directory.
        (directory / "stats.json").unlink()
        output = directory / "predicted.npy"
        subprocess.run([
            sys.executable, "scripts/infer_baker.py", "--checkpoint", str(checkpoint),
            "--pinyin", "ni3 hao3", "--output", str(output), "--device", "cpu",
        ], cwd=ROOT, env=environment, check=True)
        mel = np.load(output)
        assert mel.ndim == 2 and mel.shape[1] == 80 and np.isfinite(mel).all()
        print("Chinese synthetic training/resume/inference and English vocabulary regressions passed.")
    finally:
        shutil.rmtree(directory)
        if log_directory.exists():
            shutil.rmtree(log_directory)


if __name__ == "__main__":
    main()
