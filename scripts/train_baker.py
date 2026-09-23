"""Train Chinese GrainSpeech on aligned tone-marked pinyin syllables, without a vocoder.

The Baker frontend supplies CTC-derived syllable durations, not individual-phone
alignments. Token IDs and syllable expansion are defined by the embedded config.
"""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "grainspeech"))
sys.path.insert(0, str(ROOT))

import torch
import yaml
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from datamodule import LJSpeechDataModule
from model_l1_ssim_gvar import GrainSpeech
from text.vocabulary import normalize_pinyin, pinyin_to_phones, vocabulary_from_config


def validate_config(config):
    text = config["preprocessing"]["text"]
    if text.get("language") != "zh":
        raise ValueError("Baker training requires preprocessing.text.language: zh")
    vocabulary = vocabulary_from_config(config)
    lexicon = text.get("lexicon")
    if not isinstance(lexicon, dict) or not lexicon:
        raise ValueError("Chinese training requires an embedded text.lexicon")
    for syllable in lexicon:
        if normalize_pinyin(syllable) != syllable:
            raise ValueError(f"Lexicon key is not canonical numbered pinyin: {syllable!r}")
        vocabulary.encode(pinyin_to_phones(syllable, lexicon))
    if config["preprocessing"]["mel"]["n_mel_channels"] != 80:
        raise ValueError("GrainSpeech requires 80 mel channels")
    for feature in ("pitch", "energy"):
        if config["preprocessing"][feature]["feature"] != "phoneme_level":
            raise ValueError(
                f"GrainSpeech requires token-level {feature} "
                "(set feature: phoneme_level for tone-marked pinyin syllables)"
            )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preprocess-config", type=Path,
                        default=Path("data/Baker-features/preprocess.yaml"))
    parser.add_argument("--run-name", default="baker")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--accelerator", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--precision", default="32-true")
    parser.add_argument("--disable-cudnn", action="store_true",
                        help="Use native PyTorch CUDA kernels if installed cuDNN libraries conflict")
    parser.add_argument("--checkpoint", type=Path,
                        help="Resume a trusted Chinese full training checkpoint")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    if args.batch_size < 1 or args.epochs < 1 or args.workers < 0:
        parser.error("batch-size/epochs must be positive and workers nonnegative")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("max-steps must be positive")
    if not args.run_name or Path(args.run_name).name != args.run_name:
        parser.error("run-name must be a single directory name")
    return args


def main():
    args = parse_args()
    with args.preprocess_config.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    validate_config(config)
    seed_everything(args.seed, workers=True)
    data = LJSpeechDataModule(config, batch_size=args.batch_size, num_workers=args.workers)
    data.setup("fit")
    if not len(data.train_dataset) or not len(data.test_dataset):
        raise ValueError("Both train.txt and val.txt must contain usable examples")
    checkpoint_dir = Path("lightning_logs") / args.run_name / "checkpoints"
    if args.checkpoint is None and checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        raise FileExistsError("Run already has checkpoints; use --checkpoint to resume or a new --run-name")

    if args.checkpoint is None:
        model = GrainSpeech(config, max_epochs=args.epochs, hifigan_checkpoint=None)
        print("Training Chinese acoustic model from random initialization (not English weights)")
    else:
        # Lightning full checkpoints contain optimizer/config objects: trust the source.
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        previous = checkpoint["hyper_parameters"]["preprocess_config"]
        validate_config(previous)
        if previous["preprocessing"] != config["preprocessing"]:
            raise ValueError("Cannot change vocabulary, lexicon, or preprocessing when resuming")
        stats_path = Path(config["path"]["preprocessed_path"]) / "stats.json"
        if json.loads(stats_path.read_text()) != checkpoint["hyper_parameters"]["feature_stats"]:
            raise ValueError("Cannot change feature normalization statistics when resuming")
        if checkpoint["hyper_parameters"].get("hifigan_checkpoint") is not None:
            raise ValueError("Resume requires an acoustic-only Chinese training checkpoint")
        model = GrainSpeech.load_from_checkpoint(
            args.checkpoint, map_location="cpu", weights_only=False,
            preprocess_config=config, hifigan_checkpoint=None,
        )
        print(f"Resuming full training state: {args.checkpoint}")

    model.hparams["training_config"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    # Each invocation gets a new metrics directory; resuming must not erase old CSV logs.
    logger = CSVLogger("lightning_logs", name=args.run_name)
    callback = ModelCheckpoint(
        dirpath=checkpoint_dir, filename="{epoch}-{step}-{val_loss:.4f}",
        monitor="val_loss", mode="min", save_top_k=1, save_last=True,
    )
    trainer = Trainer(
        accelerator=args.accelerator, devices=1, precision=args.precision,
        max_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps is not None else -1,
        logger=logger, callbacks=[callback], log_every_n_steps=1,
    )
    backend = torch.backends.cudnn.flags(enabled=False) if args.disable_cudnn else nullcontext()
    with backend:
        trainer.fit(model, datamodule=data,
                    ckpt_path=str(args.checkpoint) if args.checkpoint is not None else None)
    # A step-limited smoke may finish before validation/checkpoint callbacks fire.
    trainer.save_checkpoint(checkpoint_dir / "last.ckpt")
    print(f"Saved resumable acoustic checkpoint: {checkpoint_dir / 'last.ckpt'}")


if __name__ == "__main__":
    main()
