"""English text/ARPAbet or Chinese numbered pinyin -> two TFLite models -> PCM16 WAV.

Defaults to the corrected A16W8 models. Neither PyTorch nor the original
checkpoints/vocoder configuration are needed for synthesis.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import tensorflow as tf

from infer_hifigan_tflite import ROOT, load_vocoder, synthesize

sys.path.insert(0, str(ROOT / "grainspeech"))
from text import text_to_sequence
from text.frontend import parse_phonemes, text_to_arpabet
from text.vocabulary import PhoneVocabulary, pinyin_to_phones


class AcousticModel:
    def __init__(self, path, num_threads=1, metadata_path=None):
        path = Path(path)
        model = path.read_bytes()
        explicit_metadata = metadata_path is not None
        metadata_path = Path(metadata_path) if metadata_path is not None else path.with_suffix(".json")
        self.metadata = {}
        if metadata_path.exists():
            self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            fingerprint = self.metadata.get("tflite_sha256")
            if fingerprint and hashlib.sha256(model).hexdigest() != fingerprint:
                raise ValueError("Acoustic TFLite/metadata hash mismatch")
        elif explicit_metadata:
            raise FileNotFoundError(metadata_path)
        self.interpreter = tf.lite.Interpreter(model_content=model, num_threads=num_threads)
        self.interpreter.allocate_tensors()
        inputs = self.interpreter.get_input_details()
        outputs = self.interpreter.get_output_details()
        if len(inputs) != 1 or len(outputs) != 2:
            raise ValueError("Expected one phoneme input and two acoustic outputs")
        self.input = inputs[0]
        shape = list(self.input["shape"])
        if len(shape) != 2 or shape[0] != 1 or shape[1] <= 0:
            raise ValueError("Expected a fixed [1, max_phonemes] input")
        if any(detail["dtype"] != np.float32 for detail in inputs + outputs):
            raise ValueError(
                "Expected float32 interfaces; re-export the corrected acoustic model "
                "instead of using the initial int16-output export"
            )
        lengths = [detail for detail in outputs if list(detail["shape"]) == [1]]
        mels = [
            detail for detail in outputs
            if len(detail["shape"]) == 3 and detail["shape"][0] == 1
            and detail["shape"][1] > 0 and detail["shape"][2] == 80
        ]
        if len(lengths) != 1 or len(mels) != 1:
            raise ValueError("Expected Mel [1, max_mel_frames, 80] and length [1]")
        self.length_output = lengths[0]
        self.mel_output = mels[0]
        self.max_phonemes = int(shape[1])
        self.max_mel_frames = int(self.mel_output["shape"][1])
        details = {detail["index"]: detail for detail in self.interpreter.get_tensor_details()}
        lookups = [
            op for op in self.interpreter._get_ops_details()
            if op["op_name"] == "EMBEDDING_LOOKUP"
        ]
        if len(lookups) != 1:
            raise ValueError("Expected one token embedding lookup in the acoustic model")
        self.vocab_size = int(details[lookups[0]["inputs"][1]]["shape"][0])
        self.frontend = self.metadata.get("frontend", {"language": "en"})
        self.language = self.frontend.get("language", "en")
        self.vocabulary = (
            PhoneVocabulary(self.frontend["symbols"]) if "symbols" in self.frontend else None
        )
        if self.vocabulary is not None:
            if not self.metadata.get("tflite_sha256"):
                raise ValueError("Custom vocabulary requires hash-bound acoustic metadata")
            if len(self.vocabulary) != self.vocab_size:
                raise ValueError("Frontend vocabulary and TFLite embedding dimensions disagree")
        elif self.vocab_size != 74 or self.language != "en":
            raise ValueError("Non-English acoustic models require their exported frontend JSON")
        if self.metadata:
            if (
                self.metadata["input"]["phoneme"] != list(self.input["shape"])
                or self.metadata["output"]["mel"] != list(self.mel_output["shape"])
                or self.metadata["input"].get("vocab_size", 74) != self.vocab_size
            ):
                raise ValueError("Acoustic metadata dimensions do not match the model")

    def predict(self, sequence, allow_truncation=False):
        if not sequence:
            raise ValueError("No phoneme IDs to synthesize")
        if any(type(value) is not int or not 1 <= value < self.vocab_size for value in sequence):
            raise ValueError(f"Expected integer phoneme IDs in [1, {self.vocab_size - 1}]")
        if len(sequence) > self.max_phonemes:
            raise ValueError(
                f"Input has {len(sequence)} phonemes; model capacity is {self.max_phonemes}. "
                "Use shorter text or re-export with a larger --max-phonemes."
            )
        phoneme = np.zeros((1, self.max_phonemes), dtype=np.float32)
        phoneme[0, :len(sequence)] = sequence
        self.interpreter.set_tensor(self.input["index"], phoneme)
        self.interpreter.invoke()
        raw_length = float(self.interpreter.get_tensor(self.length_output["index"])[0])
        mel = self.interpreter.get_tensor(self.mel_output["index"])[0]
        if not np.isfinite(raw_length) or not np.isfinite(mel).all():
            raise RuntimeError("Acoustic model produced non-finite output")
        length = int(np.rint(raw_length))
        if abs(raw_length - length) > 0.01 or not 1 <= length <= self.max_mel_frames:
            raise RuntimeError(f"Invalid predicted Mel length: {raw_length}")
        if length == self.max_mel_frames:
            message = (
                f"Mel output reached capacity ({length} frames) and may be truncated. "
                "Use shorter text, re-export with a larger --max-mel-frames, "
                "or explicitly pass --allow-truncation."
            )
            if not allow_truncation:
                raise ValueError(message)
            warnings.warn(message, stacklevel=2)
        return mel[:length].copy()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="English text to synthesize")
    source.add_argument("--phonemes", help="Space-separated ARPAbet, optionally in braces")
    source.add_argument("--pinyin", help="Chinese numbered pinyin; requires a Chinese acoustic model")
    parser.add_argument(
        "--acoustic-model", type=Path, default=ROOT / "outputs/grainspeech_a16w8.tflite",
    )
    parser.add_argument(
        "--vocoder-model", type=Path, default=ROOT / "outputs/hifigan_a16w8.tflite",
    )
    parser.add_argument(
        "--acoustic-metadata", type=Path, help="Default: JSON adjacent to acoustic model",
    )
    parser.add_argument(
        "--vocoder-metadata", type=Path, help="Default: JSON adjacent to vocoder model",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/grainspeech_a16w8.wav")
    parser.add_argument("--save-mel", type=Path, help="Optionally save the valid Mel as .npy")
    parser.add_argument("--threads", type=int, default=1, help="Threads per TFLite interpreter")
    parser.add_argument(
        "--allow-truncation", action="store_true",
        help="Allow audio when the predicted Mel length reaches the model's capacity",
    )
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("--threads must be positive")
    if args.output.suffix.lower() != ".wav":
        parser.error("--output must end in .wav")
    if args.save_mel and args.save_mel.suffix.lower() != ".npy":
        parser.error("--save-mel must end in .npy")
    return args


def main():
    args = parse_args()
    acoustic = AcousticModel(
        args.acoustic_model, num_threads=args.threads, metadata_path=args.acoustic_metadata,
    )
    if args.pinyin is not None:
        if acoustic.language != "zh" or acoustic.vocabulary is None:
            raise ValueError("--pinyin requires an exported Chinese acoustic model and matching JSON")
        phones = pinyin_to_phones(args.pinyin, acoustic.frontend["lexicon"])
        sequence = acoustic.vocabulary.encode(phones)
    elif acoustic.language != "en":
        raise ValueError("This acoustic model requires --pinyin, not the English text frontend")
    elif args.text is not None:
        if not args.text.strip():
            raise ValueError("--text must not be empty")
        phones = text_to_arpabet(args.text)
    else:
        phones = parse_phonemes(args.phonemes)
    arpabet = "{" + " ".join(phones) + "}"
    if args.pinyin is None:
        sequence = (
            acoustic.vocabulary.encode(phones) if acoustic.vocabulary is not None
            else text_to_sequence(arpabet, ["english_cleaners"])
        )
    print(f"Phonemes ({len(sequence)}): {arpabet}", flush=True)

    mel = acoustic.predict(sequence, allow_truncation=args.allow_truncation)
    print(f"GrainSpeech TFLite: {len(mel)} Mel frames", flush=True)
    vocoder, metadata = load_vocoder(
        args.vocoder_model, args.vocoder_metadata, num_threads=args.threads,
    )
    audio = acoustic.metadata.get("audio")
    if audio is not None and any(
        audio[key] != metadata[key] for key in ("sampling_rate", "hop_length")
    ):
        raise ValueError("Acoustic and vocoder audio settings disagree")
    waveform = synthesize(mel, vocoder, metadata)
    expected_samples = len(mel) * metadata["hop_length"]
    if waveform.shape != (expected_samples,) or not np.isfinite(waveform).all():
        raise RuntimeError("Vocoder returned an invalid waveform")
    from scipy.io import wavfile

    pcm = np.clip(waveform * 32768.0, -32768, 32767).astype(np.int16)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(args.output, metadata["sampling_rate"], pcm)
    if args.save_mel:
        args.save_mel.parent.mkdir(parents=True, exist_ok=True)
        with args.save_mel.open("wb") as stream:
            np.save(stream, mel)
        print(f"Wrote {args.save_mel}", flush=True)
    print(
        f"Wrote {args.output} "
        f"({len(pcm) / metadata['sampling_rate']:.3f} s, {metadata['sampling_rate']} Hz)",
        flush=True,
    )


if __name__ == "__main__":
    main()
