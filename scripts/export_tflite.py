"""Export the GrainSpeech acoustic model to a fixed-shape A16W8 TFLite model.

The exported model contains the phoneme-to-Mel acoustic model only. HiFi-GAN
is deliberately not included because the paper's MCU result measures Mel
generation and the vocoder needs a separate deployment strategy.

The export uses TensorFlow directly instead of tracing the PyTorch inference
wrapper. The PyTorch model uses dynamic repeat_interleave, which is not a
portable TFLite/Ethos-U graph. This implementation replaces it with a
fixed-size cumulative-duration gather.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import tensorflow as _tensorflow
except ImportError:
    _tensorflow = None


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


DEFAULT_CHECKPOINT = REPOSITORY_ROOT / "checkpoints" / "grainspeech_l1_ssim_gvar.ckpt"
DEFAULT_STATS = REPOSITORY_ROOT / "configs" / "LJSpeech" / "stats.json"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "outputs" / "grainspeech_a16w8.tflite"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--stats", type=Path,
                        help="Default: checkpoint statistics, or released English statistics")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--max-phonemes",
        type=int,
        default=128,
        help="Static phoneme capacity. Inputs longer than this are rejected.",
    )
    parser.add_argument(
        "--max-mel-frames",
        type=int,
        default=512,
        help="Static Mel capacity. Longer predictions are clipped.",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=64,
        help="Number of representative inputs used for A16W8 calibration.",
    )
    parser.add_argument(
        "--calibration-data", type=Path,
        help="Training metadata (id|speaker|{tokens}|text); required for custom-vocabulary A16W8",
    )
    parser.add_argument(
        "--float32",
        action="store_true",
        help="Export an FP32 TFLite model instead of applying A16W8 quantization.",
    )
    return parser.parse_args()


def _require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _load_checkpoint(checkpoint: Path) -> dict[str, object]:
    import torch

    try:
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        loaded = torch.load(checkpoint, map_location="cpu")
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected a checkpoint dictionary, got {type(loaded)!r}")
    return loaded


def _state_value(state: dict[str, object], name: str) -> np.ndarray:
    candidates = (
        f"phoneme2mel.{name}",
        f"model.phoneme2mel.{name}",
    )
    for key in candidates:
        value = state.get(key)
        if value is not None:
            return value.detach().cpu().numpy().copy()
    raise KeyError(f"Missing checkpoint tensor; tried: {', '.join(candidates)}")


def _load_bins(
    state: dict[str, object],
    name: str,
    stats_name: str,
    stats: dict[str, list[float]],
) -> np.ndarray:
    try:
        return _state_value(state, name)
    except KeyError:
        minimum, maximum = stats[stats_name][:2]
        return np.linspace(minimum, maximum, 31, dtype=np.float32)


_TensorFlowModule = _tensorflow.Module if _tensorflow is not None else object


class StaticGrainSpeech(_TensorFlowModule):
    """TensorFlow implementation of the fixed-shape acoustic model."""

    def __init__(
        self,
        state: dict[str, object],
        stats: dict[str, list[float]],
        max_phonemes: int,
        max_mel_frames: int,
    ) -> None:
        import tensorflow as tf

        if _tensorflow is not None:
            super().__init__()
        self.tf = tf
        self.max_phonemes = max_phonemes
        self.max_mel_frames = max_mel_frames
        self._variables: dict[str, object] = {}

        def add(name: str, value: np.ndarray) -> object:
            variable = tf.Variable(
                value.astype(np.float32, copy=False),
                trainable=False,
                name=name.replace(".", "_"),
            )
            self._variables[name] = variable
            setattr(self, f"v_{name.replace('.', '_')}", variable)
            return variable

        self._add = add
        self._add("encoder.encoder.embed.weight", _state_value(state, "encoder.encoder.embed.weight"))
        for block in ("b1", "b2"):
            for layer in ("conv1", "conv2"):
                self._add(
                    f"encoder.encoder.{block}_{layer}.weight",
                    _state_value(state, f"encoder.encoder.{block}_{layer}.weight"),
                )
                self._add(
                    f"encoder.encoder.{block}_{layer}.bias",
                    _state_value(state, f"encoder.encoder.{block}_{layer}.bias"),
                )
            for layer in ("dyt1", "dyt2"):
                for parameter in ("alpha", "weight", "bias"):
                    self._add(
                        f"encoder.encoder.{block}_{layer}.{parameter}",
                        _state_value(
                            state, f"encoder.encoder.{block}_{layer}.{parameter}"
                        ),
                    )
        for parameter in ("weight", "bias"):
            self._add(
                f"encoder.encoder.linear.{parameter}",
                _state_value(state, f"encoder.encoder.linear.{parameter}"),
            )

        for decoder in ("pitch_decoder", "energy_decoder", "duration_decoder"):
            for layer in ("conv1", "conv2"):
                for parameter in ("weight", "bias"):
                    self._add(
                        f"encoder.{decoder}.{layer}.{parameter}",
                        _state_value(state, f"encoder.{decoder}.{layer}.{parameter}"),
                    )
            for layer in ("dyt1", "dyt2"):
                for parameter in ("alpha", "weight", "bias"):
                    self._add(
                        f"encoder.{decoder}.{layer}.{parameter}",
                        _state_value(state, f"encoder.{decoder}.{layer}.{parameter}"),
                    )
            for parameter in ("weight", "bias"):
                self._add(
                    f"encoder.{decoder}.linear.{parameter}",
                    _state_value(state, f"encoder.{decoder}.linear.{parameter}"),
                )

        for decoder, stats_name in (
            ("pitch_decoder", "pitch"),
            ("energy_decoder", "energy"),
        ):
            self._add(
                f"encoder.{decoder}.bins",
                _load_bins(
                    state,
                    f"encoder.{decoder}.{stats_name}_bins",
                    stats_name,
                    stats,
                ),
            )
            self._add(
                f"encoder.{decoder}.embedding.weight",
                _state_value(state, f"encoder.{decoder}.{stats_name}_embedding.weight"),
            )
        self._add(
            "encoder.duration_decoder.bins",
            _load_bins(
                state,
                "encoder.duration_decoder.duration_bins",
                "duration",
                {"duration": [2.0, 34.0]},
            ),
        )
        self._add(
            "encoder.duration_decoder.embedding.weight",
            _state_value(state, "encoder.duration_decoder.duration_embedding.weight"),
        )
        for parameter in ("weight", "bias"):
            self._add(
                f"encoder.fusion_linear.{parameter}",
                _state_value(state, f"encoder.fusion_linear.{parameter}"),
            )

        for layer in (
            "proj",
            "block1_conv1_dw",
            "block1_conv1_pw",
            "block1_conv2_dw",
            "block1_conv2_pw",
            "block2_conv1_dw",
            "block2_conv1_pw",
            "block2_conv2_dw",
            "block2_conv2_pw",
            "block3_conv1_dw",
            "block3_conv1_pw",
            "block3_conv2_dw",
            "block3_conv2_pw",
        ):
            for parameter in ("weight", "bias"):
                self._add(
                    f"decoder.{layer}.{parameter}",
                    _state_value(state, f"decoder.{layer}.{parameter}"),
                )
        for layer in (
            "block1_dyt1",
            "block1_dyt2",
            "block2_dyt1",
            "block2_dyt2",
            "block3_dyt1",
            "block3_dyt2",
            "mel_dyt",
        ):
            for parameter in ("alpha", "weight", "bias"):
                self._add(
                    f"decoder.{layer}.{parameter}",
                    _state_value(state, f"decoder.{layer}.{parameter}"),
                )
        for layer in ("mel_linear_up", "mel_linear_down"):
            for parameter in ("weight", "bias"):
                self._add(
                    f"decoder.{layer}.{parameter}",
                    _state_value(state, f"decoder.{layer}.{parameter}"),
                )

    def _v(self, name: str) -> object:
        return self._variables[name]

    def _linear(self, x: object, name: str) -> object:
        tf = self.tf
        return tf.linalg.matmul(x, self._v(f"{name}.weight"), transpose_b=True) + self._v(
            f"{name}.bias"
        )

    def _conv1d(self, x: object, name: str, dilation: int = 1) -> object:
        tf = self.tf
        weight = self._v(f"{name}.weight")
        bias = self._v(f"{name}.bias")
        filters = tf.transpose(weight, (2, 1, 0))
        return tf.nn.conv1d(
            x,
            filters,
            stride=1,
            padding="SAME",
            dilations=dilation,
        ) + bias

    def _depthwise(self, x: object, name: str, dilation: int = 1) -> object:
        tf = self.tf
        weight = self._v(f"{name}.weight")
        filters = tf.transpose(weight, (2, 1, 0))
        filters = tf.reshape(
            filters, (1, tf.shape(filters)[0], tf.shape(filters)[2], 1)
        )
        x = tf.expand_dims(x, axis=1)
        x = tf.nn.depthwise_conv2d(
            x,
            filters,
            strides=(1, 1, 1, 1),
            padding="SAME",
            dilations=(1, dilation),
        )
        return tf.squeeze(x, axis=1) + self._v(f"{name}.bias")

    def _dyt(self, x: object, name: str) -> object:
        tf = self.tf
        return tf.tanh(self._v(f"{name}.alpha") * x) * self._v(
            f"{name}.weight"
        ) + self._v(f"{name}.bias")

    def _encoder(self, phoneme: object, valid: object) -> object:
        tf = self.tf
        # Integer lookup must not compare dequantized IDs to exact float values.
        lookup = tf.one_hot(
            tf.cast(tf.round(phoneme), tf.int32),
            self._v("encoder.encoder.embed.weight").shape[0],
            dtype=tf.float32,
        )
        x = tf.linalg.matmul(
            lookup,
            self._v("encoder.encoder.embed.weight"),
        )
        skip = x
        x = self._dyt(
            self._conv1d(x, "encoder.encoder.b1_conv1"),
            "encoder.encoder.b1_dyt1",
        ) * valid
        x = self._conv1d(x, "encoder.encoder.b1_conv2")
        x = self._dyt(x + skip, "encoder.encoder.b1_dyt2") * valid
        skip = x
        x = self._dyt(
            self._conv1d(x, "encoder.encoder.b2_conv1"),
            "encoder.encoder.b2_dyt1",
        ) * valid
        x = self._conv1d(x, "encoder.encoder.b2_conv2")
        x = self._dyt(x + skip, "encoder.encoder.b2_dyt2") * valid
        return self._linear(x, "encoder.encoder.linear") * valid

    def _predictor(
        self, features: object, valid: object, decoder: str, duration: bool = False
    ) -> object:
        tf = self.tf
        x = self._dyt(
            self._conv1d(features, f"encoder.{decoder}.conv1"),
            f"encoder.{decoder}.dyt1",
        ) * valid
        x = self._dyt(
            self._conv1d(x, f"encoder.{decoder}.conv2"),
            f"encoder.{decoder}.dyt2",
        ) * valid
        x = self._linear(x, f"encoder.{decoder}.linear") * valid
        if duration:
            x = tf.nn.relu(x) + 1.0
        return x * valid

    def _embedding(self, prediction: object, decoder: str) -> object:
        tf = self.tf
        bins = self._v(f"encoder.{decoder}.bins")
        embedding = self._v(f"encoder.{decoder}.embedding.weight")
        indices = tf.reduce_sum(
            tf.cast(prediction > bins, tf.int32),
            axis=-1,
        )
        return tf.gather(embedding, indices)

    def _mel_decoder(self, features: object, frame_valid: object) -> object:
        tf = self.tf
        x = self._depthwise(features, "decoder.proj") * frame_valid
        for block, first_dilation in (("block1", 1), ("block2", 3), ("block3", 5)):
            skip = x
            x = self._dyt(x, f"decoder.{block}_dyt1") * frame_valid
            x = self._depthwise(
                x, f"decoder.{block}_conv1_dw", dilation=first_dilation
            ) * frame_valid
            x = self._conv1d(x, f"decoder.{block}_conv1_pw") * frame_valid
            x = self._dyt(x, f"decoder.{block}_dyt2") * frame_valid
            x = self._depthwise(x, f"decoder.{block}_conv2_dw") * frame_valid
            x = self._conv1d(x, f"decoder.{block}_conv2_pw") * frame_valid
            x = (x + skip) * frame_valid
        x = self._linear(x, "decoder.mel_linear_up") * frame_valid
        x = self._dyt(x, "decoder.mel_dyt") * frame_valid
        return self._linear(x, "decoder.mel_linear_down") * frame_valid

    def __call__(self, phoneme: object) -> tuple[object, object]:
        tf = self.tf
        valid = tf.reshape(
            tf.cast(tf.not_equal(tf.round(phoneme), 0), tf.float32),
            (1, self.max_phonemes, 1),
        )
        features = self._encoder(phoneme, valid)

        pitch = self._predictor(features, valid, "pitch_decoder")
        energy = self._predictor(features, valid, "energy_decoder")
        duration = self._predictor(features, valid, "duration_decoder", duration=True)
        pitch_features = self._embedding(pitch, "pitch_decoder") * valid
        energy_features = self._embedding(energy, "energy_decoder") * valid
        duration_features = self._embedding(duration, "duration_decoder") * valid

        fused = self._linear(
            tf.concat((features, pitch_features, energy_features, duration_features), axis=-1),
            "encoder.fusion_linear",
        )
        fused *= valid

        durations = tf.clip_by_value(
            tf.round(tf.squeeze(duration, axis=-1)),
            0.0,
            float(self.max_mel_frames),
        )
        ends = tf.cumsum(durations, axis=1)
        frame_ids = tf.reshape(
            # Frame centers avoid off-by-one ownership after quantize/dequantize.
            tf.range(self.max_mel_frames, dtype=tf.float32) + 0.5,
            (1, self.max_mel_frames, 1),
        )
        owner = tf.reduce_sum(
            tf.cast(
                frame_ids >= tf.reshape(ends, (1, 1, self.max_phonemes)),
                tf.int32,
            ),
            axis=-1,
        )
        owner = tf.clip_by_value(owner, 0, self.max_phonemes - 1)
        upsampled = tf.gather(fused, owner, axis=1, batch_dims=1)

        # The int16 SUM kernel shares the per-phoneme scale and saturates.
        # CUMSUM runs on CPU; MAX uses its full utterance-length range instead.
        total_length = tf.reduce_max(ends, axis=1)
        mel_len = tf.round(tf.minimum(total_length, float(self.max_mel_frames)))
        frame_valid = tf.cast(
            tf.reshape(
                tf.range(self.max_mel_frames, dtype=tf.float32) + 0.5,
                (1, self.max_mel_frames),
            )
            < tf.reshape(mel_len, (1, 1)),
            tf.float32,
        )
        frame_valid = tf.reshape(frame_valid, (1, self.max_mel_frames, 1))
        upsampled *= frame_valid
        mel = self._mel_decoder(upsampled, frame_valid)
        return mel, mel_len


def _representative_inputs(
    max_phonemes: int, samples: int, vocab_size: int = 74
) -> Iterable[list[np.ndarray]]:
    rng = np.random.default_rng(260918856)
    for index in range(samples):
        length = min(max_phonemes, 8 + (index * 7) % max(1, max_phonemes - 7))
        ids = np.zeros((1, max_phonemes), dtype=np.float32)
        ids[0, :length] = rng.integers(1, vocab_size, size=length).astype(np.float32)
        yield [ids]


def calibration_inputs(path, frontend, max_phonemes, samples):
    sys.path.insert(0, str(REPOSITORY_ROOT / "grainspeech"))
    from text import text_to_sequence
    from text.frontend import parse_phonemes
    from text.vocabulary import PhoneVocabulary

    vocabulary = PhoneVocabulary(frontend["symbols"]) if "symbols" in frontend else None
    records = []
    oversized = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split("|")
        if len(fields) != 4:
            raise ValueError(f"{path}:{line_number}: expected four metadata columns")
        if vocabulary is not None:
            sequence = vocabulary.encode(fields[2])
        else:
            phones = parse_phonemes(fields[2])
            sequence = text_to_sequence("{" + " ".join(phones) + "}", [])
        if len(sequence) > max_phonemes:
            oversized += 1
            continue
        records.append((fields[0], sequence))
    if not records:
        raise ValueError("No real calibration sequences fit the exported input capacity")
    count = min(samples, len(records))
    # Preserve short/long inputs and the largest observed token ID, then sample the rest.
    chosen = list(dict.fromkeys((
        min(range(len(records)), key=lambda i: len(records[i][1])),
        max(range(len(records)), key=lambda i: len(records[i][1])),
        max(range(len(records)), key=lambda i: max(records[i][1])),
    )))[:count]
    rng = np.random.default_rng(260918856)
    remaining = np.array([i for i in range(len(records)) if i not in chosen])
    if len(chosen) < count:
        chosen.extend(rng.choice(remaining, count - len(chosen), replace=False).tolist())
    inputs = []
    observed = set()
    for index in chosen:
        sequence = records[index][1]
        phoneme = np.zeros((1, max_phonemes), dtype=np.float32)
        phoneme[0, :len(sequence)] = sequence
        inputs.append([phoneme])
        observed.update(sequence)
    print(f"Calibration: {count} real sequences; {oversized} oversized rows excluded", flush=True)
    return inputs, {
        "source": str(path), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "samples": count, "seed": 260918856, "oversized_rows": oversized,
        "record_ids": [records[i][0] for i in chosen],
        "unique_token_ids": len(observed), "maximum_token_id": max(observed),
    }


def _convert(
    module: StaticGrainSpeech,
    output: Path,
    max_phonemes: int,
    calibration_samples: int,
    float32: bool,
    representative_data=None,
    vocab_size: int = 74,
) -> None:
    import tensorflow as tf

    output.parent.mkdir(parents=True, exist_ok=True)
    concrete = tf.function(module).get_concrete_function(
        tf.TensorSpec((1, max_phonemes), tf.float32, name="phoneme"),
    )
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], module)
    if not float32:
        op_set = getattr(
            tf.lite.OpsSet,
            "EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8",
            None,
        )
        if op_set is None:
            raise RuntimeError(
                "This TensorFlow build does not expose the A16W8 TFLite op set; "
                "install a TensorFlow release that supports int16 activations."
            )
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: (
            iter(representative_data) if representative_data is not None
            else _representative_inputs(max_phonemes, calibration_samples, vocab_size)
        )
        # Keep unsupported bookkeeping on the CPU in the same graph.
        converter.target_spec.supported_ops = [op_set, tf.lite.OpsSet.TFLITE_BUILTINS]
        # Float interfaces do not change the A16W8 convolution/linear kernels.
        converter.inference_output_type = tf.float32
    model = converter.convert()
    output.write_bytes(model)


def main() -> None:
    args = parse_args()
    _require_positive("max-phonemes", args.max_phonemes)
    _require_positive("max-mel-frames", args.max_mel_frames)
    _require_positive("calibration-samples", args.calibration_samples)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {args.checkpoint}")
    loaded = _load_checkpoint(args.checkpoint)
    state = loaded.get("state_dict", loaded)
    if not isinstance(state, dict):
        raise TypeError("Checkpoint state_dict is not a dictionary")
    hyperparameters = loaded.get("hyper_parameters", {})
    config = hyperparameters.get("preprocess_config", {})
    frontend = config.get("preprocessing", {}).get("text", {"language": "en"})
    vocab_size = int(_state_value(state, "encoder.encoder.embed.weight").shape[0])
    if "symbols" in frontend:
        sys.path.insert(0, str(REPOSITORY_ROOT / "grainspeech"))
        from text.vocabulary import PhoneVocabulary

        if len(PhoneVocabulary(frontend["symbols"])) != vocab_size:
            raise ValueError("Checkpoint vocabulary and embedding dimensions disagree")
        if not args.float32 and args.calibration_data is None:
            raise ValueError("Custom-vocabulary A16W8 export requires --calibration-data")
        if args.output == DEFAULT_OUTPUT:
            raise ValueError("Choose a distinct --output for the custom vocabulary; keep the English model")
    elif vocab_size != 74:
        raise ValueError("Non-English embeddings require a checkpoint-local vocabulary")
    if args.stats is not None:
        stats = json.loads(args.stats.read_text(encoding="utf-8"))
        if (
            "symbols" in frontend and hyperparameters.get("feature_stats") is not None
            and stats != hyperparameters["feature_stats"]
        ):
            raise ValueError("Statistics do not match the custom-vocabulary checkpoint")
    elif hyperparameters.get("feature_stats") is not None:
        stats = hyperparameters["feature_stats"]
    elif "symbols" in frontend:
        raise ValueError("Custom-vocabulary checkpoints require embedded feature_stats or --stats")
    else:
        stats = json.loads(DEFAULT_STATS.read_text(encoding="utf-8"))
    representative_data = None
    calibration = None
    if not args.float32:
        if args.calibration_data is not None:
            representative_data, calibration = calibration_inputs(
                args.calibration_data, frontend, args.max_phonemes, args.calibration_samples,
            )
        else:
            calibration = {
                "source": "synthetic phoneme sequences", "samples": args.calibration_samples,
                "seed": 260918856,
            }
    module = StaticGrainSpeech(
        state=state,
        stats=stats,
        max_phonemes=args.max_phonemes,
        max_mel_frames=args.max_mel_frames,
    )
    _convert(
        module=module,
        output=args.output,
        max_phonemes=args.max_phonemes,
        calibration_samples=args.calibration_samples,
        float32=args.float32,
        representative_data=representative_data,
        vocab_size=vocab_size,
    )

    metadata = {
        "format": "tflite",
        "tflite_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "checkpoint_epoch": loaded.get("epoch"),
        "frontend": frontend,
        "quantization": "FP32" if args.float32 else "A16W8",
        "quantization_detail": (
            "Single graph with A16W8 convolutional/linear operators, CPU "
            "token lookup and bookkeeping, and float32 input/output interfaces."
            if not args.float32
            else "Acoustic computations are FP32; indices remain integer."
        ),
        "input": {
            "phoneme": [1, args.max_phonemes],
            "phoneme_dtype": "float32",
            "padding_id": 0,
            "vocab_size": vocab_size,
        },
        "output": {
            "mel": [1, args.max_mel_frames, 80],
            "mel_dtype": "float32",
            "mel_len": [1],
            "mel_len_dtype": "float32",
        },
        "calibration": calibration,
        "audio": {
            "sampling_rate": config.get("preprocessing", {}).get("audio", {}).get("sampling_rate", 22050),
            "hop_length": config.get("preprocessing", {}).get("stft", {}).get("hop_length", 256),
        },
        "ethos_u_note": (
            "Run Vela or the vendor Ethos-U compiler and inspect delegated ops; "
            "TFLite conversion alone does not guarantee full NPU delegation."
        ),
        "vocoder": "not included; the paper MCU result measures the acoustic model only",
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.output}")
    print(f"Wrote {args.output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
