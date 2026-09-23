"""Export the repository's HiFi-GAN generator as a fixed-shape TFLite vocoder.

The default is A16W8 with builtin CPU fallback and float32 Mel/audio interfaces.
Calibration uses saved, already-dequantized log-Mel arrays, not random noise.
Run this locally with access to the checkpoint and its adjacent configuration.
"""

import argparse
from collections import Counter
import faulthandler
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def validate_config(config):
    if config["resblock"] not in ("1", "2"):
        raise ValueError("Only repository HiFi-GAN resblock types '1' and '2' are supported")
    rates = config["upsample_rates"]
    kernels = config["upsample_kernel_sizes"]
    if not rates or len(rates) != len(kernels):
        raise ValueError("Upsample rates and kernels must have equal nonzero lengths")
    for stride, kernel in zip(rates, kernels):
        if (
            type(stride) is not int or type(kernel) is not int
            or stride <= 0 or kernel < stride or (kernel - stride) % 2
        ):
            raise ValueError("Upsampling requires positive strides and even kernel-stride")
    channels = config["upsample_initial_channel"]
    if type(channels) is not int or channels <= 0 or channels % (2 ** len(rates)):
        raise ValueError("Initial channels must be a positive multiple of 2**upsample_stages")
    residual_kernels = config["resblock_kernel_sizes"]
    dilations = config["resblock_dilation_sizes"]
    if not residual_kernels or len(residual_kernels) != len(dilations):
        raise ValueError("Residual kernels and dilation groups must have equal nonzero lengths")
    count = 3 if config["resblock"] == "1" else 2
    for kernel, group in zip(residual_kernels, dilations):
        if type(kernel) is not int or kernel <= 0 or kernel % 2 != 1:
            raise ValueError("Residual kernels must be positive odd integers")
        if len(group) != count or any(type(d) is not int or d <= 0 for d in group):
            raise ValueError(f"Each residual block must have {count} positive dilations")
    if config.get("num_mels", 80) != 80:
        raise ValueError("The repository generator requires 80 Mel bins")
    hop_length = math.prod(rates)
    if "hop_size" in config and config["hop_size"] != hop_length:
        raise ValueError("hop_size does not match the product of upsample_rates")
    if type(config["sampling_rate"]) is not int or config["sampling_rate"] <= 0:
        raise ValueError("sampling_rate must be a positive integer")


def context_frames(config):
    """Conservative Mel context for every sample in one output hop."""
    hop = math.prod(config["upsample_rates"])
    left, right = -3, hop - 1 + 3  # Final kernel-7 convolution.
    radius = max(
        (kernel - 1) // 2
        * sum(d + (1 if config["resblock"] == "1" else 0) for d in dilations)
        for kernel, dilations in zip(
            config["resblock_kernel_sizes"], config["resblock_dilation_sizes"]
        )
    )
    for stride, kernel in reversed(list(zip(
        config["upsample_rates"], config["upsample_kernel_sizes"]
    ))):
        left -= radius
        right += radius
        padding = (kernel - stride) // 2
        left = -(-(left + padding - kernel + 1) // stride)
        right = (right + padding) // stride
    return {"left": max(0, 3 - left), "right": max(0, right + 3)}


def load_mel(path):
    mel = np.load(path, allow_pickle=False)
    if mel.ndim != 2 or mel.shape[0] == 0 or mel.shape[1] != 80:
        raise ValueError(f"{path}: expected a nonempty [frames, 80] Mel array")
    if not np.issubdtype(mel.dtype, np.floating) or not np.isfinite(mel).all():
        raise ValueError(f"{path}: expected finite, already-dequantized floating-point Mel")
    return np.asarray(mel, dtype=np.float32)


def calibration_windows(mels, mel_frames, samples):
    for index in range(samples):
        mel = mels[index % len(mels)]
        window_index = index // len(mels)
        windows_for_mel = (samples - 1 - index % len(mels)) // len(mels) + 1
        last_start = max(0, len(mel) - mel_frames)
        start = round(last_start * window_index / max(1, windows_for_mel - 1))
        window = np.zeros((1, mel_frames, 80), dtype=np.float32)
        selected = mel[start:start + mel_frames]
        window[0, :len(selected)] = selected
        yield [window]


def load_generator(checkpoint, config):
    import torch
    from hifigan import AttrDict, Generator

    generator = Generator(AttrDict(config)).cpu().eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    generator.load_state_dict(state["generator"], strict=True)
    generator.remove_weight_norm()
    return generator


def pytorch_reference(generator, mel):
    import torch

    # Co-loaded TensorFlow/PyTorch oneDNN kernels can segfault on ARM Conv1d.
    # Only this comparison pass uses native kernels; restore the backend after it.
    with torch.backends.mkldnn.flags(enabled=False), torch.inference_mode():
        return generator(torch.from_numpy(mel).transpose(1, 2))[:, 0].numpy()


class StaticHiFiGAN(tf.Module):
    def __init__(self, state, config, mel_frames):
        super().__init__()
        validate_config(config)
        if mel_frames <= 0:
            raise ValueError("mel_frames must be positive")
        self.config = config
        self.mel_frames = mel_frames
        self.hop_length = math.prod(config["upsample_rates"])
        self.weights = {
            name: tf.constant(value.detach().cpu().numpy(), dtype=tf.float32)
            for name, value in state.items()
        }

    def _conv(self, x, name, dilation=1):
        kernel = tf.transpose(self.weights[f"{name}.weight"], (2, 1, 0))
        return tf.nn.conv1d(
            x, kernel, stride=1, padding="SAME", dilations=dilation,
        ) + self.weights[f"{name}.bias"]

    def _upsample(self, x, stage):
        name = f"ups.{stage}"
        stride = self.config["upsample_rates"][stage]
        # PyTorch ConvTranspose1d weights are [in, out, kernel].
        kernel = tf.expand_dims(
            tf.transpose(self.weights[f"{name}.weight"], (2, 1, 0)), 0
        )
        channels = int(self.weights[f"{name}.weight"].shape[1])
        length = int(x.shape[1]) * stride
        x = tf.nn.conv2d_transpose(
            tf.expand_dims(x, 1), kernel,
            output_shape=(1, 1, length, channels),
            strides=(1, 1, stride, 1), padding="SAME",
        )
        return tf.squeeze(x, 1) + self.weights[f"{name}.bias"]

    def _resblock(self, x, index, dilations):
        name = f"resblocks.{index}"
        for layer, dilation in enumerate(dilations):
            y = tf.nn.leaky_relu(x, alpha=0.1)
            if self.config["resblock"] == "1":
                y = self._conv(y, f"{name}.convs1.{layer}", dilation)
                y = tf.nn.leaky_relu(y, alpha=0.1)
                y = self._conv(y, f"{name}.convs2.{layer}")
            else:
                y = self._conv(y, f"{name}.convs.{layer}", dilation)
            x = x + y
        return x

    def __call__(self, mel):
        x = self._conv(mel, "conv_pre")
        count = len(self.config["resblock_kernel_sizes"])
        for stage in range(len(self.config["upsample_rates"])):
            x = self._upsample(tf.nn.leaky_relu(x, alpha=0.1), stage)
            branches = [
                self._resblock(x, stage * count + branch, dilations)
                for branch, dilations in enumerate(self.config["resblock_dilation_sizes"])
            ]
            x = branches[0]
            for branch in branches[1:]:
                x = x + branch
            x = x / float(count)
        # The repository uses F.leaky_relu(x) here: its default slope is 0.01.
        x = self._conv(tf.nn.leaky_relu(x, alpha=0.01), "conv_post")
        return tf.squeeze(tf.tanh(x), -1)


def convert_model(module, representative, float32):
    concrete = tf.function(module).get_concrete_function(
        tf.TensorSpec((1, module.mel_frames, 80), tf.float32, name="mel")
    )
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], module)
    if not float32:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS,
        ]
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32
    return converter.convert()


def inspect_model(model, num_threads=1):
    interpreter = tf.lite.Interpreter(model_content=model, num_threads=num_threads)
    interpreter.allocate_tensors()
    details = {d["index"]: d for d in interpreter.get_tensor_details()}
    summary = Counter()
    for op in interpreter._get_ops_details():
        if op["op_name"] in ("CONV_2D", "TRANSPOSE_CONV"):
            activation = int(op["inputs"][2 if op["op_name"] == "TRANSPOSE_CONV" else 0])
            weight = int(op["inputs"][1])
            types = f"{details[activation]['dtype'].__name__}/{details[weight]['dtype'].__name__}"
            summary[f"{op['op_name']} {types}"] += 1
    return interpreter, dict(summary)


def run_interpreter(interpreter, mel):
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("Expected one Mel input and one audio output")
    if inputs[0]["dtype"] != np.float32 or outputs[0]["dtype"] != np.float32:
        raise ValueError("Expected float32 vocoder interfaces")
    interpreter.set_tensor(inputs[0]["index"], mel)
    interpreter.invoke()
    audio = interpreter.get_tensor(outputs[0]["index"])
    if not np.isfinite(audio).all():
        raise RuntimeError("TFLite vocoder produced non-finite audio")
    return audio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "hifigan/LJ_V2/generator_v2")
    parser.add_argument("--config", type=Path, help="Default: config.json adjacent to checkpoint")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/hifigan_a16w8.tflite")
    parser.add_argument("--mel-frames", type=int, default=64)
    parser.add_argument("--calibration-dir", type=Path, default=ROOT / "outputs/a16w8_comparison")
    parser.add_argument("--calibration-samples", type=int, default=64)
    parser.add_argument("--float32", action="store_true")
    args = parser.parse_args()
    faulthandler.enable()
    if args.mel_frames <= 0 or args.calibration_samples <= 0:
        parser.error("--mel-frames and --calibration-samples must be positive")
    config_path = args.config or args.checkpoint.with_name("config.json")
    with config_path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    validate_config(config)
    paths = sorted(args.calibration_dir.glob("*.npy"))
    if not paths:
        raise FileNotFoundError(f"No calibration Mel .npy files in {args.calibration_dir}")
    mels = [load_mel(path) for path in paths]
    if args.calibration_samples < len(mels):
        parser.error("--calibration-samples must cover every calibration file")
    representative = lambda: calibration_windows(mels, args.mel_frames, args.calibration_samples)

    import torch

    torch.set_num_threads(1)
    print("[1/5] Loading HiFi-GAN and folding weight normalization", flush=True)
    generator = load_generator(args.checkpoint, config)
    example = next(representative())[0]
    print("[2/5] PyTorch reference inference (oneDNN/MKLDNN disabled)", flush=True)
    reference = pytorch_reference(generator, example)
    print("[3/5] TensorFlow reconstruction and FP32 comparison", flush=True)
    module = StaticHiFiGAN(generator.state_dict(), config, args.mel_frames)
    rebuilt = module(example).numpy()
    np.testing.assert_allclose(rebuilt, reference, atol=1e-4, rtol=1e-4)

    print("[4/5] TFLite conversion and calibration", flush=True)
    model = convert_model(module, representative, args.float32)
    print("[5/5] TFLite inference and output comparison", flush=True)
    interpreter, operators = inspect_model(model)
    audio = run_interpreter(interpreter, example)
    if audio.shape != reference.shape:
        raise RuntimeError(f"Unexpected audio shape {audio.shape}; expected {reference.shape}")
    if args.float32:
        np.testing.assert_allclose(audio, reference, atol=1e-4, rtol=1e-4)
    elif not any("int16/int8" in key for key in operators):
        raise RuntimeError("Conversion did not produce any A16W8 convolution operators")
    error = audio.astype(np.float64) - reference
    metadata = {
        "format": "tflite",
        "model": "HiFi-GAN",
        "quantization": "FP32" if args.float32 else "A16W8 with builtin CPU fallback",
        "input": {"name": "mel", "shape": [1, args.mel_frames, 80], "dtype": "float32",
                  "representation": "natural-log Mel amplitude; no additional normalization"},
        "output": {"name": "audio", "shape": list(audio.shape), "dtype": "float32",
                   "range": [-1.0, 1.0]},
        "sampling_rate": config["sampling_rate"],
        "hop_length": module.hop_length,
        "context_frames": context_frames(config),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "tflite_sha256": hashlib.sha256(model).hexdigest(),
        "tensorflow": tf.__version__,
        "calibration": {"samples": args.calibration_samples, "files": [str(p) for p in paths]},
        "convolution_operators": operators,
        "first_window_comparison": {
            "tensorflow_vs_pytorch_max_abs": float(np.max(np.abs(rebuilt - reference))),
            "tflite_vs_pytorch_mae": float(np.mean(np.abs(error))),
            "tflite_vs_pytorch_max_abs": float(np.max(np.abs(error))),
        },
        "deployment_note": (
            "No real-time or MCU SRAM guarantee. Compile with the target Vela configuration "
            "and confirm CPU kernels, especially transpose convolutions, on the device."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(model)
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {args.output} ({len(model)} bytes)")
    print(f"Wrote {args.output.with_suffix('.json')}")
    print(f"Convolutions: {operators}")
    print(f"First-window waveform MAE: {metadata['first_window_comparison']['tflite_vs_pytorch_mae']:.6g}")


if __name__ == "__main__":
    main()
