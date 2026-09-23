> [!TIP]
> **[▶ Listen to GrainSpeech — open the audio demo](https://lab-emi.github.io/GrainSpeech/#demo)**
>
> Play speech samples directly in your browser and compare models. **No installation required.**

[![GrainSpeech: click the banner to listen to the audio demo](assets/grainspeech-banner.png)](https://lab-emi.github.io/GrainSpeech/#demo)

# GrainSpeech

**Less Context, More Detail for Compact Speech Synthesis**

**Authors:** Zitao Liang, Chang Gao\*  
\* Corresponding author.

This is the official repository for the paper
[**“GrainSpeech: Less Context, More Detail for Compact Speech Synthesis”**](https://arxiv.org/abs/2609.18856).
GrainSpeech is a **264.8K-parameter acoustic model** for compact text-to-speech
synthesis. It introduces two changes:

1. A **fixed-receptive-field convolutional encoder** that uses focused phoneme
   context for acoustic prediction.
2. An **anti-oversmoothing Mel loss** that combines L1, SSIM, and local
   gradient-variance (GVar) supervision.

[**▶ Listen to Audio Demos**](https://lab-emi.github.io/GrainSpeech/#demo) ·
[Paper](https://arxiv.org/abs/2609.18856) ·
[Text → Spectrogram Examples](#text--spectrogram-examples) ·
[GrainSpeech Quick Start](#grainspeech-quick-start) ·
[GrainSpeech Training](#grainspeech-training)

**中文完整流程文档：** [英文/中文 TTS、GrainSpeech 与 HiFi-GAN TFLite 导出、A16W8 量化及校准数据准备](docs/TTS_TFLITE_A16W8_GUIDE_ZH.md)

![GrainSpeech quality and model-size comparison](assets/grainspeech_sota.png)

## GrainSpeech Audio Demo

### [▶ Click here to listen and compare voices](https://lab-emi.github.io/GrainSpeech/#demo)

Choose a sentence and press play on the
[**GrainSpeech paper website**](https://lab-emi.github.io/GrainSpeech/).
Listen to GrainSpeech alongside the original recording, then explore the expanded
**Compare all models** section to hear the other systems. Five LJSpeech sentences
and 12 acoustic-model variants are available, with all playback on the same page.

All 65 comparison WAV files and their provenance live in this repository,
together with the model, code and paper website. To edit or preview the website,
see [`website/README.md`](website/README.md).

## Text → Spectrogram Examples

These examples were synthesized on CPU with the released
`grainspeech_l1_ssim_gvar.ckpt` checkpoint. Each plot shows the **predicted
80-bin log-Mel spectrogram directly from GrainSpeech**, before HiFi-GAN converts
it to audio. Both plots use the same color scale; time is in seconds.

**“Small models can give every word a voice.”** — 2.40 seconds

![Predicted Mel spectrogram for Small models can give every word a voice](assets/examples/compact-speech.png)

[Listen / download WAV](assets/examples/compact-speech.wav) ·
[Raw Mel array](assets/examples/compact-speech.npy)

**“The morning light falls softly on the quiet garden.”** — 2.98 seconds

![Predicted Mel spectrogram for The morning light falls softly on the quiet garden](assets/examples/morning-light.png)

[Listen / download WAV](assets/examples/morning-light.wav) ·
[Raw Mel array](assets/examples/morning-light.npy)

After the [Quick Start](#grainspeech-quick-start) installation, reproduce both
examples with:

```bash
python scripts/generate_readme_examples.py --device cpu
```

This writes the plots, WAV files, raw NumPy arrays, and a
[generation manifest](assets/examples/manifest.json) with the input phonemes,
checkpoint hash, settings, and package versions to `assets/examples/`.
The [banner artwork](assets/branding/README.md) illustrates this workflow;
the plots above are the original model outputs.

## Architecture

![GrainSpeech architecture](assets/grainspeech_architecture.png)

## GrainSpeech Quick Start

The reference environment uses Python 3.11, PyTorch 2.12, and Lightning 2.6.
The complete package snapshot is available in
`environment/reference-pip-freeze.txt`.

```bash
git clone https://github.com/lab-emi/GrainSpeech.git
cd GrainSpeech

python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m nltk.downloader averaged_perceptron_tagger averaged_perceptron_tagger_eng cmudict
```

The last command installs the language resources used by `g2p-en` to convert
ordinary English text into ARPAbet phonemes. It only needs to be run once per
environment. Inference with `--phonemes` does not use this conversion step.

Synthesize English text with the released checkpoint:

```bash
python grainspeech/infer.py \
  --checkpoint checkpoints/grainspeech_l1_ssim_gvar.ckpt \
  --text "Grain Speech is a compact text to speech model." \
  --device cpu \
  --output outputs/grainspeech.wav
```

Use `--device cuda` for GPU inference. To control the pronunciation directly,
replace `--text` with a space-separated ARPAbet sequence, for example:

```bash
python grainspeech/infer.py \
  --checkpoint checkpoints/grainspeech_l1_ssim_gvar.ckpt \
  --phonemes "G R EY1 N S P IY1 CH" \
  --device cpu \
  --output outputs/grainspeech.wav
```

The released inference checkpoint contains the trained GrainSpeech acoustic model
and HiFi-GAN vocoder weights. The small `configs/LJSpeech/stats.json` file lets
inference run without downloading the training dataset. The standalone HiFi-GAN
file under `hifigan/LJ_V2/` is also kept because the current model
constructor uses it while loading the checkpoint and when starting a new
training run.

### Export an A16W8 TFLite acoustic model

The paper's A16W8 result means **16-bit activations and 8-bit weights** on the
GrainSpeech acoustic model; it does not include HiFi-GAN vocoding. Export a
fixed-shape model for TFLite Micro/Ethos-U as follows:

```bash
python -m pip install -r requirements-export.txt
python scripts/export_tflite.py \
  --checkpoint checkpoints/grainspeech_l1_ssim_gvar.ckpt \
  --output outputs/grainspeech_a16w8.tflite \
  --max-phonemes 128 \
  --max-mel-frames 512
```

The model accepts `phoneme` (`float32`, shape `[1, 128]`) containing integer
phoneme IDs (1-73) and uses ID `0` as padding. It returns a fixed
`float32 [1, 512, 80]` Mel tensor plus `float32 [1]` `mel_len` (an integer-valued
frame count). These float interfaces surround A16W8 convolutional/linear
operators; they do not make the acoustic network FP32. Unlike the initial
exporter, the outputs are already dequantized: do not apply another scale.
Zero-padded input positions must be at the end; output frames at or after
`mel_len` are invalid. Change both capacities to match the memory budget of
the target MCU. `--float32` exports an unquantized TFLite model for numerical
comparison.

This remains **one TFLite graph**, with builtin CPU fallback rather than
separate CPU/NPU model files. Token lookup uses integer indices (`ONE_HOT`,
which TensorFlow 2.20 folds into `EMBEDDING_LOOKUP`), avoiding exact equality
on dequantized phoneme IDs. The script uses a static cumulative-duration
gather instead of PyTorch's dynamic `repeat_interleave`. Length is taken from
the CPU cumulative sum rather than an int16 `SUM`, whose per-phoneme scale
can saturate at about 45 frames. Frame-center comparisons tolerate small
quantization errors at integer boundaries.

Compare against an FP32 export and the original README predictions:

```bash
python scripts/export_tflite.py --float32 \
  --output outputs/grainspeech_fp32_reference.tflite
python scripts/check_tflite_export.py
```

The comparison checks token lookup, accumulated lengths, frame ownership,
padding, and A16W8 convolution/linear types. It saves predicted Mels and a
JSON report to `outputs/a16w8_comparison/`. With TensorFlow 2.20 and the
default capacities/calibration, the README sentences produce 208 vs. 207
frames and 257 vs. 257 frames (A16W8 vs. FP32). Calibration currently uses
synthetic phoneme sequences, so these examples are not a general speech
quality guarantee.

To listen, run HiFi-GAN on the saved Mels in a separate process:

```bash
python scripts/vocode_tflite_mels.py --device cpu
```

This uses the existing `hifigan/LJ_V2/generator_v2` and its adjacent
configuration, writing four 22.05 kHz PCM16 WAVs to `outputs/a16w8_audio/`:
`compact-speech_a16w8.wav`, `compact-speech_fp32.wav`,
`morning-light_a16w8.wav`, and `morning-light_fp32.wav`. Both variants use
the same FP32 vocoder, without per-file loudness normalization. The A16W8
files use the saved quantized-model Mels, not the FP32 acoustic model.

After export, run the target vendor's Ethos-U compiler (for example Vela) and
inspect the delegated operator list: a TFLite file being A16W8 does not by
itself guarantee that every operator is supported by a particular U55
configuration. For a U55-128 configuration, a Vela check is:

```bash
vela --accelerator-config ethos-u55-128 \
  --memory-mode Shared_Sram \
  --system-config Ethos_U55_High_End_Embedded \
  --show-cpu-operations \
  --output-dir outputs/vela \
  outputs/grainspeech_a16w8.tflite
```

The duration/indexing and token-lookup bookkeeping may stay on the CPU while
the convolutional and linear layers are delegated to the NPU. The vocoder
remains a separate deployment problem. As a reference, Vela 5.2.0 reports
about 664 KiB SRAM, 254 KiB off-chip flash, 167 NPU operators, and 41 CPU
operators for the corrected default 128/512 model with the configuration
above. These are compiler estimates, not the total MCU firmware/arena memory
or measured latency. Confirm that the target runtime provides all remaining
CPU kernels (including their float/int16 variants).

### Export an A16W8 HiFi-GAN TFLite vocoder

The vocoder is a **second model**: GrainSpeech predicts log-Mels, and HiFi-GAN
converts them to waveform samples. `scripts/export_hifigan_tflite.py` rebuilds
the repository generator in TensorFlow, folds PyTorch weight normalization,
and supports both residual-block variants. It preserves the final
LeakyReLU slope of 0.01 (the other LeakyReLUs use 0.1).

Run locally with access to `hifigan/LJ_V2/generator_v2` and the adjacent
`config.json`, after generating the comparison Mels above:

```bash
python scripts/export_hifigan_tflite.py \
  --output outputs/hifigan_a16w8.tflite \
  --mel-frames 64 \
  --calibration-dir outputs/a16w8_comparison \
  --calibration-samples 64
```

Use `--checkpoint` and `--config` to specify a different checkpoint/configuration.
The script uses actual saved Mel windows for calibration, compares its FP32
reconstruction with PyTorch before conversion, and writes a matching JSON with
tensor shapes, sample rate, hop length, hashes, operator precisions, waveform
error on the first calibration window, and required left/right context.
These numerical checks do not replace listening or calibration on a larger,
representative speech corpus.

The PyTorch reference pass temporarily disables oneDNN/MKLDNN: on ARM64,
co-loading TensorFlow and PyTorch can otherwise crash inside optimized
`Conv1d` kernels before conversion. This does not change the exported graph
or quantization. The exporter prints five progress stages and enables Python
fault traces so a native crash can be located without guessing its stage.

The interfaces are `float32 [1, mel_frames, 80]` log-Mel input and
`float32 [1, mel_frames * hop_length]` waveform output in approximately
`[-1, 1]`. Internally, supported convolutions and transpose convolutions use
int16 activations/int8 weights. Builtin CPU fallback is allowed. No additional
Mel normalization or output dequantization should be applied.

Generate audio **using the TFLite vocoder**, without loading a PyTorch vocoder:

```bash
python scripts/infer_hifigan_tflite.py \
  --model outputs/hifigan_a16w8.tflite \
  --mel-dir outputs/a16w8_comparison
```

WAVs are written to `outputs/hifigan_a16w8_audio/`. Filenames retain the
acoustic model's `_a16w8`/`_fp32` suffix; the directory identifies the
**vocoder's** precision. For a like-for-like FP32 vocoder comparison:

```bash
python scripts/export_hifigan_tflite.py --float32 \
  --output outputs/hifigan_fp32.tflite --mel-frames 64
python scripts/infer_hifigan_tflite.py \
  --model outputs/hifigan_fp32.tflite \
  --mel-dir outputs/a16w8_comparison
```

The inference script uses overlapping Mel windows with context derived from
the generator's receptive field and discards boundary samples, rather than
naively joining independent blocks. It anchors the first and last windows to
the real utterance boundaries. Inputs shorter than a window are zero-padded
and cropped with a warning: their tail can differ from unpadded PyTorch
inference. For exact boundaries on a short utterance, export with
`--mel-frames` equal to its Mel length.

The default 64-frame window is a starting point, **not an MCU memory or
real-time guarantee**. HiFi-GAN's waveform-rate activations can be expensive,
and transpose convolutions may remain on the CPU for the chosen U55.
Compile and inspect this vocoder separately:

```bash
vela --accelerator-config ethos-u55-128 \
  --memory-mode Shared_Sram \
  --system-config Ethos_U55_High_End_Embedded \
  --show-cpu-operations \
  --output-dir outputs/vela_hifigan \
  outputs/hifigan_a16w8.tflite
```

Exporter regressions can be exercised without pretrained files with
`python scripts/check_hifigan_export.py`. Those use deliberately small,
synthetic networks; they are not measurements of the released vocoder's
quality, SRAM consumption, or NPU coverage.

### End-to-end A16W8 TFLite speech synthesis

`scripts/infer_tflite.py` runs the complete pipeline in one command:
English text/ARPAbet (or Chinese numbered pinyin) -> GrainSpeech TFLite -> valid Mel frames ->
HiFi-GAN TFLite -> PCM16 WAV. It does not import PyTorch, load either original
checkpoint, or read the original HiFi-GAN configuration. It only needs the
two exported TFLite files and their matching export JSON metadata.

```bash
python scripts/infer_tflite.py \
  --text "Small models can give every word a voice." \
  --output outputs/my_speech.wav
```

Parameters can select the models, save the intermediate Mel, and control
TFLite CPU threading:

```bash
python scripts/infer_tflite.py \
  --text "The morning light falls softly on the quiet garden." \
  --acoustic-model outputs/grainspeech_a16w8.tflite \
  --vocoder-model outputs/hifigan_a16w8.tflite \
  --vocoder-metadata outputs/hifigan_a16w8.json \
  --threads 2 \
  --save-mel outputs/my_speech.npy \
  --output outputs/my_speech.wav
```

Replace `--text` with `--phonemes "G R EY1 N S P IY1 CH"` to supply ARPAbet
directly; these options are mutually exclusive. Text input requires the
`g2p-en`/NLTK resources installed in the Quick Start. Phoneme input does not
require those language resources.

For a trained Chinese model, use `--pinyin` and its acoustic export instead:

```bash
python scripts/infer_tflite.py \
  --acoustic-model outputs/grainspeech_baker_a16w8.tflite \
  --pinyin "ni2 hao3" \
  --output outputs/chinese_speech.wav
```

The Chinese acoustic JSON is required: it binds the model to its vocabulary.
See [Chinese export instructions](data/README.md#5-export-chinese-a16w8-and-synthesize-with-two-tflite-models)
for real-data calibration and model export.

The default models use 128 phonemes and at most 512 Mel frames per utterance.
Too many phonemes are rejected. Reaching the Mel capacity also stops synthesis
to prevent unnoticed truncation; shorten the input, export a larger acoustic
model, or explicitly use `--allow-truncation`. The vocoder automatically uses
its exported context sizes for overlapping-window synthesis. Output sample
rate and hop length come from its matching JSON; do not substitute a vocoder
trained with incompatible Mel preprocessing. Use the ordinary exported models
for desktop listening, not the Vela-compiled files requiring an Ethos-U runtime.

## GrainSpeech Training

For the existing local Baker Chinese Parquet corpus used by MOSS-TTS-Nano,
see [the Baker data preparation instructions](data/README.md#reusing-the-local-baker-chinese-corpus).
This imports original audio/text/pinyin without modifying that project or
reusing its codec tokens. The separate
[Chinese alignment and training workflow](data/README.md#chinese-alignment-and-acoustic-training)
uses tone-marked pinyin syllables, pretrained CTC alignment, and acoustic-only
GrainSpeech training. The instructions below remain the LJSpeech workflow.

Want to modify GrainSpeech or train your own variant? Complete the Quick Start
installation first, then prepare LJSpeech as follows.

### 1. Download LJSpeech

Download [LJSpeech 1.1](https://keithito.com/LJ-Speech-Dataset/) and extract it
directly under the repository's `data/` directory. The final location must be:

```text
GrainSpeech/
└── data/
    └── LJSpeech-1.1/
        ├── metadata.csv
        └── wavs/
```

In other words, `metadata.csv` must be available at
`data/LJSpeech-1.1/metadata.csv` when commands are run from the repository root.

### 2. Download the phoneme alignments

Download the precomputed
[LJSpeech TextGrids](https://drive.google.com/drive/folders/1DBRkALpPd6FL9gjHMmMEdHODmkgNIIK4).
Place the downloaded `.TextGrid` files at:

```text
GrainSpeech/data/LJSpeech-1.1/TextGrid/LJSpeech/
```

The `.TextGrid` files should be directly inside the final `LJSpeech/` folder,
not inside an additional nested directory.

A TextGrid records the time interval occupied by each phoneme in an utterance.
GrainSpeech uses these phoneme-to-audio alignments to obtain duration targets and
to align pitch, energy, and Mel-spectrogram features with the phoneme sequence.

We thank the [EfficientSpeech](https://github.com/roatienza/efficientspeech)
authors for their LJSpeech processing workflow. The preprocessing code in this
repository is adapted from EfficientSpeech, which follows the FastSpeech 2 data
pipeline.

### 3. Preprocess LJSpeech

Run the preprocessing command from the repository root:

```bash
python grainspeech/preprocess.py \
  --preprocess-config configs/LJSpeech/preprocess.yaml \
  --textgrid-dir data/LJSpeech-1.1/TextGrid \
  --device auto
```

`--device auto` uses CUDA when it is available and otherwise runs on CPU.

This command cleans the transcripts, prepares normalized waveforms, installs
the TextGrids, and generates Mel spectrograms, pitch, energy, durations,
metadata splits, and normalization statistics. It writes everything under the
local `data/LJSpeech-1.1/` directory rather than into Git:

```text
data/LJSpeech-1.1/
├── metadata.csv
├── wavs/
├── TextGrid/LJSpeech/
├── raw_data/LJSpeech/LJSpeech/
└── preprocessed_data/LJSpeech/
    ├── TextGrid/LJSpeech/
    ├── mel/
    ├── pitch/
    ├── energy/
    ├── duration/
    ├── train.txt
    ├── val.txt
    ├── speakers.json
    └── stats.json
```

Full preprocessing creates normalized waveform copies and NumPy feature files,
so make sure the `data/` directory has enough free disk space.

For a quick setup test, add `--limit 1`; do not use this option for full
training. Once preprocessing finishes, validate the data layout:

```bash
python scripts/check_setup.py
```

### 4. Train GrainSpeech

Train the paper model with L1, SSIM, and GVar supervision:

```bash
python grainspeech/train_l1_ssim_gvar.py \
  --run-name grainspeech-l1-ssim-gvar \
  --preprocess-config configs/LJSpeech/preprocess.yaml \
  --hifigan-checkpoint hifigan/LJ_V2/generator_v2 \
  --accelerator gpu --devices 1 --precision 16-mixed \
  --batch-size 128 --num_workers 4 --max_epochs 5000 \
  --lr 0.001 --weight-decay 0.00001 --infer-device cuda
```

The available training objectives are:

```text
grainspeech/train.py                  # L1
grainspeech/train_l1_ssim.py          # L1 + SSIM
grainspeech/train_l1_ssim_gvar.py     # L1 + SSIM + GVar (GrainSpeech)
```

Training checkpoints and TensorBoard logs are written under
`lightning_logs/<run-name>/`. Resume a full Lightning checkpoint by adding:

```bash
--checkpoint lightning_logs/<run-name>/checkpoints/last.ckpt
```

The released checkpoint in `checkpoints/` is inference-only and cannot resume
training. Add `--compile` to enable `torch.compile` for training.

The full LJSpeech dataset, TextGrids and generated training features stay outside
Git. Only the small published listening examples under `website/demo/audio/`
are included in the repository.

## Citation

```bibtex
@article{liang2026grainspeech,
  title   = {GrainSpeech: Less Context, More Detail for Compact Speech Synthesis},
  author  = {Liang, Zitao and Gao, Chang},
  journal = {arXiv preprint arXiv:2609.18856},
  year    = {2026},
  doi     = {10.48550/arXiv.2609.18856},
  url     = {https://arxiv.org/abs/2609.18856}
}
```
