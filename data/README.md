# Dataset directory

Extract LJSpeech into `data/LJSpeech-1.1/`, then place the alignment files in
`data/LJSpeech-1.1/TextGrid/LJSpeech/`. See the Training section of the main
README for the complete preprocessing commands.

All contents below `data/` except this file are ignored by Git.

## Reusing the local Baker Chinese corpus

The public repository contains **code and documentation only**: it does not
distribute Baker recordings, transcripts, alignments, training features,
checkpoints, or generated TFLite/audio files. The [corpus publisher](https://www.data-baker.com/open_source.html)
states that the 10,000-sentence
corpus is for non-commercial use only; permission to redistribute it is not
established by the project's code license. Obtain the corpus and any required
permissions separately before following these steps.

The MOSS project uses codec-token shards for its own model. GrainSpeech instead
needs the original recordings and transcripts, available in
`~/datasets/baker/data/train-*.parquet`. Do not treat codec tokens or the MOSS
project's Mel cache as GrainSpeech training features.

After installing the main dependencies and `requirements-baker.txt`, import
a small subset first:

```bash
python scripts/prepare_baker.py \
  --corpus-dir ~/datasets/baker/data \
  --output-dir data/Baker-preview \
  --limit 8 --mel-samples 3
```

The importer leaves the source files untouched. It writes 22050 Hz mono PCM16
WAVs (peak-normalized to 0.95), Chinese `.lab` transcripts, and `manifest.jsonl`
with original text, supplied tone-marked pinyin, and source shard/row indices.
Source `file_name` IDs and embedded audio path annotations are also preserved.
Prosody markers `#1` through `#4` are removed only from `.lab`/clean text, not
from the original annotation in the manifest. Optional reference Mels use the
existing GrainSpeech audio/Mel settings, not the MOSS codec settings.

To prepare all records, omit `--limit` and choose a fresh output directory,
for example `--output-dir data/Baker`. Nonempty output directories are rejected
to avoid overwriting prior data or mixing a subset with a full preparation.
Without `--mel-samples`, no reference Mels are extracted.

Listen to the reference Mels with the existing TFLite vocoder:

```bash
python scripts/infer_hifigan_tflite.py \
  --mel-dir data/Baker-preview/mel \
  --output-dir outputs/baker_vocoder_preview
```

**This is only corpus preparation.** Its `dataset.json` remains marked
`training_ready: false`; aligned training features are generated separately
below. The LJSpeech trainer and English checkpoints must not be used unchanged
on these files. No uniformly divided duration fallback is generated.

The MOSS `ctc_align_baker.py` pipeline estimates durations in codec frames;
failed alignments can fall back to uniform durations without a per-record
failure flag. Its aligned cache omits source row IDs, and its Mel preparation
associates records by position even though alignment input can be filtered.
Do not reuse those caches as trusted GrainSpeech duration labels. New alignment
should retain the manifest IDs and explicitly report failures.

## Chinese alignment and acoustic training

The first Chinese training path uses **whole tone-marked pinyin syllables** as
tokens. It does not invent initial/final boundaries. Supplied corpus tones are
preserved, including surface-tone annotations and explicit erhua. Vocabulary,
pronunciation lookup, and normalization statistics are stored in the acoustic
checkpoint; the English vocabulary and released checkpoint remain unchanged.

Install `requirements.txt` and `requirements-baker.txt` first. On the existing
machine, `/tmp/grainspeech-export-venv/bin/python` already has these dependencies.
Run all commands from the repository root.

### 1. Align the prepared recordings

```bash
OMP_NUM_THREADS=4 python scripts/align_baker.py \
  --dataset data/Baker \
  --output-dir data/Baker-alignments \
  --disable-cudnn
```

The pinned, Apache-2.0-declared
[`wbbbbb/wav2vec2-large-chinese-zh-cn`](https://huggingface.co/wbbbbb/wav2vec2-large-chinese-zh-cn)
model is downloaded once (about 1.28 GB) into `external/alignment-models/`.
Safetensors weights are required by default and loaded explicitly as
`Wav2Vec2ForCTC`, with missing/unexpected weights rejected rather than randomly
initializing an alignment head. All recordings and transcripts are processed locally. Native Transformers
and PyTorch are used; MFA, Kaldi, and torchaudio are not required.

Alignment uses waveform-derived CTC character spans and associates them with
the supplied pinyin. Explicit trailing Hanzi `er` is grouped into its erhua
syllable. Since CTC blank is not silence, boundaries are placed midway between
adjacent character spans, with the outer edges at the recording boundaries.
These are **automatic syllable estimates, not ground-truth phoneme timings**.
Inspect the generated `TextGrid/Baker/` and per-record scores before a long run.

Unsupported characters, mixed-script text, transcript/pinyin mismatches,
high ASR character error rate, low character confidence, and impossible paths
are rejected with reasons in `records/<id>.json`. Nothing is substituted with
unknown tokens or uniform durations. Consequently, fewer than 10,000 recordings
will be retained. `summary.json` reports actual counts, model revision, source
manifest hash, and thresholds. Do not weaken thresholds merely to retain more
data.

For a representative pilot, add `--sample-size 64` and use a separate output
such as `data/Baker-align-pilot`. This samples across the corpus with a fixed
seed; `--limit 64` instead selects the first 64 recordings, which are not
necessarily representative. To resume an interrupted alignment run, repeat the
same command with `--resume`; source, model revision, subset, and thresholds
must match. A completed subset cannot be silently resumed as a full run.

`--disable-cudnn` is an explicit workaround for the current Torch/CUDA
environment's `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`: it retains CUDA but
uses native PyTorch convolution kernels. Omit it on a healthy cuDNN installation.
Alternatively, alignment/features accept `--device cpu`, and training accepts
`--accelerator cpu`. No system CUDA libraries are changed.

### 2. Extract training features

```bash
OMP_NUM_THREADS=4 python scripts/preprocess_baker.py \
  --dataset data/Baker \
  --alignments data/Baker-alignments \
  --output-dir data/Baker-features \
  --disable-cudnn
```

The shared GrainSpeech extractor generates syllable-level duration, pitch,
energy, and 80-bin log-Mel arrays. Every duration sequence must sum to the
corresponding Mel frame count. Statistics are fitted on the training split
only; validation is deterministic and nonempty. The output includes
`preprocess.yaml`, `train.txt`, `val.txt`, `stats.json`, `rejected.json`, and a
new `dataset.json`. A fresh output directory is required to avoid mixing
normalized features from different runs.

### 3. Train the Chinese acoustic model

```bash
OMP_NUM_THREADS=4 python scripts/train_baker.py \
  --preprocess-config data/Baker-features/preprocess.yaml \
  --run-name grainspeech-baker \
  --accelerator gpu --batch-size 16 --workers 4 \
  --epochs 1000 --precision 32-true \
  --disable-cudnn
```

This trains GrainSpeech from random initialization with the L1 + SSIM + GVar
objective. It does **not** load or train HiFi-GAN, read its original config,
or overwrite the English model. Checkpoints go to
`lightning_logs/grainspeech-baker/checkpoints/`; each invocation gets a separate
CSV metrics directory. To resume, add
`--checkpoint lightning_logs/grainspeech-baker/checkpoints/last.ckpt`.
Vocabulary, preprocessing settings, and statistics must match.

For a short plumbing run, use a separate run name and add `--max-steps 4`;
that checkpoint is not suitable for judging speech quality. Initial model
quality and convergence require actual training, not just successful execution.

### 4. Generate a Mel after training

```bash
python scripts/infer_baker.py \
  --checkpoint lightning_logs/grainspeech-baker/checkpoints/last.ckpt \
  --pinyin "ni2 hao3" \
  --output outputs/baker_mel.npy

python scripts/infer_hifigan_tflite.py \
  --mel outputs/baker_mel.npy \
  --output-dir outputs/baker_audio
```

Currently inference takes **numbered pinyin**, with optional `sp` pauses; it
does not yet perform automatic Hanzi normalization, polyphone resolution, or
tone sandhi. Supply the intended surface tones explicitly. Unknown syllables
are errors. Mel inference and TFLite vocoding run in separate processes to
avoid mixing PyTorch and TensorFlow native backends on ARM.

### 5. Export Chinese A16W8 and synthesize with two TFLite models

Choose the best validation checkpoint, which is not necessarily `last.ckpt`.
The following example uses the first completed Baker run's best checkpoint:

```bash
python scripts/export_tflite.py \
  --checkpoint lightning_logs/grainspeech-baker/checkpoints/epoch=129-step=46930-val_loss=12.9407.ckpt \
  --calibration-data data/Baker-features/train.txt \
  --calibration-samples 256 \
  --output outputs/grainspeech_baker_a16w8.tflite

python scripts/infer_tflite.py \
  --acoustic-model outputs/grainspeech_baker_a16w8.tflite \
  --pinyin "ni2 hao3 huan1 ying2 shi3 yong4 zhong1 wen2 yu3 yin1 he2 cheng2" \
  --output outputs/baker_a16w8.wav
```

The exporter reads the checkpoint-local vocabulary and statistics. Quantization
uses real training sequences with their Chinese IDs, not random English IDs.
Keep the adjacent `grainspeech_baker_a16w8.json`: it contains the vocabulary,
pinyin lookup, calibration provenance, model hash, and audio settings needed
by inference. Both TFLite models have float32 interfaces; convolution and
linear operations inside the acoustic model use A16W8. Token lookup and
unsupported bookkeeping remain CPU operations. No MCU/NPU coverage is implied
until the Chinese model is compiled and measured on the target.

English inference remains the default; Chinese requires `--pinyin` and the
Chinese model/JSON pair. The default capacities are 128 syllables and 512 Mel
frames; inference rejects oversized input and possible output truncation.
`--threads`, `--save-mel`, and `--vocoder-model` work as in the English pipeline.
The original PyTorch checkpoint and training dataset are not needed at runtime.

For an FP32 acoustic TFLite reference, export the same checkpoint with
`--float32 --output outputs/grainspeech_baker_fp32.tflite`.
`scripts/check_baker_tflite.py` renders paired FP32/A16W8 listening files into
`outputs/baker_a16w8_audio/`, using the same A16W8 vocoder for both. It compares
against the matching PyTorch reference Mels in `outputs/baker_trained_mels/`.
