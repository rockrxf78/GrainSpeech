# GrainSpeech + HiFi-GAN：英文与中文 TTS 训练、TFLite 导出及 A16W8 量化

本文整理本次会话的实现、操作流程、问题定位和产物，面向带 Arm Ethos-U55 的 MCU 部署准备。记录日期：**2026-09-20**。

命令与当前仓库脚本一致；示例路径对应本次机器上的文件。文中的数值是本次实验记录，不代表不同环境、重新训练或更换校准数据后仍会完全相同。

**公开仓库只包含代码与文档**，不包含 Baker 录音、标注、对齐、训练特征、checkpoint、导出的 TFLite 或试听文件。下文产物路径是本次本地实验记录；在新环境使用时须自行准备有权使用的数据并重新生成相应产物。标贝科技的[数据集说明](https://www.data-baker.com/open_source.html)注明仅支持非商用，代码许可证不代表获得语料的再分发许可。

> **当前完成的范围**
>
> - 英文：已发布 GrainSpeech → FP32/A16W8 TFLite → HiFi-GAN → WAV。
> - 中文：复用 Baker 原始语料 → 自动音节对齐 → 特征提取 → 训练 GrainSpeech → FP32/A16W8 TFLite → 同一个 A16W8 HiFi-GAN → WAV。
> - 中文推理目前接收**带调拼音**，不是任意汉字直接输入；自动文本规范化、多音字消歧和变调前端尚未接入。
> - 两个模型都有桌面 TFLite 推理产物，但**完整链路尚未完成 MCU 板端内存、延迟、实时率验证**。

## 目录

1. [架构、量化含义与接口](#1-架构量化含义与接口)
2. [环境与文件约定](#2-环境与文件约定)
3. [英文 TTS 完整流程](#3-英文-tts-完整流程)
4. [HiFi-GAN 导出、量化与校准数据](#4-hifi-gan-导出量化与校准数据)
5. [中文 TTS 完整流程](#5-中文-tts-完整流程)
6. [本次问题与修复记录](#6-本次问题与修复记录)
7. [Ethos-U55 部署边界](#7-ethos-u55-部署边界)
8. [产物、结果与复现检查](#8-产物结果与复现检查)
9. [脚本索引与参考资料](#9-脚本索引与参考资料)

## 1. 架构、量化含义与接口

### 1.1 两个神经网络，不是一个直接输出音频的模型

```text
英文文本 → g2p-en → ARPAbet ID ─┐
                              ├→ GrainSpeech → 有效 log-Mel → HiFi-GAN → PCM16 WAV
中文带调拼音 → 中文词表 ID ──────┘
```

**GrainSpeech 是声学模型，只输出 Mel 声谱图。** HiFi-GAN 是独立声码器，把 Mel 转成波形。论文中 GrainSpeech 的 MCU 声学模型结果不能直接视为“两个模型组成的完整 TTS 在 MCU 上实时运行”的证明。

本次采用训练后量化（PTQ）：训练好的浮点权重加代表性输入进行校准，而不是量化感知训练（QAT）。

**A16W8 表示内部目标算子使用 16 位激活和 8 位权重。** 不表示整个 TFLite 文件里所有张量都必须是 int16/int8；输入输出、查表、索引和部分辅助运算仍可使用 float32 或整数 CPU 算子。

### 1.2 当前默认接口

| 模型 | 输入 | 输出 | 说明 |
|---|---|---|---|
| 英文 GrainSpeech | `float32 [1,128]` | `float32 [1,512,80]` Mel + `float32 [1]` 长度 | 有效 ID 1–73，0 为尾部 padding |
| 中文 GrainSpeech | `float32 [1,128]` | 同上 | ID 来自该模型的 JSON 词表；本次 embedding 共 1608 行 |
| HiFi-GAN | `float32 [1,64,80]` | `float32 [1,16384]` 波形 | 22050 Hz，hop 256，输出约在 `[-1,1]` |

输入 ID 虽然放在 float32 张量里，数值仍必须是合法整数 ID。声学输出已经反量化，**不要再乘量化 scale**；只取 `mel[:mel_len]`，不要把补齐到 512 帧的无效尾部送入声码器。

默认容量可以修改：

- `--max-phonemes`：英文音素数或中文音节数上限。
- `--max-mel-frames`：声学输出帧数上限。512 帧在当前 hop/采样率下约为 5.94 秒，不是任意长句保证。
- `--mel-frames`：声码器每个固定窗口的 Mel 帧数，不是整句长度限制。

端到端脚本在长度达到声学容量时默认拒绝输出，以避免无提示截断。`--allow-truncation` 只适合明确接受截断的情况。

### 1.3 两种校准数据不能混淆

| 导出对象 | 校准输入 | 不是 |
|---|---|---|
| GrainSpeech | 真实前端产生的音素/音节 ID 序列 | WAV、Mel、声码器编码 |
| HiFi-GAN | 真实浮点 log-Mel 窗口 | 音素 ID、图片、随机噪声、未反量化 int16 Mel |

校准用于估计激活范围，不更新训练权重。样本数量不是唯一指标：应覆盖目标语言、句长、发音、音高、能量和实际使用场景。试听集与校准集尽量分开。

## 2. 环境与文件约定

### 2.1 工作目录与 Python

以下命令都在仓库根目录运行：

```bash
cd /home/ronren/GrainSpeech
PY=/tmp/grainspeech-export-venv/bin/python
```

这是本次已配置好的解释器；`/tmp` 下环境不保证永久存在。其他机器应创建持久虚拟环境，然后把 `PY` 指向它：

```bash
python3.12 -m venv .venv-export
PY="$PWD/.venv-export/bin/python"
"$PY" -m pip install --upgrade pip
"$PY" -m pip install -r requirements-export.txt
"$PY" -m pip install -r requirements-baker.txt
```

`requirements-export.txt` 包含训练基础依赖、TensorFlow、Vela；`requirements-baker.txt` 追加 Parquet 和中文 CTC 对齐所需依赖。只做英文任务可以不安装 Baker 依赖。

本次主要环境：

| 项目 | 记录 |
|---|---|
| 机器 | Linux aarch64，DGX Spark / NVIDIA GB10，20 核 CPU |
| Python | 3.12.3 |
| PyTorch | 2.12.0，CUDA 13.0 构建 |
| TensorFlow | 2.20.0，本次导出/推理使用 CPU |
| NumPy / SciPy | 1.26.4 / 1.14.1 |
| Transformers | 4.57.6 |
| Vela | 5.2.0 |

依赖文件不是完全锁定的环境快照；曾遇到 TensorFlow 2.20 与基础依赖中的 TensorBoard 2.21 版本要求冲突。已有可用环境不要为重现步骤盲目升级 Torch/CUDA；新环境应检查 `"$PY" -m pip check`，解决依赖冲突后再使用。

### 2.2 权重、配置与 JSON

- 英文声学权重：`checkpoints/grainspeech_l1_ssim_gvar.ckpt`。
- 英文统计量：`configs/LJSpeech/stats.json`。
- 原始 HiFi-GAN：`hifigan/LJ_V2/generator_v2`，**原本就在仓库中**，不是本次另找的外部声码器。
- HiFi-GAN 导出需要与权重匹配的原始配置；默认读取权重旁的 `config.json`。
- `.tflite` 旁边的同名 `.json` 是**导出元数据**，不是原始 HiFi-GAN 配置。

本次真实 HiFi-GAN 导出由用户在本地执行。本文不复述原始配置内容；相关命令应在有权访问该权重和配置的本地环境执行。普通 TFLite 推理不需要原始声码器配置。

两个 TFLite 模型和各自 JSON 应一起保存。JSON 包含形状、音频参数、hash 等信息；中文声学 JSON 还包含词表和拼音映射。不要混用不同导出批次的模型与 JSON。

训练权重、数据、`outputs/` 和 `lightning_logs/` 默认不纳入 Git；提交代码和文档不会自动备份这些产物。

## 3. 英文 TTS 完整流程

### 3.1 使用已发布权重建立浮点参考

如果仅做导出和试听，**不需要重新训练英文模型**。

普通英文文本需要一次性安装 NLTK 资源：

```bash
"$PY" -m nltk.downloader \
  averaged_perceptron_tagger averaged_perceptron_tagger_eng cmudict

"$PY" grainspeech/infer.py \
  --checkpoint checkpoints/grainspeech_l1_ssim_gvar.ckpt \
  --text "Small models can give every word a voice." \
  --device cpu \
  --output outputs/english_pytorch.wav
```

直接输入 ARPAbet 可以绕过英文 G2P：

```bash
"$PY" grainspeech/infer.py \
  --phonemes "G R EY1 N S P IY1 CH" \
  --device cpu \
  --output outputs/english_phonemes.wav
```

原 PyTorch 入口需要原始声码器文件。复现 README 的两句参考 Mel/WAV：

```bash
"$PY" scripts/generate_readme_examples.py --device cpu
```

产物在 `assets/examples/`，包含 `.npy` Mel 和输入/权重来源清单；它也是英文导出回归的参考。

### 3.2 可选：从 LJSpeech 训练英文模型

需要重新训练时，准备：

```text
data/LJSpeech-1.1/
├── metadata.csv
├── wavs/
└── TextGrid/LJSpeech/*.TextGrid
```

LJSpeech 数据及配套音素 TextGrid 的下载入口见仓库 [README](../README.md#grainspeech-training)。TextGrid 负责提供音素时长监督，不能仅用 WAV 和文本均分时长代替。

```bash
"$PY" grainspeech/preprocess.py \
  --preprocess-config configs/LJSpeech/preprocess.yaml \
  --textgrid-dir data/LJSpeech-1.1/TextGrid \
  --device auto

"$PY" scripts/check_setup.py

"$PY" grainspeech/train_l1_ssim_gvar.py \
  --run-name grainspeech-l1-ssim-gvar \
  --preprocess-config configs/LJSpeech/preprocess.yaml \
  --hifigan-checkpoint hifigan/LJ_V2/generator_v2 \
  --accelerator gpu --devices 1 --precision 16-mixed \
  --batch-size 128 --num_workers 4 --max_epochs 5000 \
  --lr 0.001 --weight-decay 0.00001 --infer-device cuda
```

这是仓库英文训练入口示例，不是本次重跑过的英文训练任务。上述批大小和 GPU 配置需按机器调整。发布的 inference-only checkpoint 不可当作完整训练状态续训；续训使用训练产生的 `last.ckpt`。

### 3.3 导出 FP32 TFLite

导出器不是直接追踪整个 PyTorch TTS，而是用 TensorFlow 重建声学网络，再转换成固定形状 TFLite。

```bash
"$PY" scripts/export_tflite.py \
  --checkpoint checkpoints/grainspeech_l1_ssim_gvar.ckpt \
  --stats configs/LJSpeech/stats.json \
  --float32 \
  --max-phonemes 128 \
  --max-mel-frames 512 \
  --output outputs/grainspeech_fp32_reference.tflite
```

### 3.4 英文声学量化：快速校准与真实校准

**本次最初使用的快速路径**：未传 `--calibration-data` 时，英文导出使用固定随机种子生成合法英文 ID 序列，默认 64 组。

```bash
"$PY" scripts/export_tflite.py \
  --checkpoint checkpoints/grainspeech_l1_ssim_gvar.ckpt \
  --stats configs/LJSpeech/stats.json \
  --max-phonemes 128 \
  --max-mel-frames 512 \
  --calibration-samples 64 \
  --output outputs/grainspeech_a16w8.tflite
```

这适合快速检查转换是否工作，但随机 ID 不代表真实语言分布，不能作为最终语音质量保证。

**推荐的真实校准路径**：使用 LJSpeech 预处理生成的训练清单：

```bash
"$PY" scripts/export_tflite.py \
  --checkpoint checkpoints/grainspeech_l1_ssim_gvar.ckpt \
  --stats configs/LJSpeech/stats.json \
  --calibration-data data/LJSpeech-1.1/preprocessed_data/LJSpeech/train.txt \
  --calibration-samples 256 \
  --max-phonemes 128 \
  --max-mel-frames 512 \
  --output outputs/grainspeech_en_realcal_a16w8.tflite
```

清单格式与训练 metadata 相同：

```text
样本ID|说话人|{空格分隔的 ARPAbet 音素}|原始文本
```

导出校准只读取第三列，不需要该清单对应的 WAV 或 Mel 文件。没有完整 LJSpeech 时，也可以用目标场景文本经过**同一个英文前端**生成清单。以下示例使用文本文件，每行一句：

```bash
# 先自行准备 outputs/english_calibration_sentences.txt，覆盖真实目标场景。
"$PY" - <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, "grainspeech")
from text.frontend import text_to_arpabet

source = Path("outputs/english_calibration_sentences.txt")
rows = []
for index, line in enumerate(source.read_text(encoding="utf-8").splitlines()):
    text = line.strip()
    if not text:
        continue
    if "|" in text:
        raise ValueError("Calibration text must not contain the metadata separator |")
    phones = text_to_arpabet(text)
    rows.append(f"en-{index:05d}|target|{{{' '.join(phones)}}}|{text}\n")
if not rows:
    raise ValueError("No calibration sentences")
Path("outputs/english_calibration.txt").write_text("".join(rows), encoding="utf-8")
PY
```

然后把 `--calibration-data` 指向 `outputs/english_calibration.txt`。

真实校准会优先覆盖最短、最长以及最大 token ID 的样本，再用固定种子补充采样；超过输入容量的行被排除并报告。实际校准数量不超过可用行数，不会因为设置了 256 就把两句话变成 256 句。元数据记录样本 ID、源文件 hash、实际数量和 token 覆盖信息。

### 3.5 验证声学导出并准备声码器校准 Mel

对默认英文 FP32/A16W8 文件执行：

```bash
"$PY" scripts/check_tflite_export.py
```

得到：

```text
outputs/a16w8_comparison/
├── compact-speech_fp32.npy
├── compact-speech_a16w8.npy
├── morning-light_fp32.npy
├── morning-light_a16w8.npy
└── comparison.json
```

脚本检查查表、累计长度、帧归属、padding 和算子类型。自定义模型路径等参数用 `--help` 查看。

这四个 Mel 文件是本次 HiFi-GAN 校准的最小可运行数据集；它们只是两句文本的两种声学输出，不应误称为充分覆盖所有英文场景的大型校准集。下一章说明如何扩大数据集。

### 3.6 英文双 TFLite 端到端推理

完成下一章的声码器导出后：

```bash
"$PY" scripts/infer_tflite.py \
  --text "Small models can give every word a voice." \
  --acoustic-model outputs/grainspeech_a16w8.tflite \
  --vocoder-model outputs/hifigan_a16w8.tflite \
  --threads 2 \
  --save-mel outputs/english_end_to_end.npy \
  --output outputs/english_end_to_end.wav
```

模型路径是默认值时可以省略。`--text` 可换成 `--phonemes`。此路径不加载 PyTorch 权重，不读取原始 HiFi-GAN 配置。

## 4. HiFi-GAN 导出、量化与校准数据

### 4.1 来源与是否需要另训中文声码器

本次使用 `hifigan/LJ_V2/generator_v2`，其 SHA-256 为：

```text
3fac378c5918fb2c102733f21eeaa8e9a4ca6cda24dbfddc55bbb947c78d562f
```

它不直接接收语言文字，因此可以先尝试跨语言复用。中文阶段先做了：

```text
真实中文录音 → 相同参数的 Mel → 已有 A16W8 HiFi-GAN → 重建音频
```

用户认为三句重建效果很好，因此本次**只训练中文 GrainSpeech，没有重新训练 HiFi-GAN**。这不等于保证它适配任何语言、说话人和音高范围。

### 4.2 声码器校准输入要求

每个文件为 `.npy`，必须满足：

| 项目 | 当前要求 |
|---|---|
| 形状 | 非空 `[frames,80]`；不要存成 `[80,frames]` 或 `[1,frames,80]` |
| 类型 | 有限浮点数，通常 float32 |
| 表示 | 自然对数 Mel 幅度，已经反量化，无额外归一化 |
| 当前音频参数 | 22050 Hz，FFT/window 1024，hop 256 |
| Mel 参数 | 80 维，fmin 0，fmax 8000；具体提取实现也应匹配 |

80 维相同并不代表两个项目的 Mel 一定兼容。不要混入其他采样率、power、频率范围、对数底或归一化方式生成的缓存。

可以使用：

- 实际部署的声学模型预测的有效 Mel，尤其是量化声学模型的输出。
- 相同预处理生成的真实录音 Mel，用于覆盖自然语音范围。
- 两者混合，但要明确来源，避免少量重复样本冒充覆盖度。

校准目录只扫描**当前目录**的 `*.npy`，不递归。导出器先将这些 Mel 载入内存，再生成窗口；不要把整个大型语料的 Mel 目录不加选择地传入。

### 4.3 最小复现：使用英文对比 Mel

先完成 3.5，确保 `outputs/a16w8_comparison/` 下存在四个 `.npy`。

**FP32 声码器：**

```bash
"$PY" scripts/export_hifigan_tflite.py \
  --checkpoint hifigan/LJ_V2/generator_v2 \
  --config hifigan/LJ_V2/config.json \
  --float32 \
  --mel-frames 64 \
  --calibration-dir outputs/a16w8_comparison \
  --calibration-samples 64 \
  --output outputs/hifigan_fp32.tflite
```

即使指定 `--float32`，**当前脚本仍需要 Mel 目录**，用于 PyTorch/TensorFlow/TFLite 数值参考比较；这时不执行量化校准。

**A16W8 声码器：**

```bash
"$PY" scripts/export_hifigan_tflite.py \
  --checkpoint hifigan/LJ_V2/generator_v2 \
  --config hifigan/LJ_V2/config.json \
  --mel-frames 64 \
  --calibration-dir outputs/a16w8_comparison \
  --calibration-samples 64 \
  --output outputs/hifigan_a16w8.tflite
```

`--calibration-samples` 是**窗口数，不是音频文件数**。当前脚本要求它至少等于目录里的 `.npy` 文件数，以便每个文件都被覆盖。

窗口生成策略：在文件间轮换，对每条 Mel 的可用起点从头到尾均匀取窗；短于固定窗口的输入右侧补零。64 个窗口不一定都是不同的，尤其在样本过少或很短时。

### 4.4 扩大校准数据：以中文训练 Mel 为例

中文特征生成后，可以从**训练划分**选择一批 Mel，单独准备校准目录。下面脚本选择最多 128 条，保留最短/最长样本并随机补充，使用绝对路径软链接避免重复复制：

```bash
"$PY" - <<'PY'
from pathlib import Path
import json
import numpy as np

root = Path("data/Baker-features").resolve()
out = Path("outputs/hifigan_baker_calibration")
if out.exists() and any(out.iterdir()):
    raise FileExistsError("Use a fresh calibration directory")
records = []
for line in (root / "train.txt").read_text(encoding="utf-8").splitlines():
    name, speaker, _, _ = line.split("|")
    path = root / "mel" / f"{speaker}-mel-{name}.npy"
    mel = np.load(path, mmap_mode="r", allow_pickle=False)
    if mel.ndim != 2 or mel.shape[1] != 80 or not len(mel):
        raise ValueError(path)
    records.append((path, len(mel)))
if not records:
    raise ValueError("No training Mels")
count = min(128, len(records))
selected = list(dict.fromkeys([
    min(range(len(records)), key=lambda i: records[i][1]),
    max(range(len(records)), key=lambda i: records[i][1]),
]))[:count]
rng = np.random.default_rng(1234)
remaining = [i for i in range(len(records)) if i not in selected]
selected += rng.choice(remaining, count - len(selected), replace=False).tolist()
out.mkdir(parents=True, exist_ok=True)
for index in selected:
    path, _ = records[index]
    (out / path.name).symlink_to(path)
(out / "selection.json").write_text(json.dumps({
    "seed": 1234, "source": str(root / "train.txt"),
    "files": [str(records[i][0]) for i in selected],
}, indent=2) + "\n")
print(f"Prepared {len(selected)} Mel files in {out}")
PY
```

用它导出一个**新名字**的声码器，保留已试听的版本：

```bash
"$PY" scripts/export_hifigan_tflite.py \
  --checkpoint hifigan/LJ_V2/generator_v2 \
  --config hifigan/LJ_V2/config.json \
  --mel-frames 64 \
  --calibration-dir outputs/hifigan_baker_calibration \
  --calibration-samples 512 \
  --output outputs/hifigan_baker_calibrated_a16w8.tflite
```

这是扩大中文校准覆盖的可选复现方案，**不是本次已经生成并试听的声码器来源**。本次中英文试听使用的仍是四个英文预测 Mel 校准的 `outputs/hifigan_a16w8.tflite`。

更贴近实际链路时，可以逐句用 `scripts/infer_tflite.py --save-mel` 收集不同文本的量化声学输出，或用 `scripts/infer_baker.py` 收集中文浮点预测 Mel，再放入专门的校准目录。只保留有效帧，并另外保留未用于校准的试听句子。

### 4.5 导出检查及分块推理

导出脚本会：

1. 加载 PyTorch generator 并折叠 weight normalization。
2. 生成 PyTorch 参考波形。
3. 比较 TensorFlow 重建的 FP32 输出。
4. 转换 FP32/A16W8 TFLite。
5. 检查 TFLite 接口、卷积类型和首窗口误差，写出同名 JSON。

只用 TFLite 声码器生成音频：

```bash
"$PY" scripts/infer_hifigan_tflite.py \
  --model outputs/hifigan_a16w8.tflite \
  --mel-dir outputs/a16w8_comparison \
  --output-dir outputs/hifigan_a16w8_audio
```

单文件或多个指定文件使用 `--mel file1.npy file2.npy`。FP32 声码器对照只需更换 `--model` 和输出目录。

**不能把独立窗口的全部波形直接拼接。** 当前实际导出需要左右各 13 帧上下文；64 帧窗口在内部位置通常保留 38 帧的中心有效部分。推理函数自动重叠分块、裁掉上下文，并将首尾窗口锚定到真实句子边界。

短于窗口的句子会补零、裁剪并发出 warning，尾部可能与变长 PyTorch 推理不同。若专门研究某条短句的边界，可以把 `--mel-frames` 设为它的实际长度重新导出。

## 5. 中文 TTS 完整流程

### 5.1 复用的数据究竟是什么

用户原项目：`~/MOSS-TTS-Nano/train_full_baker.py`，其中训练读取：

```text
~/datasets/baker/processed_shards/shard_*.pt
```

这些 shard 是 MOSS 使用的音素 ID 和神经音频 codec codes，**不能直接作为 GrainSpeech 的 Mel 训练目标**。

可复用的原始数据是：

```text
~/datasets/baker/data/train-00000-of-00008.parquet
...
~/datasets/baker/data/train-00007-of-00008.parquet
```

8 个文件，每个 1250 条，共 10000 条。关键列为 `file_name`、`text`、`text_pinyin`、`audio.bytes`、`audio.path`；原始音频为 48 kHz 单声道 PCM16。

旧项目的 CTC 缓存存在对齐失败后回退均分时长、缺少逐条失败标志和来源 ID 等问题；其特征关联还可能受过滤后的记录顺序影响。因此本次重新从原始录音和标注生成对齐，不直接沿用这些 duration/Mel 缓存。

### 5.2 原始数据准备与声码器适配试听

先做小样本：

```bash
"$PY" scripts/prepare_baker.py \
  --corpus-dir ~/datasets/baker/data \
  --output-dir data/Baker-preview \
  --limit 8 --mel-samples 3

"$PY" scripts/infer_hifigan_tflite.py \
  --mel-dir data/Baker-preview/mel \
  --output-dir outputs/baker_vocoder_preview
```

这一步试听的是**真实录音 Mel 的重建**，不是已经训练好的中文 TTS。

全量准备：

```bash
"$PY" scripts/prepare_baker.py \
  --corpus-dir ~/datasets/baker/data \
  --output-dir data/Baker
```

输出包含：

```text
data/Baker/
├── raw/Baker/*.wav
├── raw/Baker/*.lab
├── manifest.jsonl
└── dataset.json
```

WAV 转为 22050 Hz 单声道、峰值归一化到 0.95。清单保留原始文本、带调拼音、来源 shard/row 和原始 ID；`#1`–`#4` 韵律标记只从清理后的文本移除，原始标注仍保留。

原始数据目录不改动，非空输出目录会被拒绝。本次准备了 10000 条，约 11.86 小时。此阶段 `training_ready: false` 是正常的，因为还没有对齐和训练特征。

### 5.3 中文 token 设计与自动对齐

第一版采用**完整带调拼音音节**，例如 `ni2`、`hao3`、`menr2`，而不是汉字 ID 或声母/韵母拆分。声调和儿化保留在 token 中；`er2` 本身不是附加儿化。

没有采用 native MFA 的原因是本次 Linux ARM64 环境缺少可直接使用的相关依赖，且不同 Mandarin 字典与声学模型不能任意配对。最终使用本地 Transformers Mandarin CTC：

```text
模型：wbbbbb/wav2vec2-large-chinese-zh-cn
revision：b654b5f3df69725df32808de454a1b865c829e4c
权重格式：safetensors
缓存：external/alignment-models/
```

`external/alignment-models/` 是 Hugging Face 模型下载缓存，不是生成的中文语料，也不是最终 GrainSpeech 或 HiFi-GAN 权重。目录名中的 `models--...` 对应缓存的模型仓库；`ls -lh` 显示的 12K 只是目录项大小，不是权重占用。本机三个目录的用途如下：

| 缓存模型 | 与本次全量 Baker 对齐的关系 |
|---|---|
| `models--wbbbbb--wav2vec2-large-chinese-zh-cn` | **实际使用**：`data/Baker-alignments/run.json` 记录的模型及上述固定 revision。 |
| `models--jonatasgrosman--wav2vec2-large-xlsr-53-chinese-zh-cn` | 下载过的另一候选中文识别模型；**没有用于本次全量对齐**。仅凭缓存目录不能判断其是否用于其他试验。 |
| `models--qinyue--wav2vec2-large-xlsr-53-chinese-zn-cn-aishell1` | 下载过的另一候选中文识别模型；**没有用于本次全量对齐**。仅凭缓存目录不能判断其是否用于其他试验。 |

对齐器读取 `data/Baker/` 中**已有的录音和文字/带调拼音标注**，用 CTC 预测汉字在音频中的位置，再转为拼音音节时长；它不生成语料文字、不训练 GrainSpeech，也不参与 TFLite 量化或最终合成。对齐结果保存在 `data/Baker-alignments/`，随后由 `scripts/preprocess_baker.py` 转成 `data/Baker-features/` 供声学模型训练。已有特征、checkpoint 和 TFLite 推理不依赖这些下载缓存；若需要重新对齐，则需要保留或重新下载对应的固定 revision。`external/` 被 Git 忽略，没有随项目数据与模型产物提交；不要将缓存目录误认为已上传的训练权重。

模型页面声明 Apache-2.0；部署分发前仍应检查所用模型、数据集及依赖的实际许可证。本次音频和文本只在本地处理，不上传外部识别服务。

先做代表性随机抽样可用 `--sample-size 64`；`--limit 64` 只取开头，未必代表整个语料。全量命令：

```bash
OMP_NUM_THREADS=4 "$PY" scripts/align_baker.py \
  --dataset data/Baker \
  --output-dir data/Baker-alignments \
  --disable-cudnn
```

对齐过程在 16 kHz 输入上推理，但训练录音仍保留 22050 Hz。CTC 对齐汉字后与人工拼音对应，显式的“汉字 + 儿”可合并为儿化音节。字符发射之间的空白不等于静音，当前在相邻跨度间取中点作为音节分界，首尾使用录音边界。

因此这些是**自动估计的音节时长，不是真实人工音素边界**，不会把音节时间再均分成声母/韵母时间。

默认筛选：

- 词表外字符、混合脚本或文本/拼音关联错误：拒绝。
- 贪心识别 CER 超过 0.35：拒绝。
- 最低字符置信度低于 0.01：拒绝。
- 不存在合法完整 CTC 路径：拒绝，不造均分时长。

逐条结果写入 `records/<id>.json`，接受样本写入 `TextGrid/Baker/`，汇总在 `summary.json`。中断后同命令加 `--resume`；源数据、模型、采样范围和阈值必须相同，不能把已完成的小样本目录直接当作全量运行续跑。

本次全量结果：

| 项目 | 条数 |
|---|---:|
| 接受 | 6027 |
| 拒绝：最低字符置信度不足 | 3107 |
| 拒绝：CTC 词表外字符 | 526 |
| 拒绝：CER 超阈值 | 338 |
| 拒绝：不支持的非汉字字符 | 2 |

接受音频约 6.85 小时。拒绝不都是生僻字，也不直接表示录音质量差；这是对齐器覆盖与筛选策略共同造成的。不要为了提高数量随意降低阈值。

### 5.4 训练特征生成

```bash
OMP_NUM_THREADS=4 "$PY" scripts/preprocess_baker.py \
  --dataset data/Baker \
  --alignments data/Baker-alignments \
  --output-dir data/Baker-features \
  --disable-cudnn
```

产生 `mel/`、`pitch/`、`energy/`、`duration/`、`train.txt`、`val.txt`、`stats.json`、`speakers.json`、`preprocess.yaml`、`rejected.json` 和新的 `dataset.json`。

关键约束：

- 音节数与 duration/pitch/energy 长度一致。
- duration 总和等于 Mel 帧数。
- 所有特征有限，单个音节不能短于一个有效 Mel 帧。
- 统计量只在训练划分拟合，避免验证数据泄漏。
- 失败项明确记录；无有效音高等原因可能进一步减少样本。

本次最终训练 5771 条、验证 256 条。

此脚本目前逐条提取，没有多录音并行 worker 参数，也不支持特征生成断点续跑。20 核机器可以尝试 `OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8`，但这只影响支持线程的库，不意味着并行处理 8 条录音。已经运行时不建议只为改线程数中断重做。

### 5.5 训练、续训和选择 checkpoint

```bash
OMP_NUM_THREADS=4 "$PY" scripts/train_baker.py \
  --preprocess-config data/Baker-features/preprocess.yaml \
  --run-name grainspeech-baker \
  --accelerator gpu --batch-size 16 --workers 4 \
  --epochs 1000 --precision 32-true \
  --disable-cudnn
```

从随机初始化训练中文声学模型，目标为 L1 + SSIM + GVar，并包含 pitch、energy、duration 监督。HiFi-GAN 不参与训练，也不需要加载其原始配置。

续训重复命令并加：

```bash
--checkpoint lightning_logs/grainspeech-baker/checkpoints/last.ckpt
```

词表、预处理参数和归一化统计不能在续训时更换。每次运行的 CSV 指标在独立的 `version_*` 目录，旧日志不覆盖；最优及最后 checkpoint 在 `checkpoints/`。

小规模跑通流程可用独立 `--run-name` 加 `--max-steps 4`，这种 checkpoint 只能检查代码通路，不能用于判断语音效果。

本次 1000 轮训练：

| 权重 | 零基 epoch | 验证损失 |
|---|---:|---:|
| 最优 `epoch=129-step=46930-val_loss=12.9407.ckpt` | 129 | 12.9407 |
| 最后 `last.ckpt` | 999 | 14.7772 |

后续量化选择最优权重，而不是因为 `last.ckpt` 更新更晚就优先选它。训练损失下降不保证验证效果或听感一直改善。

### 5.6 中文 PyTorch Mel 与浮点试听参考

```bash
BEST=lightning_logs/grainspeech-baker/checkpoints/epoch=129-step=46930-val_loss=12.9407.ckpt

"$PY" scripts/infer_baker.py \
  --checkpoint "$BEST" \
  --pinyin "ni2 hao3 huan1 ying2 shi3 yong4 zhong1 wen2 yu3 yin1 he2 cheng2" \
  --output outputs/baker_trained_mels/welcome_best.npy

"$PY" scripts/infer_baker.py \
  --checkpoint "$BEST" \
  --pinyin "jin1 tian1 de5 tian1 qi4 hen2 hao3" \
  --output outputs/baker_trained_mels/weather_best.npy

"$PY" scripts/infer_baker.py \
  --checkpoint "$BEST" \
  --pinyin "wo3 men5 zheng4 zai4 xun4 lian4 yi2 ge4 xiao3 xing2 zhong1 wen2 yu3 yin1 mo2 xing2" \
  --output outputs/baker_trained_mels/training_best.npy

"$PY" scripts/infer_hifigan_tflite.py \
  --mel-dir outputs/baker_trained_mels \
  --output-dir outputs/baker_trained_audio
```

输入应为预期实际发音的声调，例如这里用 `ni2 hao3`。前端不会自动推断汉字、多音字或变调。Mel 推理与 TFLite 声码器在两个进程运行，避免不必要的 PyTorch/TensorFlow 混用。

### 5.7 中文声学校准数据准备

直接使用已经生成的：

```text
data/Baker-features/train.txt
```

每行第三列是词表匹配的音节：

```text
样本ID|Baker|{ni2 hao3 ...}|原始中文文本
```

**不能使用英文 ID 1–73 的随机校准来替代。** 当前导出器要求自定义词表 A16W8 导出显式提供 `--calibration-data`，从 checkpoint 读取中文词表和特征统计量，并检查 embedding 大小。

本次使用 256 条真实训练序列，输入容量 128，没有因输入过长排除样本。最大 ID、最长/最短序列及固定种子采样策略与英文真实校准相同。

校准清单本身不需要 WAV/Mel，但它的 token 必须来自训练时的同一词表。不要更换词表顺序、重新从汉字生成另一套拼音，再沿用旧模型权重。

### 5.8 中文 FP32 与 A16W8 TFLite 导出

```bash
"$PY" scripts/export_tflite.py \
  --checkpoint "$BEST" \
  --float32 \
  --max-phonemes 128 \
  --max-mel-frames 512 \
  --output outputs/grainspeech_baker_fp32.tflite

"$PY" scripts/export_tflite.py \
  --checkpoint "$BEST" \
  --calibration-data data/Baker-features/train.txt \
  --calibration-samples 256 \
  --max-phonemes 128 \
  --max-mel-frames 512 \
  --output outputs/grainspeech_baker_a16w8.tflite
```

无需再指定英文统计量；导出器使用 checkpoint 内的中文 `feature_stats`。若显式传 `--stats`，它必须匹配该中文 checkpoint。

输出使用独立名字，**不要覆盖英文模型**。本次 A16W8 文件为 924824 字节，约 903.15 KiB；FP32 文件为 1595360 字节。由于 embedding、CPU 辅助运算和元数据等并非全部 int8，不能按全部权重缩小四倍来估算文件大小。

中文音节词表比英文更大，本次中文声学网络约 387K 参数，不能继续套用原英文模型 264.8K 的参数量或编译内存。

### 5.9 双 A16W8 推理与量化试听

```bash
"$PY" scripts/infer_tflite.py \
  --acoustic-model outputs/grainspeech_baker_a16w8.tflite \
  --vocoder-model outputs/hifigan_a16w8.tflite \
  --pinyin "ni2 hao3 huan1 ying2 shi3 yong4 zhong1 wen2 yu3 yin1 he2 cheng2" \
  --threads 2 \
  --save-mel outputs/baker_quantized_mel.npy \
  --output outputs/baker_quantized.wav
```

默认读取模型旁同名 JSON，也可以用 `--acoustic-metadata` / `--vocoder-metadata` 显式指定。脚本检查 hash、词表、形状及采样率/hop，拒绝把英文前端接到中文模型。

生成本次三组成对试听：

```bash
"$PY" scripts/check_baker_tflite.py
```

依赖 5.6 的三条 `*_best.npy` 浮点参考和 5.8 的两个 TFLite 文件。输出：

```text
outputs/baker_a16w8_audio/
├── welcome_fp32.wav
├── welcome_a16w8.wav
├── weather_fp32.wav
├── weather_a16w8.wav
├── training_fp32.wav
├── training_a16w8.wav
├── mel/
└── comparison.json
```

这里的 `fp32` / `a16w8` 指**声学模型精度**，两组都使用相同的 A16W8 HiFi-GAN。对比时才不会把声码器变化误认为声学量化变化。

## 6. 本次问题与修复记录

### 6.1 英文预测长度从 207 帧变成约 45 帧

最初 A16W8 导出有两个独立问题：

1. **查表失败**：输入 ID 经量化/反量化后不再精确，用浮点 `Equal` 对比 ID 会导致 embedding 查不到，甚至变为全零。
2. **总长度饱和**：int16 `SUM` 沿用了单音素 duration 的量化 scale，整句累计长度在约 45 帧处饱和。

修复保持一个图，不拆成多个声学模型：

- 对 ID 做 `round`、转 int32，再 one-hot 查表；TensorFlow 2.20 折叠为 CPU `EMBEDDING_LOOKUP`。
- 固定累计 duration 和 gather 代替动态 `repeat_interleave`。
- CPU `CUMSUM` 得到完整累计时长，用最大累计值取整句长度，绕开 int16 `SUM`。
- 用帧中心 `frame_index + 0.5` 比较归属，降低量化边界误差。
- bucketize 对应比较使用 `>`，与 PyTorch `right=False` 一致。

修复后，两句英文分别为 207→208 帧、257→257 帧，不再有约 45 帧截断。

**调试注意**：TFLite 中间张量默认复用内存，读取没有保留的中间 buffer 可能看到垃圾值。排查时使用 `experimental_preserve_all_tensors=True`；不要据此错误判断图中出现 NaN/异常量化。

### 6.2 HiFi-GAN 在 Removing weight norm 后段错误

实际问题在随后 PyTorch 参考推理的优化 Conv1d 路径。ARM64 下同进程加载 TensorFlow/PyTorch 时可复现；仅 PyTorch 正常。

导出器在参考比较范围内使用：

```python
with torch.backends.mkldnn.flags(enabled=False), torch.inference_mode():
    ...
```

作用域结束恢复后端，不修改导出模型或量化策略。新增五阶段进度日志及 fault handler。仍需定位 native 错误时，可由用户运行：

```bash
"$PY" -X faulthandler scripts/export_hifigan_tflite.py \
  --calibration-dir outputs/a16w8_comparison \
  --output outputs/hifigan_debug_a16w8.tflite
```

模型重建还必须正确处理转置卷积权重布局、weight normalization、两种残差块，以及**最后一层 LeakyReLU slope=0.01，其他相关层为 0.1**。

### 6.3 DGX Spark 的 cuDNN 版本冲突

中文对齐/训练曾出现：

```text
CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH
```

`--disable-cudnn` 是显式规避选项：用 PyTorch 原生 CUDA 卷积，**仍然使用 GPU**，不改系统 CUDA 库。它与上一节 CPU MKLDNN 问题不同。

健康环境可省略此选项；对齐/预处理可用 `--device cpu`，训练可用 `--accelerator cpu`。`scripts/export_tflite.py` 没有该选项，不要照搬给所有脚本。

### 6.4 其他常见提示

| 提示 | 含义与处理 |
|---|---|
| 缺少 `averaged_perceptron_tagger_eng` | 英文 NLTK 资源未装齐，不是声学模型错误；运行 3.1 的 downloader |
| `weights_only was not set ... False` | Lightning 保存完整 checkpoint 的提示，不等于训练失败；加载完整 checkpoint 只用可信本地文件 |
| `tf.lite.Interpreter` 弃用警告 | 当前环境仍可运行；将来迁移 LiteRT 需重新回归，不要仅为消除 warning 改推理链路 |
| `Rejected ...` | 对齐显式筛选，不是崩溃；查看逐条原因及汇总 |
| hash mismatch | TFLite 与 JSON 不配套，不要手工改 hash 绕过 |
| 输出容量达到 512 | 可能截断，应缩短输入或重新导出更大容量 |
| HiFi-GAN 短输入 padding warning | 固定窗口补零边界与变长推理不同，不表示整句无法合成 |

## 7. Ethos-U55 部署边界

### 7.1 Vela 是下一步，不是桌面 TFLite 推理的替代品

英文声学编译示例：

```bash
vela --accelerator-config ethos-u55-128 \
  --memory-mode Shared_Sram \
  --system-config Ethos_U55_High_End_Embedded \
  --show-cpu-operations \
  --output-dir outputs/vela \
  outputs/grainspeech_a16w8.tflite
```

中文声学和声码器分别编译：

```bash
vela --accelerator-config ethos-u55-128 \
  --memory-mode Shared_Sram \
  --system-config Ethos_U55_High_End_Embedded \
  --show-cpu-operations \
  --output-dir outputs/vela_baker \
  outputs/grainspeech_baker_a16w8.tflite

vela --accelerator-config ethos-u55-128 \
  --memory-mode Shared_Sram \
  --system-config Ethos_U55_High_End_Embedded \
  --show-cpu-operations \
  --output-dir outputs/vela_hifigan \
  outputs/hifigan_a16w8.tflite
```

若 Vela 安装在上述虚拟环境但未激活，用 `$(dirname "$PY")/vela` 替换 `vela`。U55 MAC 配置、memory mode 和 system config 应按实际 MCU 修改，不能因为有 U55 就假定一定是 U55-128。

桌面试听用普通导出的 `.tflite`；Vela 编译后的 Ethos-U 模型需要对应 runtime/delegate，不能直接当作普通 CPU TFLite 使用。

### 7.2 已知结果与尚未完成的工作

本次修复后的**英文** 128/512 模型，在 Vela 5.2.0、U55-128 示例配置下曾报告：

- 167 个 NPU operator，41 个 CPU operator。
- SRAM 约 664.09 KiB，off-chip flash 约 254.16 KiB。

这是特定模型和编译配置的估计，不是固件总 RAM，也不是中文模型或 HiFi-GAN 的数据。

MCU 部署仍需检查：

- TFLite Micro/Ethos-U runtime 是否提供剩余 CPU 算子及其 float/int16 变体，尤其查表、累计和索引。
- 声码器 `TRANSPOSE_CONV` 等算子的实际委派情况。
- 模型常量、tensor arena、两个模型之间的 Mel 缓冲、分块上下文、音频输出和固件内存。
- 输入容量、Mel 容量、声码器窗口大小对内存与时延的影响。
- 全链路耗时、实时率、音频连续输出、板端与桌面数值差异。

允许部分算子在 CPU 上运行，并不表示任意 MCU runtime 自动拥有桌面 CPU 的所有 fallback 内核。HiFi-GAN 在波形采样率上运行的中间激活尤其可能占用大量内存。

## 8. 产物、结果与复现检查

### 8.1 本次主要模型与试听目录

| 路径 | 用途 |
|---|---|
| `outputs/grainspeech_a16w8.tflite` | 修复后的英文声学 A16W8 |
| `outputs/grainspeech_fp32_reference.tflite` | 英文声学 FP32 对照 |
| `outputs/hifigan_a16w8.tflite` | 本次中英文共同使用的 A16W8 声码器 |
| `outputs/grainspeech_baker_fp32.tflite` | 第 129 轮中文声学 FP32 |
| `outputs/grainspeech_baker_a16w8.tflite` | 同权重量化的中文声学 A16W8 |
| `outputs/a16w8_comparison/` | 英文声学 Mel 对照及 HiFi-GAN 初始校准来源 |
| `outputs/a16w8_audio/` | 英文声学 FP32/A16W8 + PyTorch FP32 声码器 |
| `outputs/hifigan_a16w8_audio/` | 英文声学 FP32/A16W8 + A16W8 声码器 |
| `outputs/baker_vocoder_preview/` | 真实中文录音 Mel 经 A16W8 声码器重建 |
| `outputs/baker_trained_audio/` | 中文 PyTorch 声学模型 + A16W8 声码器 |
| `outputs/baker_a16w8_audio/` | 中文 FP32/A16W8 TFLite 声学成对试听 |

若仍存在 `grainspeech_a16w8_before_hybrid_fix.*`，它是早期错误版本备份，不应用于推理。

### 8.2 中文量化实测

| 句子 | FP32 帧数 | A16W8 帧数 | 同帧 Mel MAE | 音频时长 |
|---|---:|---:|---:|---:|
| 你好，欢迎使用中文语音合成。 | 252 | 252 | 0.034419 | 2.926 秒 |
| 今天的天气很好。 | 147 | 147 | 0.072138 | 1.707 秒 |
| 我们正在训练一个小型中文语音模型。 | 336 | 336 | 0.051046 | 3.901 秒 |

FP32 TFLite 与已保存的 PyTorch Mel 最大绝对误差约不超过 `1.6e-5`。中文 A16W8 的 16 个 `CONV_2D`、7 个 `DEPTHWISE_CONV_2D`、7 个 `FULLY_CONNECTED` 均检查为 int16 激活/int8 权重。三句没有出现早期英文的长度崩塌。

本次 HiFi-GAN 有 74 个 A16W8 `CONV_2D` 和 4 个 A16W8 `TRANSPOSE_CONV`。首校准窗口 TensorFlow/PyTorch 最大误差约 `2.624e-5`，A16W8/PyTorch 波形 MAE 约 `0.01446`；单窗口误差不能替代全句试听。

会话中用户已认可英文量化链路、中文真实 Mel 重建，以及中文训练后 PyTorch 声学输出的可懂度。中文 A16W8 三组试听已经交付；本文不把数值接近等同于已经获得全面主观音质评测结论。

### 8.3 识别本次具体导出文件

| 文件 | SHA-256 |
|---|---|
| `grainspeech_baker_fp32.tflite` | `a50374b4a797fc17570fc1f983ebded2572442e977f33f94c7a4aecbbaca2460` |
| `grainspeech_baker_a16w8.tflite` | `e101218a63247facd1d644052bf8a6dbb6a91bc1c0ceadbd18fe4b8c216418f2` |
| `hifigan_a16w8.tflite` | `7ea6d1442c8657a76d6e84dd96fcb8307615962f7e795320e88b8bce1eb6b847` |

重新导出后 hash 可能改变，应该使用新生成的配套 JSON，不能强求 hash 与本文一致。

### 8.4 有针对性的复现检查

```bash
# 已导出英文模型及端到端链路
"$PY" scripts/check_tflite_export.py
"$PY" scripts/check_infer_tflite.py

# HiFi-GAN 导出实现的合成小网络回归，不依赖原始预训练配置
"$PY" scripts/check_hifigan_export.py

# Baker 导入与 CTC 算法回归
"$PY" scripts/check_baker_data.py
"$PY" scripts/check_baker_alignment.py

# 已导出中文模型、PyTorch参考与实际试听文件
"$PY" scripts/check_baker_tflite.py
```

`scripts/check_chinese_model.py` 覆盖训练、续训、便携 checkpoint 推理及英文词表兼容。当前机器若仍有 cuDNN 问题，可对这次回归显式禁用：

```bash
OMP_NUM_THREADS=4 "$PY" - <<'PY'
import runpy
import sys
import torch
sys.path.insert(0, "scripts")
torch.backends.cudnn.enabled = False
runpy.run_path("scripts/check_chinese_model.py", run_name="__main__")
PY
```

这些检查中有的依赖本次已导出模型、参考 Mel 或数据；不是单靠克隆仓库就都有这些大文件。脚本成功运行只证明对应检查项，不等于音质、长句泛化或 MCU 实时性已经全部验证。

## 9. 脚本索引与参考资料

| 脚本 | 作用 |
|---|---|
| `scripts/export_tflite.py` | 英文/中文声学 FP32、A16W8 导出及真实序列校准 |
| `scripts/export_hifigan_tflite.py` | HiFi-GAN 重建、FP32/A16W8 导出及真实 Mel 校准 |
| `scripts/infer_tflite.py` | 英文文本/ARPAbet 或中文拼音 → 两个 TFLite → WAV |
| `scripts/infer_hifigan_tflite.py` | 已保存有效 Mel → 分块 TFLite 声码器 → WAV |
| `scripts/vocode_tflite_mels.py` | 早期对照：已保存 Mel → PyTorch FP32 HiFi-GAN |
| `scripts/prepare_baker.py` | 原始 Parquet → WAV/lab/来源清单 |
| `scripts/align_baker.py` | 本地 CTC 自动音节对齐与显式筛选 |
| `scripts/preprocess_baker.py` | 接受的对齐 → 特征、划分、训练统计 |
| `scripts/train_baker.py` | 中文声学训练与续训，不加载 HiFi-GAN |
| `scripts/infer_baker.py` | 中文 checkpoint + 带调拼音 → 浮点 Mel |
| `grainspeech/text/vocabulary.py` | checkpoint 局部词表与带调拼音查表 |
| `grainspeech/text/frontend.py` | 英文 G2P 与 ARPAbet 验证 |

参考：

- [GrainSpeech 论文](https://arxiv.org/abs/2609.18856)
- [仓库使用说明](../README.md)
- [Baker 数据与训练说明](../data/README.md)
- [本次最终使用的中文对齐模型](https://huggingface.co/wbbbbb/wav2vec2-large-chinese-zh-cn)

**推荐后续顺序：** 先确认中文 A16W8 听感和更多未见文本，再按目标 MCU 的实际 RAM/NPU 配置编译与测量；自动汉字前端、声码器进一步压缩或微调，应作为明确的新任务分别验证。
