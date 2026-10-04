# MOSS-TTS Delay LoRA 微调

`sft_lora.py` 是独立训练脚本，基于原全参代码实现，不导入或调用 `sft.py`。
数据集和音频预处理仍使用现有 `dataset.py`、`prepare_data.py`。
默认只训练 Qwen3 主干的 q/k/v/o_proj、gate/up/down_proj 的低秩参数；文本与音频 embedding、各输出头冻结。
这是非量化 LoRA，仍需加载完整基础模型；显存占用取决于基础权重、序列长度和激活。

## 安装与训练（Windows 优先）

以下示例以 Windows 10/11、PowerShell 和仓库根目录为准。建议先创建并激活虚拟环境，再按仓库说明安装与当前 CUDA 版本匹配的 PyTorch、Transformers 等运行环境：

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[finetune-lora]"
```

如果 PowerShell 阻止激活脚本，可仅对当前用户执行一次：

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

以下命令直接运行 Python 训练入口，无需启动 `.sh` 脚本。PowerShell 使用反引号 `` ` `` 续行；为便于复制，也可将整条命令写成一行。文档中的相对路径均相对于仓库根目录。

### 数据准备与 LJSpeech 连续短音频合并

数据格式与预处理方式见 [原微调文档](README_zh.md)。如果原始数据是连续编号的 LJSpeech 短音频，可使用 `scripts\merge_ljspeech.py` 合并。推荐用 `--target-dist` 按多个时长区间随机混合合并（上限 90 秒），并保留短尾段。这是微调数据准备的起步配置，不是官方最佳长度；合并脚本的详细参数与用法见下文说明。

依赖：`pip install numpy soundfile`。合并过程在 CPU 上进行，不需要 GPU 或模型。

#### 1. 输入格式与路径规则

输入为 UTF-8（支持 BOM）清单，文件扩展名可以是 `.csv` 或 `.txt`，内容分隔符为 `|`：

```text
成音频文件夹/切片_1.wav|挺全面的哈，
成音频文件夹/切片_2.wav|接下来我们继续。
```

- **路径兼容**：路径兼容正反斜杠。相对路径默认相对清单同目录的 `wavs` 文件夹，例如 `metadata.csv` 中的 `切片_1.wav` 会解析为 `wavs/切片_1.wav`，无需传 `--audio-root`；可用 `--audio-root` 显式覆盖。
- **文本列选择**：标准三列 LJSpeech 可用 `--text-column 2` 选择规范化文本。

#### 2. 合并操作与命令示例

选择一个尚不存在的输出目录，执行全量合并。推荐按多个时长区间随机混合（规则见第 4 节）：

```powershell
python scripts\merge_ljspeech.py `
  --input "D:\dataset\metadata.txt" `
  --output-dir "D:\dataset\merged_mix" `
  --max-seconds 90 --target-dist "5-15:0.15,15-30:0.15,30-60:0.30,60-90:0.40" --seed 42
```

备选：固定目标时长。所有样本都集中在 60 秒左右，缺少短段，推理时用短分段容易与训练分布不匹配，一般不推荐：

```powershell
python scripts\merge_ljspeech.py `
  --input "D:\dataset\metadata.txt" `
  --output-dir "D:\dataset\merged_60s" `
  --target-seconds 60 --max-seconds 90 --min-seconds 0
```

#### 3. 分组规则与连续性

- **连续性判定**：默认保持清单顺序。只有同一输入清单、同一文件夹、文件名末尾数字之前的前缀相同、编号递增 1 且采样率/声道相同，才会拼接。遇到编号缺口或格式变化就另起一组；没有数字编号的文件单独保留。
- **排序选项**：若清单是 1、10、2 这种字典序，可加 `--order natural`（按自然数值排序），但应先确认编号确实代表时间顺序。
- **分组合并上限**：`--max-clips 4` 限制最多四条一组，并非保证每组四条。
- **拼接方式**：直接连接波形，保留原始静音；不重采样、不淡化、不插入静音。输出 float32 WAV，避免额外 PCM 量化，但比 PCM16 更占磁盘。采样率和声道保持原样。默认中文文本直接连接，不添加标点；英文可以用 `--text-joiner " "`。

#### 4. 时长控制与过滤机制

不传 `--target-dist` 时，脚本默认目标时长为 60 秒，最大时长为 90 秒，最小时长为 0 秒（第 2 节的备选命令显式写出了这些默认值）：

- `--target-seconds 60`：累计达到设定秒数就结束当前组，不保证恰好相等，也不会截断原始切片。例如已有 55 秒，加入下一条 20 秒后输出 75 秒。
- `--max-seconds 90`：每组的时长上限。若已有 55 秒，下一条为 40 秒，合计超过 90 秒，则先输出当前 55 秒组，40 秒切片进入下一组。单条原始音频已超过上限时会报错。
- `--min-seconds 0`：默认保留所有短组，包括孤立单条和未达到目标的尾段（尾段是连续分组最后剩余的切片，不是音频末尾的静音）。若仍需沿用短组过滤，可设置 `--min-seconds 15`：不足 15 秒丢弃，恰好 15 秒保留，逐组打印被过滤的来源、条数和时长。

**随机目标时长（多长度混合，推荐）**：`--target-dist` 代替固定的 `--target-seconds`。每开始一个新组时，先按权重抽一个区间，再在区间内均匀抽目标时长。这样一次合并就能得到长短混合的数据，每条音频只出现一次（命令见第 2 节）：

- 格式为 `下限-上限:权重`，用逗号分隔；权重不必加起来等于 1，各区间上限不能超过 `--max-seconds`。不能和 `--target-seconds` 同时使用。
- 权重控制的是**组数**比例；长组更长，所以按时长算长段占比更高。运行后会打印每个区间的目标权重、实际组数占比和时长占比。
- 实际分布会偏离权重：编号缺口和 `--max-seconds` 会让组提前结束，单条原始切片的长度也限制了最短区间。可先加 `--dry-run` 查看分布，再调整权重。
- `--seed` 固定随机结果，默认 42；换一个 seed 再合并一次，可以得到切点不同的另一份数据。
- `--max-seconds` 应覆盖推理时单段的最长时长并留余量。例如推理每段约 1 分钟时，慢语速段可能到 75 秒，用 90 比较稳妥。推理时的分段字数要手动设置成与此匹配的值（约为 1 分钟对应的字数），不要用混合数据统计出的平均字数。

**校验与安全**：
- 过滤发生在 `--limit` 之前；全部被过滤时仅打印提示，不创建输出目录。
- 重复路径、空文本会报错；当前缺失音频文件会被跳过，应核对输入记录与最终选中条数。
- 脚本不做语义分段、说话人或声场检测：同前缀连续编号仍可能不是连续录音，需确认来自同一说话人的连续录音并人工抽查接缝。
- 脚本不会修改源文件，也不会覆盖已有输出目录。中途失败的输出目录保留用于检查；修正问题后使用新目录重跑。`--limit` 仅限制输出条数，仍会检查所有输入文件头。

#### 5. 输出结构

合并后输出目录包含：

- `wavs/`：保留来源相对音频根目录的文件夹层级，以组内首条文件名加 `_merge.wav` 命名。例如 `说话人/xxxx001.wav` 至 `说话人/xxxx003.wav` 合并为 `wavs/说话人/xxxx001_merge.wav`。保留的单条也使用此后缀。根目录外的绝对路径保留直接上级文件夹名；多个输入产生同名输出时会在写入前报错，请分开处理。
- `metadata.txt`：相对输出目录的 `路径|文本`。若将它再次输入本脚本，需显式设置 `--audio-root` 为该输出目录（路径已带 `wavs/`）。
- `train_raw.jsonl`：含绝对音频路径、文本、语言，可直接交给项目的 `prepare_data.py` 重新编码，不含参考音频或时长条件。
- `sources.jsonl`：来源文件、文本、拼接位置（采样帧），方便检查接缝。

#### 6. 音频特征预编码（prepare_data.py）

合并生成 `train_raw.jsonl` 或已有未编码数据后，执行以下命令提取音频 Token 并生成最终训练数据：

```powershell
python moss_tts_delay\finetuning\prepare_data.py `
  --model-path OpenMOSS-Team/MOSS-TTS-v1.5 `
  --codec-path OpenMOSS-Team/MOSS-Audio-Tokenizer `
  --input-jsonl D:\dataset\merged_mix\train_raw.jsonl --output-jsonl train_with_codes.jsonl `
  --device auto
```

已有预编码数据可跳过此步。下面以单卡、几十小时单说话人数据为例，显式设置
`rank=16`、`alpha=32`、学习率 `1e-4`：

```powershell
python -m accelerate.commands.launch --num_processes 1 moss_tts_delay\finetuning\sft_lora.py `
  --model-path OpenMOSS-Team/MOSS-TTS-v1.5 `
  --train-jsonl train_with_codes.jsonl `
  --output-dir output\moss_tts_lora `
  --per-device-batch-size 1 --gradient-accumulation-steps 8 `
  --learning-rate 1e-4 --num-epochs 3 --mixed-precision bf16 `
  --gradient-checkpointing --attn-implementation sdpa `
  --lora-r 16 --lora-alpha 32 --lora-dropout 0.05
```

在 Git Bash/WSL 中可将上面的 `python -m accelerate.commands.launch` 换回 `accelerate launch`，并将路径分隔符改为 `/`。

脚本默认 `rank=8`、`alpha=16`、学习率 `1e-4`；上述命令覆盖 rank 和 alpha。
原 `sft.py` 保持全参训练用途。
使用 `--lora-target-modules q_proj,v_proj` 可缩小训练范围；未匹配到主干线性层的名称会报错。

### 单卡 RTX 5090 的试跑建议

先使用上述 batch size 1、BF16、梯度检查点和 SDPA 配置，并在命令末尾增加
`--max-train-steps 10` 验证训练和保存，正式训练时删除该参数。
建议先使用第 2 节随机混合时长（上限 90 秒）合并的数据试跑，同时保留短句和尾段。目标音频和参考音频（如有）均提前编码；参考音频长度需单独设置，合并参数只控制目标音频。
当前脚本不会自动截断超长样本；如显存不足，优先缩短单条音频和参考音频长度。
梯度累积不会减少单条样本的显存占用。

当前尚未在 RTX 5090 上实测，不能保证任意音频长度都能放入显存。

### 多卡与分片输入

多卡时将 `--num_processes 1` 替换为
`--config_file moss_tts_delay\finetuning\configs\accelerate_ddp_8gpu.yaml`，按实际 GPU 数调整配置。
FSDP 要求 `fsdp_use_orig_params: true`，
保存要求完整 state dict；ZeRO-3 要求 `zero3_save_16bit_model: true`。分片后端保存时会聚合完整权重，需预留内存。
这些分布式路径沿用原训练脚本，实际兼容性应在目标训练环境验证。
训练时各进程读取全部指定 JSONL，由 Accelerate 统一分配 batch；即使输入是预处理分片，也走同一流程。
因此各进程需要容纳全部 JSONL 数据的 CPU 内存。`--max-train-steps` 指定训练步数时可跨越 `--num-epochs` 完成。
Windows PowerShell 不会自动展开参数中的通配符，分片输入请保留引号并交给脚本处理，例如 `--train-jsonl "train_with_codes.rank*.jsonl"`。

## 保存、继续训练和推理

每轮保存 `output\moss_tts_lora\checkpoint-epoch-N\`（N 从 0 开始），包含
`adapter_model.safetensors`、`adapter_config.json`、`finetune_args.json`，以及完整续训状态：

- `training_state/`：优化器、学习率调度器、各进程 Python/NumPy/PyTorch/CUDA RNG，以及 FP16 scaler（启用时）。
- `trainer_state.json`：global step、下一 epoch/batch 位置、数据指纹、进程数和后端；此文件在保存成功后才写入。

默认每轮保存；训练命令增加 `--save-steps 500` 可每 500 个梯度累积更新边界额外保存到
`checkpoint-step-500/` 等目录。仅在梯度累积完成后保存，不保存尚未提交的梯度。
单卡/DDP 只存 adapter 和训练状态；FSDP/DeepSpeed 还保留后端原生模型状态，磁盘开销更大。
已有 checkpoint 目录不会被覆盖；若回退到较早 checkpoint 重跑，请指定新的输出目录。

完整断点续训：

```powershell
python -m accelerate.commands.launch --num_processes 1 moss_tts_delay\finetuning\sft_lora.py `
  --train-jsonl train_with_codes.jsonl `
  --output-dir output\moss_tts_lora_resumed `
  --resume-from-checkpoint output\moss_tts_lora\checkpoint-step-500 `
  --save-steps 500
```

完整恢复自动使用 checkpoint 的基础模型路径、batch size、梯度累积、学习率、调度计划、
seed、LoRA 配置等训练设置，并跳过已完成 batch；这些设置会覆盖当前命令中的对应值。
`--train-jsonl`、输出目录、日志和保存频率可重新指定，但 JSONL 内容及文件顺序、GPU 进程数、后端须保持一致。
基础模型文件、外部参考音频和运行环境也应保持不变；数据指纹只校验 JSONL。
已达到原训练目标的 checkpoint 会直接结束，不会重新 warmup 或额外训练。
同一环境下恢复随机状态和样本顺序，但 GPU 非确定性算子仍可能造成数值差异。
本功能的真实 GPU、FSDP/DeepSpeed 恢复尚待目标环境验证。

用相同基础模型加 `--lora-resume-adapter output\moss_tts_lora\checkpoint-epoch-0` 可继续训练。
此选项仅恢复 adapter 权重，优化器、调度器和 epoch 计数重新开始，LoRA 结构以保存的 adapter 配置为准。
它用于开始新的训练计划，不能与 `--resume-from-checkpoint` 同时使用。
旧版本仅含 adapter 的 checkpoint 只能使用此选项。

推理时可以显式加载两部分：

```python
from peft import PeftModel
from moss_tts_delay.modeling_moss_tts import MossTTSDelayModel

base = MossTTSDelayModel.from_pretrained("OpenMOSS-Team/MOSS-TTS-v1.5")
model = PeftModel.from_pretrained(base, "output/moss_tts_lora/checkpoint-epoch-0")
model.eval()
# processor 从基础模型加载，后续沿用现有推理流程。
```

要直接供现有推理脚本使用，先合并成完整模型：

```powershell
python moss_tts_delay\finetuning\merge_lora.py `
  --model-path OpenMOSS-Team/MOSS-TTS-v1.5 `
  --adapter-path output\moss_tts_lora\checkpoint-epoch-0 `
  --output-dir output\moss_tts_lora_merged --dtype bfloat16
```

合并在 CPU 上执行，默认 float32；可用 `--dtype bfloat16` 降低内存占用。
输出目录必须为空或不存在，基础模型必须与训练时一致。

## MOSS-TTSD 基座的 LoRA

TTSD-v1.0 与 v1.5 同为 `moss_tts_delay` 架构，代码沿用同一套，不需要替换 `.py` 文件。区别有三点：
- RVQ 为 16 路；
- prompt 多一个 `- Scene:` 字段，且 `Tokens` 固定为 None；
- 文本不做规范化，用 `[S1]`/`[S2]` 标注说话人。

`processing_moss_tts.py` 新增 `prompt_template`（`auto` / `moss_tts` / `ttsd`）。`auto` 在模型 n_vq=16 时选 `ttsd`，否则选 `moss_tts`，所以 v1.5 的行为不变。`ttsd` 分支逐项对齐官方 TTSD 的 `processing_moss_tts.py`。

1. 预编码：不指定 `--n-vq` 时按模型 n_vq 编码（TTSD 为 16 路）。已有的 32 路编码数据也能直接用，训练时会自动截取前 16 路，只是文件更大。

   ```powershell
   python moss_tts_delay\finetuning\prepare_data.py `
     --model-path OpenMOSS-Team/MOSS-TTSD-v1.0 `
     --codec-path OpenMOSS-Team/MOSS-Audio-Tokenizer `
     --input-jsonl D:\dataset\merged_mix\train_raw.jsonl --output-jsonl train_ttsd_codes.jsonl `
     --device auto
   ```

2. 训练：只需换 `--model-path`，模板会自动解析为 `ttsd`，也可以显式写 `--prompt-template ttsd`。单说话人数据的文本不必改：`--speaker-tag auto`（默认）会给没有 `[Sx]` 标签的文本自动加 `[S1]` 前缀。启动日志会打印 `prompt_template=auto -> ttsd`。

   ```powershell
   python -m accelerate.commands.launch --num_processes 1 moss_tts_delay\finetuning\sft_lora.py `
     --model-path OpenMOSS-Team/MOSS-TTSD-v1.0 `
     --train-jsonl train_ttsd_codes.jsonl `
     --output-dir output\ttsd_lora `
     --per-device-batch-size 1 --gradient-accumulation-steps 4 `
     --learning-rate 1e-4 --lr-scheduler-type cosine --warmup-steps 50 `
     --weight-decay 0.01 --max-grad-norm 1.0 `
     --channelwise-loss-weight "1,16" `
     --num-epochs 18 --mixed-precision bf16 `
     --gradient-checkpointing --attn-implementation sdpa `
     --lora-r 32 --lora-alpha 64 --lora-dropout 0.05 `
     --save-steps 100
   ```

   - `--channelwise-loss-weight "1,16"`：强烈建议显式传入。TTSD 音频只有 16 路，传 `"1,16"` 才能使每个音频头权重与 v1.5 一样保持为 1.0；若保留默认 `"1,32"` 则每个音频头权重会变成 2.0。
   - `--lora-r 32 --lora-alpha 64`：TTSD 仍为 8B 语言模型底座，r=32 容量充足。
   - 轮数与过拟合：TTSD 通道少一半（16 路），单人声数据比 v1.5 更易拟合，建议开启 `--save-steps` 并重点抽测 Epoch 6 ~ 12 检查点。
   - 实际使用的模板记录在 `finetune_args.json`（`resolved_prompt_template`）和每个 checkpoint 的 `processor_config.json` 里。完整续训时会自动恢复。

3. 合并：用法同上，`--model-path` 必须是 TTSD。`merge_lora.py` 会把 adapter 的 `prompt_template` 写入合并后模型的 `processor_config.json`。

4. 推理：GGUF 部署见 [openmoss 文档第 13 节](../openmoss/README_zh.md)，务必加 `--template ttsd`。推理文本同样需要 `[S1]` 前缀，Gradio 在 `--template ttsd` 下会自动补上。

> [!NOTE]
> 以上改动尚未在 GPU 上实测。建议先运行 `python scripts\check_ttsd_prompt.py`，确认 v1.5 未回归、TTSD prompt 与官方 processor 逐 token 一致；再加 `--max-train-steps 10` 试跑。

## TensorBoard 训练日志

Delay LoRA 已使用 TensorBoard 替代 W&B，不再接受 `--wandb-*` 参数。
安装依赖：`pip install tensorboard`，或重新安装 `pip install -e ".[finetune-lora]"`。

默认启用日志，保存到 `<output-dir>/tensorboard`，只由主进程写入。
记录 loss、学习率、每步耗时、每秒步数、每秒样本数、epoch、预计剩余秒数及训练参数。
`--logging-steps` 控制指标记录间隔；loss 沿用原训练脚本的口径，即记录时最后一个 micro-batch 的跨进程平均值。

```powershell
tensorboard --logdir output/moss_tts_lora/tensorboard --port 6006
```

浏览器打开 `http://localhost:6006`。可用 `--tensorboard-log-dir 路径` 自定义日志目录，
用 `--no-tensorboard` 关闭。不同训练实验应使用不同目录。
完整断点续训时继续使用原日志目录，保留 checkpoint 已完成步数的记录，并隐藏其后失效的旧指标。
训练正常结束或抛出异常时关闭 writer，将已排队事件写入磁盘。
