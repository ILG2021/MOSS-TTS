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

数据格式与预处理方式见 [原微调文档](README_zh.md)。如果原始数据是连续编号的 LJSpeech
短音频，可先按推荐的约 2 分钟长度合并。先用 `--dry-run` 检查分组计划：

```powershell
python scripts\merge_ljspeech.py `
  --input "D:\dataset\metadata.txt" `
  --output-dir "D:\dataset\merged_2min" `
  --target-seconds 120 --max-seconds 150 --dry-run
```

确认后去掉 `--dry-run` 生成合并数据：

```powershell
python scripts\merge_ljspeech.py `
  --input "D:\dataset\metadata.txt" `
  --output-dir "D:\dataset\merged_2min" `
  --target-seconds 120 --max-seconds 150
```

该命令会生成 `merged_2min\train_raw.jsonl`；完整参数和输入约束见
[`scripts/merge_ljspeech.md`](../../scripts/merge_ljspeech.md)。未编码的数据再执行：

```powershell
python moss_tts_delay\finetuning\prepare_data.py `
  --model-path OpenMOSS-Team/MOSS-TTS-v1.5 `
  --codec-path OpenMOSS-Team/MOSS-Audio-Tokenizer `
  --input-jsonl D:\dataset\merged_2min\train_raw.jsonl --output-jsonl train_with_codes.jsonl `
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
建议使用约 2 分钟的音频素材，目标音频和参考音频均提前编码。
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
