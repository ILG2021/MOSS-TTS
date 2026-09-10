# MOSS-TTS Delay LoRA 微调

`sft_lora.py` 是独立训练脚本，基于原全参代码实现，不导入或调用 `sft.py`。
数据集和音频预处理仍使用现有 `dataset.py`、`prepare_data.py`。
默认只训练 Qwen3 主干的 q/k/v/o_proj、gate/up/down_proj 的低秩参数；文本与音频 embedding、各输出头冻结。
这是非量化 LoRA，仍需加载完整基础模型；显存占用取决于基础权重、序列长度和激活。

## 安装与训练

先按仓库说明安装 PyTorch、Transformers 等运行环境，再在仓库根目录执行：

```bash
pip install -e ".[finetune-lora]"
```

以下命令直接运行 Python 训练入口，无需启动脚本。多行命令使用 Bash 的 `\` 续行；
在 PowerShell 中请将命令合为一行，或将行末 `\` 改为反引号。

数据格式与预处理方式见 [原微调文档](README_zh.md)。未编码的数据先执行：

```bash
python moss_tts_delay/finetuning/prepare_data.py \
  --model-path OpenMOSS-Team/MOSS-TTS-v1.5 \
  --codec-path OpenMOSS-Team/MOSS-Audio-Tokenizer \
  --input-jsonl train_raw.jsonl --output-jsonl train_with_codes.jsonl \
  --device auto
```

已有预编码数据可跳过此步。下面以单卡、几十小时单说话人数据为例，显式设置
`rank=16`、`alpha=32`、学习率 `1e-4`：

```bash
accelerate launch --num_processes 1 moss_tts_delay/finetuning/sft_lora.py \
  --model-path OpenMOSS-Team/MOSS-TTS-v1.5 \
  --train-jsonl train_with_codes.jsonl \
  --output-dir output/moss_tts_lora \
  --per-device-batch-size 1 --gradient-accumulation-steps 8 \
  --learning-rate 1e-4 --num-epochs 3 --mixed-precision bf16 \
  --gradient-checkpointing --attn-implementation sdpa \
  --lora-r 16 --lora-alpha 32 --lora-dropout 0.05
```

脚本默认 `rank=8`、`alpha=16`、学习率 `1e-4`；上述命令覆盖 rank 和 alpha。
原 `sft.py` 保持全参训练用途。
使用 `--lora-target-modules q_proj,v_proj` 可缩小训练范围；未匹配到主干线性层的名称会报错。

### 单卡 RTX 5090 的试跑建议

先使用上述 batch size 1、BF16、梯度检查点和 SDPA 配置，并在命令末尾增加
`--max-train-steps 10` 验证训练和保存，正式训练时删除该参数。
建议先用 5～15 秒音频切片测试，目标音频和参考音频均提前编码。
当前脚本不会自动截断超长样本；如显存不足，优先缩短单条音频和参考音频长度。
梯度累积不会减少单条样本的显存占用。

当前尚未在 RTX 5090 上实测，不能保证任意音频长度都能放入显存。

### 多卡与分片输入

多卡时将 `--num_processes 1` 替换为
`--config_file moss_tts_delay/finetuning/configs/accelerate_ddp_8gpu.yaml`，按实际 GPU 数调整配置。
FSDP 要求 `fsdp_use_orig_params: true`，
保存要求完整 state dict；ZeRO-3 要求 `zero3_save_16bit_model: true`。分片后端保存时会聚合完整权重，需预留内存。
这些分布式路径沿用原训练脚本，实际兼容性应在目标训练环境验证。
训练时各进程读取全部指定 JSONL，由 Accelerate 统一分配 batch；即使输入是预处理分片，也走同一流程。
因此各进程需要容纳全部 JSONL 数据的 CPU 内存。`--max-train-steps` 指定训练步数时可跨越 `--num-epochs` 完成。
分片输入请显式指定，例如 `--train-jsonl "train_with_codes.rank*.jsonl"`。

## 保存、继续训练和推理

每轮保存 `output/moss_tts_lora/checkpoint-epoch-N/`（N 从 0 开始），包含
`adapter_model.safetensors`、`adapter_config.json`、`finetune_args.json`，以及完整续训状态：

- `training_state/`：优化器、学习率调度器、各进程 Python/NumPy/PyTorch/CUDA RNG，以及 FP16 scaler（启用时）。
- `trainer_state.json`：global step、下一 epoch/batch 位置、数据指纹、进程数和后端；此文件在保存成功后才写入。

默认每轮保存；训练命令增加 `--save-steps 500` 可每 500 个梯度累积更新边界额外保存到
`checkpoint-step-500/` 等目录。仅在梯度累积完成后保存，不保存尚未提交的梯度。
单卡/DDP 只存 adapter 和训练状态；FSDP/DeepSpeed 还保留后端原生模型状态，磁盘开销更大。
已有 checkpoint 目录不会被覆盖；若回退到较早 checkpoint 重跑，请指定新的输出目录。

完整断点续训：

```bash
accelerate launch --num_processes 1 moss_tts_delay/finetuning/sft_lora.py \
  --train-jsonl train_with_codes.jsonl \
  --output-dir output/moss_tts_lora_resumed \
  --resume-from-checkpoint output/moss_tts_lora/checkpoint-step-500 \
  --save-steps 500
```

完整恢复自动使用 checkpoint 的基础模型路径、batch size、梯度累积、学习率、调度计划、
seed、LoRA 配置等训练设置，并跳过已完成 batch；这些设置会覆盖当前命令中的对应值。
`--train-jsonl`、输出目录、日志和保存频率可重新指定，但 JSONL 内容及文件顺序、GPU 进程数、后端须保持一致。
基础模型文件、外部参考音频和运行环境也应保持不变；数据指纹只校验 JSONL。
已达到原训练目标的 checkpoint 会直接结束，不会重新 warmup 或额外训练。
同一环境下恢复随机状态和样本顺序，但 GPU 非确定性算子仍可能造成数值差异。
本功能的真实 GPU、FSDP/DeepSpeed 恢复尚待目标环境验证。

用相同基础模型加 `--lora-resume-adapter output/moss_tts_lora/checkpoint-epoch-0` 可继续训练。
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

```bash
python moss_tts_delay/finetuning/merge_lora.py \
  --model-path OpenMOSS-Team/MOSS-TTS-v1.5 \
  --adapter-path output/moss_tts_lora/checkpoint-epoch-0 \
  --output-dir output/moss_tts_lora_merged --dtype bfloat16
```

合并在 CPU 上执行，默认 float32；可用 `--dtype bfloat16` 降低内存占用。
输出目录必须为空或不存在，基础模型必须与训练时一致。
