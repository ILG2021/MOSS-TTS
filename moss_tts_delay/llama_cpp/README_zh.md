# MOSS-TTS-Delay：llama.cpp 推理后端（Windows 原生指南）

[English](README.md) | [简体中文](README_zh.md)

本模块使用 llama.cpp 运行 Qwen3 backbone（GGUF）、NumPy/PyTorch 运行 embedding 与 33 个 LM heads、ONNX Runtime/TensorRT 运行音频 codec，并由 Python 完成 delay-pattern 解码。

本文以 **Windows 10/11、PowerShell、MOSS-TTS-v1.5 Delay 8B 和 LoRA checkpoint** 为主线，从 LoRA 合并一直写到 Q4_K_M 推理。全部命令均为 Windows 原生命令，不需要 Bash。

> 本后端仅适用于 `MossTTSDelay`，包括 `OpenMOSS-Team/MOSS-TTS-v1.5`。不要用于 Local、Realtime 或 Nano。

## 1. 最终产物与量化范围

转换结果不是单个 GGUF，而是四部分：

```text
weights/moss-tts-v1.5-lora-q4km/
├── backbone_q4_k_m.gguf       # Qwen3 backbone，Q4_K_M
├── embeddings/                # 文本 + 32 路音频 embedding，FP16 .npy
├── lm_heads/                  # 文本 + 32 路音频 head，FP16 .npy
├── qwen3_backbone/            # 转换中间文件和 tokenizer
└── extraction_meta.json
```

`Q4_K_M` 只量化 Qwen3 backbone。Embedding、LM heads 和音频 codec 不会被 `llama-quantize` 量化。普通 `llama-cli` 不能完成 TTS，必须使用本目录的 pipeline 和 bridge。

LoRA 默认冻结 embedding 和 LM heads，但仍应从**合并后的同一 checkpoint**提取全部文件，不能混用其他版本的 tokenizer、embedding 或 heads。

## 2. Windows 前置条件

请准备：

1. Windows 10/11 x64；
2. Git、Python 3.10/3.11 x64、CMake 3.21+；
3. Visual Studio 2022 Build Tools，勾选“使用 C++ 的桌面开发”和 Windows SDK；
4. NVIDIA 路线需要匹配驱动的 CUDA Toolkit；纯 CPU 路线不需要 CUDA；
5. 至少约 45 GB 可用磁盘；
6. 合并 8B LoRA 建议至少 32 GB 系统内存，内存不足时增加 Windows 页面文件。

检查工具：

```powershell
git --version
py --version
cmake --version
```

如果 CMake 找不到 MSVC，请使用“Developer PowerShell for VS 2022”。下文假设当前目录为仓库根目录：

```powershell
Set-Location D:\vibecoding\MOSS-TTS
```

请将示例绝对路径替换为你的实际路径。

## 3. 创建转换环境

LoRA 合并和权重提取需要 PyTorch。可复用训练环境，也可新建环境：

```powershell
py -3.10 -m venv .venv-convert
.\.venv-convert\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e ".[torch-runtime,finetune-lora,llama-cpp-onnx]"
```

CUDA wheel 要与机器环境兼容；只在 CPU 合并时可安装 CPU 版 PyTorch。如果 PowerShell 禁止激活脚本，可对当前用户执行一次：

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

验证依赖：

```powershell
python -c "import torch, peft, safetensors, numpy; print(torch.__version__); print('packages OK')"
```

## 4. 检查 LoRA checkpoint

本文示例使用 `output\moss_tts_lora\checkpoint-epoch-0`。检查必需文件：

```powershell
Test-Path output\moss_tts_lora\checkpoint-epoch-0\adapter_config.json
Test-Path output\moss_tts_lora\checkpoint-epoch-0\adapter_model.safetensors
```

两项都应为 `True`。`adapter_config.json` 中的基础模型必须与训练时一致；v1.5 示例应使用 `OpenMOSS-Team/MOSS-TTS-v1.5`，不要误用 1.0 checkpoint。

## 5. 合并 LoRA

输出目录必须不存在或为空：

```powershell
python moss_tts_delay\finetuning\merge_lora.py `
  --model-path OpenMOSS-Team/MOSS-TTS-v1.5 `
  --adapter-path output\moss_tts_lora\checkpoint-epoch-0 `
  --output-dir output\moss_tts_v1_5_lora_merged `
  --codec-path OpenMOSS-Team/MOSS-Audio-Tokenizer `
  --dtype bfloat16
```

- `--model-path` 必须与训练时完全一致；
- `--dtype bfloat16` 比默认 float32 节省内存；
- 脚本使用 `safe_merge=True`，非有限权重会直接报错；
- 全参数 SFT 无需合并，后续直接使用完整 SFT checkpoint。

检查输出：

```powershell
Test-Path output\moss_tts_v1_5_lora_merged\config.json
Get-ChildItem output\moss_tts_v1_5_lora_merged -Filter "*.safetensors"
```

建议先用原 PyTorch 推理链路试听合并模型。合并模型本身异常时，不要继续量化。

## 6. 提取 backbone、embeddings 和 heads

```powershell
python moss_tts_delay\llama_cpp\conversion\extract_weights.py `
  --model output\moss_tts_v1_5_lora_merged `
  --output weights\moss-tts-v1.5-lora-q4km
```

校验结果：

```powershell
(Get-ChildItem weights\moss-tts-v1.5-lora-q4km\embeddings -Filter "*.npy").Count
(Get-ChildItem weights\moss-tts-v1.5-lora-q4km\lm_heads -Filter "*.npy").Count
Test-Path weights\moss-tts-v1.5-lora-q4km\qwen3_backbone\config.json
```

Delay v1.5 的前两项都应为 `33`，最后一项应为 `True`。数量不符时不要继续。

## 7. Windows 原生编译 llama.cpp

建议将 llama.cpp 放在仓库同级目录：

```powershell
Set-Location ..
git clone https://github.com/ggerganov/llama.cpp.git
Set-Location llama.cpp
```

NVIDIA GPU 构建：

```powershell
cmake -S . -B build -A x64 `
  -DBUILD_SHARED_LIBS=ON `
  -DGGML_CUDA=ON `
  -DLLAMA_CURL=OFF
cmake --build build --config Release -j
```

纯 CPU 构建：

```powershell
cmake -S . -B build -A x64 `
  -DBUILD_SHARED_LIBS=ON `
  -DGGML_CUDA=OFF `
  -DLLAMA_CURL=OFF
cmake --build build --config Release -j
```

必须启用 `BUILD_SHARED_LIBS`。检查产物：

```powershell
Get-ChildItem .\build -Recurse -Filter llama-quantize.exe
Get-ChildItem .\build -Recurse -Filter llama.dll
```

下文假设 llama.cpp 位于 `D:\vibecoding\llama.cpp`。llama.cpp 的 C API 会变化，因此转换、量化、bridge 编译和运行应使用同一 checkout；更新后要重新编译 bridge。

## 8. 转换为 F16 GGUF

```powershell
Set-Location D:\vibecoding\MOSS-TTS
python D:\vibecoding\llama.cpp\convert_hf_to_gguf.py `
  weights\moss-tts-v1.5-lora-q4km\qwen3_backbone `
  --outfile weights\moss-tts-v1.5-lora-q4km\backbone_f16.gguf `
  --outtype f16
```

检查输出：

```powershell
Get-Item weights\moss-tts-v1.5-lora-q4km\backbone_f16.gguf | Select-Object FullName,Length
```

F16 backbone 通常约 16 GB。转换器若提示缺包，请在当前环境安装其明确提示的依赖后重试。

## 9. 量化为 Q4_K_M

定位量化程序：

```powershell
$quantize = Get-ChildItem D:\vibecoding\llama.cpp\build -Recurse -Filter llama-quantize.exe | Select-Object -First 1 -ExpandProperty FullName
$quantize
```

确认路径正确后执行：

```powershell
& $quantize `
  weights\moss-tts-v1.5-lora-q4km\backbone_f16.gguf `
  weights\moss-tts-v1.5-lora-q4km\backbone_q4_k_m.gguf `
  Q4_K_M
```

```powershell
Get-Item weights\moss-tts-v1.5-lora-q4km\backbone_q4_k_m.gguf | Select-Object FullName,Length
```

Q4_K_M 通常约 4.8 GB。保留 F16 文件直到试听完成。量化有损，不要对已量化 GGUF 二次量化。

| 格式 | 近似大小 | 建议 |
|---|---:|---|
| Q4_K_M | 4.8 GB | 默认体积/质量折中 |
| Q5_K_M | 5.7 GB | 更保守 |
| Q6_K | 6.6 GB | 音色、韵律敏感时优先测试 |
| Q8_0 | 8.7 GB | 最大限度减少 backbone 损失 |

## 10. 下载 ONNX 音频 codec

```powershell
huggingface-cli download OpenMOSS-Team/MOSS-Audio-Tokenizer-ONNX `
  --local-dir weights\MOSS-Audio-Tokenizer-ONNX
```

命令不存在时先执行：

```powershell
python -m pip install --upgrade "huggingface_hub[cli]"
```

检查：

```powershell
Test-Path weights\MOSS-Audio-Tokenizer-ONNX\encoder.onnx
Test-Path weights\MOSS-Audio-Tokenizer-ONNX\decoder.onnx
```

两项都应为 `True`。ONNX 是 Windows 首选入门后端；TensorRT engine 与显卡、CUDA 和 TensorRT 版本绑定，不提供通用预编译文件。

## 11. 编译 Windows bridge DLL

```powershell
powershell -ExecutionPolicy Bypass -File `
  moss_tts_delay\llama_cpp\build_bridge.ps1 `
  -LlamaCppDir D:\vibecoding\llama.cpp
```

脚本使用 CMake/MSVC 生成 `backbone_bridge.dll`，并把 `llama.dll` 及同目录 ggml DLL 复制到 `moss_tts_delay\llama_cpp\`。检查：

```powershell
Test-Path moss_tts_delay\llama_cpp\backbone_bridge.dll
Test-Path moss_tts_delay\llama_cpp\llama.dll
```

更新或重编 llama.cpp 后重新运行本节，避免混用不同构建的 `.lib` 和 DLL。

## 12. 创建微调模型配置

```powershell
Copy-Item configs\llama_cpp\default.yaml configs\llama_cpp\finetuned-v1.5-q4km.yaml
```

编辑新文件，至少修改：

```yaml
backbone_gguf: weights/moss-tts-v1.5-lora-q4km/backbone_q4_k_m.gguf
embedding_dir: weights/moss-tts-v1.5-lora-q4km/embeddings
lm_head_dir: weights/moss-tts-v1.5-lora-q4km/lm_heads
tokenizer_dir: weights/moss-tts-v1.5-lora-q4km/qwen3_backbone

audio_backend: onnx
audio_encoder_onnx: weights/MOSS-Audio-Tokenizer-ONNX/encoder.onnx
audio_decoder_onnx: weights/MOSS-Audio-Tokenizer-ONNX/decoder.onnx

heads_backend: auto
n_ctx: 4096
n_batch: 512
n_threads: 8
n_gpu_layers: -1
max_new_tokens: 3072
use_gpu_audio: true
flash_attn: auto
```

- `heads_backend: auto`：有 PyTorch 时使用 torch heads，否则 NumPy；
- `n_gpu_layers: -1`：尽可能全部放 GPU；纯 CPU 使用 `0`；
- 纯 CPU 的 `n_threads` 可先设为物理核心数；
- 增大 `n_ctx` 会增加 KV cache；
- 显存不足时减少 `n_gpu_layers`、`n_ctx`、`n_batch`，或参考 `trt-8gb.yaml`；
- 路径相对于启动命令的当前目录，建议始终从仓库根目录运行。

## 13. 首次推理

先创建输出目录：

```powershell
New-Item -ItemType Directory -Force outputs | Out-Null
```

无参考音频：

```powershell
python -m moss_tts_delay.llama_cpp `
  --config configs\llama_cpp\finetuned-v1.5-q4km.yaml `
  --text "你好，这是微调模型的 llama.cpp 推理测试。" `
  --language Chinese `
  --output outputs\finetuned_q4km.wav `
  --profile
```

语音克隆：

```powershell
python -m moss_tts_delay.llama_cpp `
  --config configs\llama_cpp\finetuned-v1.5-q4km.yaml `
  --text "你好，这是参考音色克隆测试。" `
  --language Chinese `
  --reference samples\reference.wav `
  --output outputs\finetuned_q4km_clone.wav `
  --profile
```

v1.5 已知语言时建议使用完整名称，如 `Chinese`、`English`、`French`。输出为 24 kHz。

强制 NumPy heads：

```powershell
python -m moss_tts_delay.llama_cpp `
  --config configs\llama_cpp\finetuned-v1.5-q4km.yaml `
  --text "这是纯 NumPy 输出头测试。" `
  --language Chinese `
  --heads-backend numpy `
  --output outputs\finetuned_q4km_numpy.wav
```

NumPy heads 不需要 PyTorch，但通常明显慢于 torch heads。

## 14. 可选：无 PyTorch 独立推理环境

完成转换后，部署环境可不装 PyTorch：

```powershell
deactivate
py -3.10 -m venv .venv-llama
.\.venv-llama\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[llama-cpp-onnx]"
```

使用 `--heads-backend numpy` 或在 YAML 设置 `heads_backend: numpy`。bridge DLL、GGUF、`.npy` 权重和 ONNX codec 仍需保留。

## 15. Python API

```python
import soundfile as sf
from moss_tts_delay.llama_cpp import LlamaCppPipeline, PipelineConfig

config = PipelineConfig.from_yaml("configs/llama_cpp/finetuned-v1.5-q4km.yaml")
with LlamaCppPipeline(config) as pipeline:
    waveform = pipeline.generate(
        text="你好世界！",
        reference_audio="samples/reference.wav",  # 不克隆时设为 None
        language="Chinese",
    )
sf.write("outputs/python_api.wav", waveform, 24000)
```

## 16. 量化验证

使用相同文本、参考音频和采样参数比较：

1. LoRA 合并后的 Hugging Face BF16 模型；
2. F16 GGUF backbone；
3. Q4_K_M GGUF backbone。

检查说话人相似度、微调词汇发音、长句漏字/重复、停顿韵律和多次生成稳定性。若 Q4_K_M 明显退化，改用 Q5_K_M 或 Q6_K。官方评测只能作为参考，微调音色必须实际试听。

## 17. 常见问题

### 找不到 `backbone_bridge.dll`

重新执行第 11 节，DLL 应位于 `moss_tts_delay\llama_cpp\backbone_bridge.dll`。

### `OSError: [WinError 126]`

这通常表示依赖 DLL 缺失：重新运行 `build_bridge.ps1`；安装 Microsoft Visual C++ Redistributable 2022 x64；CUDA 构建时检查驱动和 CUDA；不要混用另一份 llama.cpp 的 DLL。

### bridge 出现 llama API 符号/类型错误

bridge 与 llama.cpp C API 不匹配。确保使用同一 checkout 进行 llama.cpp 编译、GGUF 转换和 bridge 编译；更新后重新构建。

### 转换器不识别模型

输入必须是提取后的 `weights\moss-tts-v1.5-lora-q4km\qwen3_backbone`，不能直接输入完整 MOSS-TTS checkpoint。

### 输出像基础模型，LoRA 效果消失

通常是转换了官方基础模型。查看 `extraction_meta.json` 的 `source_model`，应指向合并目录 `output\moss_tts_v1_5_lora_merged`。

### embeddings/heads 不是各 33 个

checkpoint 不完整、架构不匹配或提取失败。不要从官方包复制文件补齐，应返回合并步骤排查。

### CUDA 显存不足

依次降低 `n_gpu_layers`、`n_ctx` 和 `n_batch`；启用 flash attention；尝试 `q8_0` KV cache；避免 torch heads 与 GPU ONNX codec 同时长期占用过多显存。

### ONNX Runtime 没有使用 GPU

```powershell
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

应包含 `CUDAExecutionProvider`。否则检查 `onnxruntime-gpu`、CUDA/cuDNN 和 PATH，也可把 `use_gpu_audio` 设为 `false`。

### 中文或多语种不稳定

显式传入语言名，并确认 tokenizer、embedding、heads 和 GGUF 均来自同一个 v1.5 合并 checkpoint。

## 18. 官方模型快速体验

不使用微调权重时可下载官方包：

```powershell
huggingface-cli download OpenMOSS-Team/MOSS-TTS-GGUF --local-dir weights\MOSS-TTS-GGUF
huggingface-cli download OpenMOSS-Team/MOSS-Audio-Tokenizer-ONNX --local-dir weights\MOSS-Audio-Tokenizer-ONNX
```

完成 llama.cpp 与 bridge 编译后运行：

```powershell
python -m moss_tts_delay.llama_cpp `
  --config configs\llama_cpp\default.yaml `
  --text "你好世界！" `
  --language Chinese `
  --output outputs\official_q4km.wav
```

官方 GGUF 不能替代自己的 LoRA 合并与转换结果。

## 19. 批量评测与官方量化结果

```powershell
python scripts\batch_eval_llama_cpp.py `
  --config configs\llama_cpp\finetuned-v1.5-q4km.yaml `
  --benchmark-dir D:\datasets\eval\tts `
  --result-dir results\llama_cpp_finetuned_q4km `
  --suite seed-tts
```

官方 Seed-TTS zero-shot 结果如下；Baseline 是 Hugging Face 原模型，GGUF 使用 llama.cpp 和 TensorRT codec：

| 量化 | EN WER ↓ | EN SIM ↑ | ZH CER ↓ | ZH SIM ↑ |
|---|---:|---:|---:|---:|
| Baseline | 1.79 | 71.46 | 1.32 | 77.05 |
| Q8_0 | 3.21 | 68.61 | 1.56 | 76.03 |
| Q6_K | 3.11 | 68.77 | 1.44 | 76.06 |
| Q5_K_M | 2.95 | 68.55 | 1.50 | 75.96 |
| Q4_K_M | 2.83 | 68.15 | 1.58 | 75.71 |

## 20. 架构与相关文件

```text
文本/参考音频 → Tokenizer/ONNX encoder → 33 路 delay prompt
→ FP16 embedding → llama.cpp Qwen3 backbone
→ NumPy/Torch LM heads → delay 状态机与采样
→ 32 路 audio codes → ONNX/TensorRT decoder → 24 kHz 波形
```

```text
moss_tts_delay/llama_cpp/
├── pipeline.py
├── backbone.py
├── backbone_bridge.c
├── CMakeLists.txt
├── build_bridge.ps1          # Windows 原生构建
├── build_bridge.sh           # Linux 构建
├── embedding.py
├── lm_heads.py
├── delay_state.py
├── sampling.py
├── processor.py
└── conversion/extract_weights.py
```

LoRA 训练与 checkpoint 说明见 [Delay LoRA 微调文档](../finetuning/README_lora_zh.md)，通用转换说明见 [转换指南](conversion/README_zh.md)。
