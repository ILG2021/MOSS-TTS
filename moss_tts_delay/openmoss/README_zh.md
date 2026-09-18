# 基于 openmoss 的 LoRA → 全 GGML → 常驻 WebUI（Windows 原生）

本方案使用 [`pwilkin/openmoss`](https://github.com/pwilkin/openmoss)，不是本仓库的 `moss_tts_delay/llama_cpp` 混合后端。openmoss 中 Qwen3 backbone 由 libllama 执行，audio embeddings、33 路 LM heads、MOSS AudioTokenizer encoder/RVQ/decoder 均由 GGML graph 执行，不依赖 PyTorch 或 ONNX 做最终推理。

## 1. 最终结构

```text
MOSS-TTS-v1.5 原始基座
  → openmoss converter
      ├─ moss-tts-base.gguf              # Qwen3 backbone
      ├─ moss-tts-base.extras.gguf       # embeddings + heads + 完整 codec
      └─ convert-scratch\qwen3_backbone  # 转 LoRA 时需要保留
  → llama-quantize Q4_K_M
      ├─ moss-tts-base-q4km.gguf
      └─ moss-tts-base-q4km.extras.gguf
LoRA checkpoint A/B/...
  → convert_moss_lora_to_gguf.py
      └─ speaker-a.gguf / speaker-b.gguf / ...
  → moss-tts-server.exe（一次加载、持续常驻）
  → openmoss 自带 WebUI，或本仓库 Gradio 代理界面
```

## 2. 检查 LoRA 产物（不要合并）

多音色运行时切换必须保留独立 adapter，不要先执行 `merge_lora.py`。每个 LoRA 目录至少应包含：

```powershell
Test-Path .\output\speaker-a\adapter_config.json
Test-Path .\output\speaker-a\adapter_model.safetensors
```

本方案只支持 `moss_tts_delay/finetuning/sft_lora.py` 产生的 language-model LoRA。若 adapter 还训练了 `emb_ext`、audio LM heads 或 codec，这些张量不能作为 llama.cpp runtime LoRA 使用，转换脚本会直接拒绝，避免静默丢权重。

## 3. 使用本仓库的 openmoss 扩展版

```powershell
Set-Location integrations\openmoss
git submodule update --init --recursive
```

不要另行克隆未经修改的 `pwilkin/openmoss` 来执行本文命令：上游版本没有本文使用的 `--lora`、`--parallel` 和 `--cache-type-*` 集成。本目录以该项目为基础，并锁定了与代码匹配的 llama.cpp submodule。

## 4. Windows 原生 CUDA 编译

官方 llama.cpp release 可以直接提供 `llama-quantize.exe`，但不能替代这里的 `moss-tts-server.exe`：后者包含 openmoss 的 codec、MOSS heads、多 LoRA 和 Session 池代码，必须从本目录编译。

使用“x64 Native Tools Command Prompt for VS 2022”，或先加载 VS x64 环境。建议为 CUDA 单独使用干净构建目录，避免旧 CMake cache 的 toolset 冲突：

```powershell
cmake -S . -B build-cuda -A x64 `
  -DGGML_CUDA=ON `
  -DCMAKE_CUDA_ARCHITECTURES=native `
  -T "cuda=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
cmake --build build-cuda --config Release -j
```

产物通常为：

```text
build-cuda\Release\moss-tts-cli.exe
build-cuda\Release\moss-tts-server.exe
build-cuda\bin\Release\llama.dll
build-cuda\bin\Release\ggml*.dll
```

运行时必须让 DLL 可见。可将 `build-cuda\bin\Release` 加入当前会话 PATH：

```powershell
$env:PATH = "integrations\openmoss\build-cuda\bin\Release;$env:PATH"
```

如果 CMake 报 `No CUDA toolset found`，需要在 Visual Studio Installer 中安装“使用 C++ 的桌面开发”，并安装与 VS 集成的 CUDA Toolkit；更换生成器或 CUDA toolset 后必须改用新构建目录。

## 5. 使用 openmoss 转换完整模型

在 openmoss 目录创建转换环境：

```powershell
py -3.10 -m venv .venv-convert
.\.venv-convert\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\third_party\llama.cpp\requirements\requirements-convert_lora_to_gguf.txt
python -m pip install safetensors huggingface_hub
```

转换未合并的原始基座，并保留随后转换 LoRA 所需的 Qwen3 HF 临时目录：

```powershell
python scripts\convert_hf_to_gguf.py `
  --moss-tts OpenMOSS-Team/MOSS-TTS-v1.5 `
  --codec OpenMOSS-Team/MOSS-Audio-Tokenizer `
  --backbone-dtype bf16 `
  --sidecar-dtype bf16 `
  --scratch-dir .\weights\convert-scratch `
  --keep-scratch `
  --output .\weights\moss-tts-base.gguf
```

如果模型已下载，`--moss-tts` 和 `--codec` 都可改为本地目录，从而完全离线转换。不要删除 `weights\convert-scratch\qwen3_backbone`，每个 PEFT adapter 转 GGUF 时都要用它作为未量化 base 描述。

必须得到两个文件：

```powershell
Test-Path .\weights\moss-tts-base.gguf
Test-Path .\weights\moss-tts-base.extras.gguf
Test-Path .\weights\convert-scratch\qwen3_backbone\config.json
```

第二个文件包含完整 AudioTokenizer、audio embeddings 和 LM heads。它不是本仓库 llama.cpp 管线使用的 `.npy + ONNX` 组合。

## 6. 量化 backbone 为 Q4_K_M

openmoss 的 in-tree llama.cpp 构建默认不生成量化工具。使用已经编译好的官方 llama.cpp `llama-quantize.exe`，并先将其所在目录加入 `PATH`：

```powershell
& llama-quantize.exe `
  --token-embedding-type bf16 `
  .\weights\moss-tts-base.gguf `
  .\weights\moss-tts-base-q4km.gguf `
  Q4_K_M
```

token embedding 必须保留 BF16。不要量化 sidecar；Q4_K_M 只用于 backbone。为保证 openmoss 能按 backbone 文件名准确找到 sidecar，复制成相同 stem：

```powershell
Copy-Item `
  .\weights\moss-tts-base.extras.gguf `
  .\weights\moss-tts-base-q4km.extras.gguf
```

最终必须同时存在：

```text
moss-tts-base-q4km.gguf
moss-tts-base-q4km.extras.gguf
```

### 6.1 转换每个 LoRA

```powershell
New-Item -ItemType Directory -Force .\weights\loras
python scripts\convert_moss_lora_to_gguf.py `
  ..\..\output\speaker-a `
  --base-qwen .\weights\convert-scratch\qwen3_backbone `
  --outfile .\weights\loras\speaker-a.gguf
```

每个音色重复一次。这里不会把 LoRA 合并进 Q4_K_M，也不会复制基座；生成的是可由 llama.cpp 在 context 上动态选择的 GGUF adapter。

## 7. 先用 CLI 验证

CLI 用于验证基座、sidecar、CUDA 和 codec 是否能完整工作；当前 CLI 不接受请求级 `--lora`，LoRA 选择要在下一节通过 server API 验证。

```powershell
.\build-cuda\Release\moss-tts-cli.exe `
  --model .\weights\moss-tts-base-q4km.gguf `
  --text "你好，这是 openmoss 基座连通性测试。" `
  --language zh `
  --max-new-tokens 600 `
  --output .\outputs\test.wav
```

参考音频克隆：

```powershell
.\build-cuda\Release\moss-tts-cli.exe `
  --model .\weights\moss-tts-base-q4km.gguf `
  --text "这是参考音色克隆测试。" `
  --reference .\reference.wav `
  --language zh `
  --output .\outputs\clone.wav
```

## 8. openmoss 原生常驻服务与自带 WebUI

```powershell
.\build-cuda\Release\moss-tts-server.exe `
  --model .\weights\moss-tts-base-q4km.gguf `
  --lora speaker-a=.\weights\loras\speaker-a.gguf `
  --host 127.0.0.1 `
  --port 8080 `
  --main-gpu 0 `
  --n-gpu-layers -1 `
  --n-ctx 8192 `
  --n-batch 512
```

启动阶段读取模型和所有 `--lora` 一次。`GET /health` 返回 `ok` 后，backbone、heads、codec 和 adapter 会一直保留在该 C++ 进程中；请求只选择 adapter、复用模型并重置生成状态，不会每次从磁盘重新载入。打开 `http://127.0.0.1:8080/` 即为 openmoss 自带界面。

测试 API：

```powershell
$body = @{ text = "常驻服务测试"; language = "zh"; lora = "speaker-a"; max_new_tokens = 600 } | ConvertTo-Json
Invoke-WebRequest `
  -Uri http://127.0.0.1:8080/tts `
  -Method Post `
  -ContentType application/json `
  -Body $body `
  -OutFile outputs\server-test.wav
```

## 9. 类似 `moss_tts_app.py` 的 Gradio 界面与首次懒加载

本仓库提供 `clis/moss_tts_openmoss_app.py`。这个 Python 进程只负责界面和 HTTP；真正推理由 openmoss 完成。第一次点击生成时它才启动 `moss-tts-server.exe`，等待 `/info` 可用并核对架构及 LoRA 列表后发送请求。server 随后一直运行并常驻显存，直至正常关闭 Gradio 进程。

```powershell
Set-Location ..\..
.\.venv\Scripts\Activate.ps1
python clis\moss_tts_openmoss_app.py `
  --openmoss-server integrations\openmoss\build-cuda\Release\moss-tts-server.exe `
  --model integrations\openmoss\weights\moss-tts-base-q4km.gguf `
  --lora speaker-a=integrations\openmoss\weights\loras\speaker-a.gguf `
  --lora speaker-b=integrations\openmoss\weights\loras\speaker-b.gguf `
  --parallel 2 `
  --main-gpu 0 `
  --n-gpu-layers -1 `
  --n-ctx 4096 `
  --n-batch 256 `
  --cache-type-k q8_0 `
  --cache-type-v q8_0 `
  --codec-cpu `
  --port 7860
```

访问 `http://127.0.0.1:7860`。加 `--preload` 可改成启动界面时立即加载。

这里的进程关系是：

```text
浏览器 → Gradio(Python，仅 UI/HTTP) → /tts → moss-tts-server.exe(C++/GGML)
```

Gradio 不导入 PyTorch、不持有模型，也不会为每次生成启动新进程。首次请求由带锁的懒加载逻辑启动一个 server；后续请求只调用 HTTP API。若同一地址已经有兼容的 openmoss server，Gradio 会核对其 `architecture` 和 `lora_adapters` 后直接复用，不重复加载。正常退出 Gradio 时，只会终止由它自己启动的子 server；手动启动的外部 server 不受影响。任务管理器强制结束 Python 或系统断电时，`atexit` 无法保证执行。

未指定 `--parallel` 时使用单 Session 串行执行；配置多个 Session 后，server 才会并行处理对应数量的请求。

工作区中的 `integrations/openmoss` 扩展版支持共享一份 Q4_K_M 基座的多 LoRA Session 池。`--parallel 2` 创建两个独立 llama context/KV cache，但不会复制 backbone、extras 或 adapter 权重；Gradio 同时允许两个请求进入 server。不同请求可在界面选择不同 LoRA。并发数增加会增加 KV cache 和工作区显存，建议从 2 开始测量。

## 10. 显存不足

优先顺序：

1. 使用 Q4_K_M backbone；
2. 加 `--codec-cpu`，只把 AudioTokenizer/codec 权重与计算放到 CPU；audio embeddings 和 LM heads 仍留在 GPU；
3. 加 `--cache-type-k q8_0 --cache-type-v q8_0`，把每个 Session 的 KV cache 从 F16 压到 Q8_0；
4. 把 `--n-ctx 8192` 降为 `4096` 或 `2048`；
5. 把 `--n-batch 512` 降为 `256` 或 `128`；
6. 把 `--parallel 2` 降为 `1`；
7. 把 `--n-gpu-layers -1` 改为 `28`、`24`、`20`，逐步减少 GPU offload。

Q4_K_M 是 backbone 权重量化，`--cache-type-* q8_0` 是 KV cache 量化，二者可以同时使用。Q8_0 通常是 8GB 显存的稳妥起点；Q4_0 更省显存，但更容易影响生成质量，应在自己的音色和语种集合上做 A/B 测试。

`--codec-cpu` 是精确拆分：GPU 保留 Q4_K_M backbone、audio embeddings、LM heads 和 KV cache，CPU 只承载 AudioTokenizer/codec。它主要减少 codec 占用的显存，代价是音频编码/解码变慢；文本/音频 token 自回归部分仍在 GPU。旧的 `--aux-cpu` 已移除，因为把所有辅助 graph 搬到 CPU 的减速明显，实际节省不划算。

不要使用 `--skip-codec` 做正常 WebUI 推理；该选项只输出 audio codes，不能生成最终 WAV。不要加 `--no-flash-attn`，除非当前 GPU/backend 的 Flash Attention 存在兼容问题；llama.cpp 要求量化 V cache 必须开启 Flash Attention，关闭时需同时改成 `--cache-type-v f16`。

## 11. 如何确认没有重复加载

1. 第一次请求前查看显存；第一次请求触发 server 后显存应明显上升。
2. 第二次请求时不应再次出现 openmoss 的模型加载日志。
3. 多次请求之间 `moss-tts-server.exe` PID 保持不变。
4. `http://127.0.0.1:8080/info` 的 request counter 持续增加。
5. 任务管理器或 `nvidia-smi` 中模型显存保持占用，直至关闭界面/server。

常驻的是模型权重和运行时。KV cache、请求输入及临时 graph buffer 会随生成长度变化，这是正常现象，不代表重新读取模型。

## 12. 多 LoRA 的容量边界

`--lora NAME=PATH` 指定的 adapter 在 server 启动时一次性读取并常驻，单个请求通过 JSON 的 `lora` 字段选择；切换音色不会重新读取磁盘。当前实现适合“少量热门音色常驻”。它不是无限容量的 LoRA LRU：如果要部署上百个音色，建议把音色分成多个有界热集 worker，再由上层网关按音色路由；不要把全部 adapter 无限制塞进一个 8GB 进程。
