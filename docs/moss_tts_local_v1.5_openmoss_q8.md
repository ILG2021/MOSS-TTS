# MOSS-TTS Local v1.5 OpenMOSS Q8_0 部署指南

本文说明如何为 OpenMOSS C++ 后端准备完整的 MOSS-TTS Local Transformer
v1.5 Q8_0 模型。最终需要两个同名前缀的文件：

```text
moss-tts-local-q8_0.gguf
moss-tts-local-q8_0.extras.gguf
```

第一个文件是 Qwen3 backbone，第二个文件包含 Local Transformer、音频
embedding、LM heads 和 MOSS-Audio-Tokenizer-v2。

Backbone 和 sidecar 的精度可以独立选择。推荐优先使用：

```text
Backbone：Q8_0
Sidecar：BF16 混合精度
```

该组合由 backbone 提供主要的显存节省，同时让 Local Transformer 和 Audio
Tokenizer 保持较高精度。显存更紧张时，再将 sidecar 改为 Q8_0 混合精度。

## 1. 环境准备

需要：

- Python 3.10 或更高版本
- Git 与 Git LFS
- CMake 3.18 或更高版本
- Windows 上的 Visual Studio 2022 C++ 工具链
- CUDA Toolkit（CUDA 构建时需要）
- `llama-quantize`，来自完整的 llama.cpp 构建或 llama.cpp 发行包

安装模型转换依赖：

```powershell
python -m pip install safetensors numpy huggingface_hub gguf
```

初始化子模块：

```powershell
git submodule update --init --recursive
```

## 2. 编译 OpenMOSS CUDA 后端

在 Visual Studio Developer PowerShell 中执行：

```powershell
cmake -S integrations/openmoss `
  -B integrations/openmoss/build-cuda `
  -DGGML_CUDA=ON

cmake --build integrations/openmoss/build-cuda `
  --config Release `
  -j
```

生成的服务端通常位于：

```text
integrations/openmoss/build-cuda/Release/moss-tts-server.exe
```

如果 DLL 位于 `build-cuda/bin/Release`，启动脚本会把该目录加入子进程的
`PATH`。手动启动时需要自行确保 DLL 可被找到。

## 3. 转换未量化模型

以下命令从两个官方 Hugging Face 仓库读取权重：

- `OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5`
- `OpenMOSS-Team/MOSS-Audio-Tokenizer-v2`

```powershell
python integrations/openmoss/scripts/convert_hf_to_gguf.py `
  --moss-tts OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 `
  --codec OpenMOSS-Team/MOSS-Audio-Tokenizer-v2 `
  --output integrations/openmoss/weights/moss-tts-local.gguf `
  --sidecar-dtype bf16
```

这一步生成：

```text
moss-tts-local.gguf
moss-tts-local.extras.gguf
```

此时 backbone 仍保持源模型精度；sidecar 采用 BF16 混合精度：

- 普通权重：BF16
- Local audio embedding：F16
- 数值敏感张量：F32

audio embedding 必须保持 F16，因为当前 C++ Local decoder 会直接读取其
F16 行数据。请使用本仓库修正后的转换脚本。

## 4. 获取 llama-quantize

OpenMOSS 内嵌的 llama.cpp 默认可能不构建命令行工具。可以使用任意兼容的
完整 llama.cpp 构建或官方发行包中的：

```text
llama-quantize.exe
```

如果自行编译 llama.cpp：

```powershell
cmake -S integrations/openmoss/third_party/llama.cpp `
  -B integrations/openmoss/third_party/llama.cpp/build-tools `
  -DGGML_CUDA=ON `
  -DLLAMA_BUILD_TOOLS=ON

cmake --build integrations/openmoss/third_party/llama.cpp/build-tools `
  --config Release `
  --target llama-quantize `
  -j
```

可执行文件通常位于：

```text
integrations/openmoss/third_party/llama.cpp/build-tools/bin/Release/llama-quantize.exe
```

## 5. 将 backbone 量化为 Q8_0

```powershell
integrations/openmoss/third_party/llama.cpp/build-tools/bin/Release/llama-quantize.exe `
  --token-embedding-type bf16 `
  integrations/openmoss/weights/moss-tts-local.gguf `
  integrations/openmoss/weights/moss-tts-local-q8_0.gguf `
  Q8_0
```

不要省略 `--token-embedding-type bf16`。文本 embedding 通过行索引访问，必须
保持非量化格式。

不要将 `llama-quantize` 用于 `.extras.gguf`。Sidecar 包含自定义 GGML 权重，
必须由 OpenMOSS 转换器写入。

## 6. 为 Q8 backbone 生成匹配的 sidecar

OpenMOSS 根据 backbone 文件名寻找 sidecar。例如：

```text
moss-tts-local-q8_0.gguf
→ moss-tts-local-q8_0.extras.gguf
```

因此需要为量化后的文件生成同名前缀 sidecar。

### 推荐：BF16 sidecar

```powershell
python integrations/openmoss/scripts/convert_hf_to_gguf.py `
  --moss-tts OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 `
  --codec OpenMOSS-Team/MOSS-Audio-Tokenizer-v2 `
  --output integrations/openmoss/weights/moss-tts-local-q8_0.gguf `
  --sidecar-dtype bf16 `
  --sidecar-only
```

该配置采用：

- Qwen3 backbone：Q8_0
- Local Transformer 和 Audio Tokenizer 大部分权重：BF16
- Local audio embedding：F16
- 数值敏感张量：F32

这是质量和显存之间更稳妥的默认选择。

### 更省显存：Q8_0 sidecar

```powershell
python integrations/openmoss/scripts/convert_hf_to_gguf.py `
  --moss-tts OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 `
  --codec OpenMOSS-Team/MOSS-Audio-Tokenizer-v2 `
  --output integrations/openmoss/weights/moss-tts-local-q8_0.gguf `
  --sidecar-dtype q8_0 `
  --sidecar-only
```

`--sidecar-only` 会保留已经量化好的 backbone，只写入：

```text
integrations/openmoss/weights/moss-tts-local-q8_0.extras.gguf
```

两种 sidecar 使用相同文件名，不能同时放在同一目录中。切换精度时应重新生成
并覆盖 `.extras.gguf`，或者为两套模型使用不同的 backbone 文件名前缀。

精度组合对比：

| Backbone | Sidecar | 音频相关精度 | 显存 | 建议用途 |
|---|---|---|---|---|
| Q8_0 | BF16 混合精度 | 较高 | 中等 | 推荐默认配置 |
| Q8_0 | Q8_0 混合精度 | 略低 | 较低 | 显存紧张 |
| BF16 | BF16 混合精度 | 最高 | 最高 | 质量验证与基准 |

## 7. 检查文件

```powershell
Get-Item `
  integrations/openmoss/weights/moss-tts-local-q8_0.gguf, `
  integrations/openmoss/weights/moss-tts-local-q8_0.extras.gguf
```

两个文件必须位于同一目录，并且除 `.extras` 外名称完全一致。

可以使用 OpenMOSS 信息工具检查架构和张量元数据：

```powershell
integrations/openmoss/build-cuda/Release/moss-tts-info.exe `
  integrations/openmoss/weights/moss-tts-local-q8_0.gguf
```

预期架构为：

```text
moss_tts_local
```

## 8. 启动 Gradio 应用

```powershell
python clis/moss_tts_local_v1.5_openmoss_app.py `
  --openmoss-server integrations/openmoss/build-cuda/Release/moss-tts-server.exe `
  --model integrations/openmoss/weights/moss-tts-local-q8_0.gguf `
  --preload
```

默认地址：

```text
http://127.0.0.1:7861
```

服务端默认使用 `voices` 目录启用 voice registry。批量合成会注册一个临时
voice，使整批复用同一份参考音频 tokens，完成或失败后自动删除。

## 9. 显存不足时

首先尝试把 audio codec 放到 CPU：

```powershell
python clis/moss_tts_local_v1.5_openmoss_app.py `
  --openmoss-server integrations/openmoss/build-cuda/Release/moss-tts-server.exe `
  --model integrations/openmoss/weights/moss-tts-local-q8_0.gguf `
  --codec-cpu `
  --preload
```

这会降低显存占用，但参考音频编码和最终波形解码会变慢。文本 tokenizer 一直
运行在 CPU，不受该参数影响。

还可以减少上下文或 batch：

```text
--n-ctx 4096 --n-batch 256
```

Local Transformer 当前只支持：

```text
--parallel 1
```

## 10. 常见错误

### `audio embedding 0 is not f16`

`.extras.gguf` 由旧转换器生成，audio embedding 被写成 BF16 或 Q8_0。使用
当前转换器重新执行第 6 步。无需重新量化 backbone。

### 找不到 `.extras.gguf`

Backbone 与 sidecar 前缀不匹配。以下组合不会自动匹配：

```text
moss-tts-local-q8_0.gguf
moss-tts-local.extras.gguf
```

必须改为：

```text
moss-tts-local-q8_0.gguf
moss-tts-local-q8_0.extras.gguf
```

### `voice registry disabled`

正在运行的是旧版 `moss-tts-server.exe`，或者端口上已有旧进程。重新编译服务端，
结束旧进程后再启动。也可以手动启动并显式传入：

```text
--voice-dir voices
```

### 修改 C++ 后仍出现旧行为

确认 8080 端口没有残留 server，然后检查传给 `--openmoss-server` 的确切路径。
重新构建 C++ 后必须重启该进程，仅重启 Gradio 不够。

## 11. 最终文件布局示例

```text
integrations/openmoss/
├── build-cuda/
│   └── Release/
│       └── moss-tts-server.exe
├── voices/
└── weights/
    ├── moss-tts-local-q8_0.gguf
    └── moss-tts-local-q8_0.extras.gguf
```

部署完成后，未量化的 `moss-tts-local.gguf` 和
`moss-tts-local.extras.gguf` 可以另行归档。确认 Q8 模型可正常生成之前不要删除
原始文件。
