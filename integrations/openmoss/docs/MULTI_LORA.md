# Q4_K_M multi-LoRA serving

This fork adds runtime LoRA selection and a pool of llama contexts for the
`moss_tts_delay` family. Backbone, MOSS auxiliary weights, codec weights, and
LoRA adapters are loaded once. Each parallel slot owns an independent llama
context and KV cache.

## Convert the base while retaining the extracted Qwen3 checkpoint

```powershell
python scripts\convert_hf_to_gguf.py `
  --moss-tts OpenMOSS-Team/MOSS-TTS-v1.5 `
  --codec OpenMOSS-Team/MOSS-Audio-Tokenizer `
  --scratch-dir weights\convert-scratch `
  --keep-scratch `
  --output weights\moss-tts-base.gguf
```

Quantize the backbone and copy the sidecar to the matching stem:

```powershell
llama-quantize.exe --token-embedding-type bf16 `
  weights\moss-tts-base.gguf weights\moss-tts-base-q4km.gguf Q4_K_M
Copy-Item weights\moss-tts-base.extras.gguf weights\moss-tts-base-q4km.extras.gguf
```

## Convert each PEFT adapter

MOSS adapters use a `language_model` prefix while the openmoss backbone is a
plain Qwen3 checkpoint. The wrapper normalizes that prefix and delegates the
actual GGUF serialization to llama.cpp.

```powershell
python scripts\convert_moss_lora_to_gguf.py `
  D:\loras\speaker-a `
  --base-qwen weights\convert-scratch\qwen3_backbone `
  --outfile weights\loras\speaker-a.gguf
```

Repeat for every speaker. Only the default language-model LoRA targets are
supported; adapters that modify MOSS embeddings, heads, or codec tensors are
rejected.

## Start the server

```powershell
.\build-cuda\Release\moss-tts-server.exe `
  --model weights\moss-tts-base-q4km.gguf `
  --lora speaker-a=weights\loras\speaker-a.gguf `
  --lora speaker-b=weights\loras\speaker-b.gguf `
  --parallel 2 `
  --n-ctx 4096 `
  --cache-type-k q8_0 `
  --cache-type-v q8_0 `
  --n-gpu-layers -1
```

Call the native endpoint with a request-level adapter:

```json
{
  "text": "Hello from speaker A.",
  "language": "en",
  "lora": "speaker-a",
  "lora_scale": 1.0
}
```

Omit `lora` to use the base model. Unknown adapters return HTTP 400.

`--parallel` controls independent llama contexts, not copies of model weights.
Each slot still adds a KV cache and compute workspace. Start at 2, measure peak
VRAM, and only then increase it. Streaming is intentionally disabled when more
than one slot is configured because codec streaming workspaces are not yet
session-local. Non-streaming codec encode/decode and shared auxiliary GGML
graphs are protected by short critical sections.

Different LoRA requests run in separate contexts. This improves aggregate
throughput but may reduce the tokens/second of each request when both contexts
compete for one GPU. It is not continuous batching.

## 8 GB starting point

Q4_K_M quantizes the shared backbone; it does not quantize each session's KV
cache. For two concurrent sessions on an 8 GB card, start with:

```powershell
--parallel 2 --n-ctx 4096 --n-batch 256 `
--cache-type-k q8_0 --cache-type-v q8_0 --codec-cpu
```

`q8_0` roughly halves KV memory relative to `f16` with a smaller quality risk
than `q4_0`. If memory is still insufficient, first reduce `--n-ctx` to 2048,
then try one slot, and only then reduce GPU layers. `--codec-cpu` moves only the
AudioTokenizer codec weights and graphs to system RAM; audio embeddings and LM
heads remain on GPU. Codec encode/decode becomes slower, while autoregressive
token generation keeps its GPU auxiliary path.

All configured LoRAs are loaded once at server startup and remain resident.
This is ideal for a small hot set of voices. It is not an unbounded adapter
cache: for hundreds of voices, run several worker processes with bounded hot
sets behind a router, or add a ref-counted LoRA LRU before production use.
