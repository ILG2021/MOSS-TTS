# Upstream provenance

This directory vendors and modifies [`pwilkin/openmoss`](https://github.com/pwilkin/openmoss).

- Upstream base commit: `5a4eafc5d15cc0760e8883e1eda0fd5bde61da1f`
- Vendoring model: files in this directory are tracked directly by the parent MOSS-TTS repository.
- Exception: `third_party/llama.cpp` remains a Git submodule of the parent repository.
- llama.cpp commit used by this integration: `050ee92d04c2e1f639025786dea701c70e7d4204`

Local changes add Windows-focused Q4_K_M deployment, persistent multi-LoRA serving,
shared-weight inference contexts, quantized KV cache options, and the API-only
Gradio frontend integration documented in `docs/MULTI_LORA.md`.
