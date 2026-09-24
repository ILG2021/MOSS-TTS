# MOSS-TTS Local v1.5 with openmoss

`clis/moss_tts_local_v1.5_openmoss_app.py` is the Gradio application for the
native C++ `moss_tts_local` pipeline. It is separate from
`clis/moss_tts_openmoss_app.py`, which remains the delay-pattern application.

The local-pattern application preserves the model's 48 kHz stereo output and
uses openmoss's `ref_text` continuation protocol. Plain voice cloning sends a
reference without `ref_text`; continuation transcribes the reference and sends
the transcript and audio together.

## Prepare the model

Build openmoss as described in `integrations/openmoss/README.md`, then convert
the Local Transformer and Audio Tokenizer v2 checkpoints:

```powershell
python integrations/openmoss/scripts/convert_hf_to_gguf.py `
  --moss-tts OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 `
  --codec OpenMOSS-Team/MOSS-Audio-Tokenizer-v2 `
  --output integrations/openmoss/weights/moss-tts-local.gguf
```

Keep `moss-tts-local.gguf` and `moss-tts-local.extras.gguf` in the same
directory. The sidecar contains the 12 local codebook heads, local transformer,
and stereo codec weights.

## Run

```powershell
python clis/moss_tts_local_v1.5_openmoss_app.py `
  --openmoss-server integrations/openmoss/build-cuda/Release/moss-tts-server.exe `
  --model integrations/openmoss/weights/moss-tts-local.gguf `
  --preload
```

The UI defaults to `http://127.0.0.1:7861`. The app starts a private native
server at `127.0.0.1:8080`, or connects to an already running compatible
server. If port 8080 is occupied by a delay-pattern server, choose another port
with `--openmoss-port`.

Local-pattern inference currently requires `--parallel 1`; the app enforces
this because the depth transformer and codec retain shared mutable graphs.
Use `--codec-cpu` only when the server is also started with that setting.

## Batch synthesis

Paste one utterance per line and click **批量合成**. Blank lines are ignored.
If a line starts with digits, the complete leading digit run is used as its
line number and removed from the synthesized text. Other lines receive a
one-based sequence number. For example:

```text
12 This is the twelfth sentence.
第二句没有显式编号。
```

Each line produces one 48 kHz stereo WAV, even when that line is internally
split at punctuation because it exceeds the character limit. Output names use
`NNNN_sanitized-text.wav`; the text portion is limited to 100 characters.
Duplicate names receive a numeric suffix. After all lines finish, the app
returns a ZIP containing the WAV files. Per-line WAV files and rolling-reference
audio are temporary and are removed after the ZIP is finalized; only the ZIP is
retained in the output directory.

Continuation mode needs the optional ASR dependency:

```powershell
pip install -e ".[app-asr]"
```
