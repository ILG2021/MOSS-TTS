"""Token-level parity check: local MossTTSDelayProcessor vs. the official HF processors.

Checks
  1. moss_tts template vs OpenMOSS-Team/MOSS-TTS-v1.5 remote processor  (regression)
  2. ttsd template     vs OpenMOSS-Team/MOSS-TTSD-v1.0 remote processor (new branch)

No GPU and no audio codec are needed (random audio codes are used). Only the
config / tokenizer / processing_moss_tts.py of each repo are downloaded.

    python scripts/check_ttsd_prompt.py
    python scripts/check_ttsd_prompt.py --v15 D:/models/MOSS-TTS-v1.5 --ttsd D:/models/MOSS-TTSD-v1.0

Exit code 0 = all cases identical.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from moss_tts_delay.processing_moss_tts import MossTTSDelayProcessor  # noqa: E402

TEXTS = [
    "[S1]弟兄姊妹们，今天我们来看一段经文。你看，这里说得很清楚，对吧？",
    "[S1]Hello there, this is a short English sentence. [S2]And this is speaker two.",
    "[S1]数字123和符号……还有“引号”，以及换行\n第二行。",
]


def build_pair(repo: str, template: str):
    config = AutoConfig.from_pretrained(repo, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    remote_cls = get_class_from_dynamic_module(
        "processing_moss_tts.MossTTSDelayProcessor", repo
    )
    remote = remote_cls(tokenizer=tokenizer, audio_tokenizer=None, model_config=config)
    local = MossTTSDelayProcessor(
        tokenizer=tokenizer, audio_tokenizer=None, model_config=config, prompt_template=template
    )
    return config, local, remote


def cases(n_vq_codes: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)

    def codes(t):
        return torch.randint(0, 1024, (t, n_vq_codes), generator=g)

    for text in TEXTS:
        # generation, no reference
        yield "gen/no-ref", "generation", [{"role": "user", "text": text}]
        # generation, one reference
        yield "gen/ref", "generation", [{"role": "user", "text": text, "reference": [codes(37)]}]
        # generation, [S1]=None + [S2]=ref (TTSD renders "[S1]: None")
        yield "gen/null+ref", "generation", [
            {"role": "user", "text": text, "reference": [None, codes(23)]}
        ]
        # continuation: user + assistant prefix audio
        yield "continuation", "continuation", [
            {"role": "user", "text": text},
            {"role": "assistant", "audio_codes_list": [codes(41)]},
        ]


def run(name: str, repo: str, template: str, codes_n_vq: int | None) -> int:
    config, local, remote = build_pair(repo, template)
    n_vq = int(config.n_vq)
    codes_n_vq = codes_n_vq or n_vq
    print(f"\n== {name}: {repo}  (model n_vq={n_vq}, local template={local.prompt_template}, "
          f"codes n_vq={codes_n_vq})")
    failures = 0
    for label, mode, conv in cases(codes_n_vq):
        try:
            a = local([conv], mode=mode)["input_ids"]
            b = remote([conv], mode=mode)["input_ids"]
        except Exception as exc:  # report and continue
            print(f"  [ERROR] {label}: {type(exc).__name__}: {exc}")
            failures += 1
            continue
        same = a.shape == b.shape and torch.equal(a, b)
        print(f"  [{'OK' if same else 'DIFF'}] {label:13s} {mode:12s} shape={tuple(a.shape)}"
              + ("" if same else f" vs {tuple(b.shape)}"))
        if not same:
            failures += 1
            ta = local.tokenizer.decode(a[0, :, 0].tolist())
            tb = remote.tokenizer.decode(b[0, :, 0].tolist())
            for i, (x, y) in enumerate(zip(ta, tb)):
                if x != y:
                    print(f"     first text diff at char {i}: local={ta[max(0, i-40):i+40]!r}")
                    print(f"                                 remote={tb[max(0, i-40):i+40]!r}")
                    break
    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v15", default="OpenMOSS-Team/MOSS-TTS-v1.5")
    ap.add_argument("--ttsd", default="OpenMOSS-Team/MOSS-TTSD-v1.0")
    ap.add_argument("--skip-v15", action="store_true")
    ap.add_argument("--skip-ttsd", action="store_true")
    args = ap.parse_args()

    failures = 0
    if not args.skip_v15:
        failures += run("v1.5 regression", args.v15, "moss_tts", None)
    if not args.skip_ttsd:
        failures += run("TTSD parity (16-layer codes)", args.ttsd, "ttsd", 16)
        failures += run("TTSD parity (32-layer codes, truncation)", args.ttsd, "ttsd", 32)

    print("\nALL IDENTICAL" if failures == 0 else f"\n{failures} case(s) differ")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
