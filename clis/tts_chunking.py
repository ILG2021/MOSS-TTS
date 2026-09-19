"""Character budgeting and rolling references for the Gradio TTS app."""
import gc
import json
import math
from pathlib import Path
import re
import threading


def count_chars(text):
    """Count all Unicode characters, including punctuation and whitespace."""
    return len(text)


def dataset_stats(path, text_field="text"):
    count = total = 0
    minimum = None
    maximum = 0
    with Path(path).open(encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)[text_field]
                if not isinstance(value, str) or not count_chars(value):
                    raise ValueError("text must be a non-empty string")
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid {text_field}: {exc}") from exc
            size = count_chars(value)
            count += 1
            total += size
            minimum = size if minimum is None else min(minimum, size)
            maximum = max(maximum, size)
    if not count:
        raise ValueError(f"{path}: no text records")
    return dict(records=count, total_chars=total, mean_chars=total / count,
                min_chars=minimum, max_chars=maximum,
                recommended_chunk_chars=max(1, math.floor(total / count + 0.5)))


def split_text(text, max_chars):
    return list(iter_text_chunks(text, max_chars))


def iter_text_chunks(text, max_chars):
    """F5-style punctuation packing, with a hard fallback for long sentences.

    Unlike F5's UTF-8 byte budget, use the dataset's Unicode character count.
    Preserve all original characters, including spaces between English words.
    """
    if not isinstance(max_chars, int) or max_chars < 1:
        raise ValueError("分段总字数必须为正整数")
    boundaries = re.finditer(r"(?<=[;:,.!?])(?=\s)|(?<=[；：，。！？])|(?<=\n)|\Z", text)
    current, start = "", 0
    for boundary in boundaries:
        piece = text[start:boundary.end()]
        start = boundary.end()
        while count_chars(piece) > max_chars:
            if current:
                yield current
                current = ""
            cut = max_chars
            # Prefer a word boundary in the latter half of the available span.
            spaces = [m.end() for m in re.finditer(r"\s+", piece[:cut])]
            if spaces and spaces[-1] >= cut // 2:
                cut = spaces[-1]
            yield piece[:cut]
            piece = piece[cut:]
        if count_chars(current + piece) > max_chars:
            yield current
            current = ""
        current += piece
    if current:
        yield current


def reference_tail(audio, sample_rate):
    """Return at most 10 seconds, starting at a low-energy point near -10s."""
    import numpy as np
    audio = np.asarray(audio, dtype=np.float32)
    if sample_rate <= 0 or audio.ndim != 1 or not audio.size:
        raise ValueError("Expected non-empty mono audio and a positive sample rate")
    start = max(0, len(audio) - round(10 * sample_rate))
    if start == 0:
        return audio.copy()
    # Search only forward, so the selected tail never exceeds ten seconds.
    window = max(1, round(0.25 * sample_rate))
    end = min(start + round(2 * sample_rate), len(audio) - window)
    probes = list(range(start, end + 1, 512))
    best = min(probes, key=lambda p: float(np.mean(audio[p:p + window].astype(np.float64) ** 2)))
    return audio[best:].copy()


_ASR_MODEL = None
_ASR_CONFIG = None
_ASR_LOAD_LOCK = threading.Lock()


def load_asr(model_path="large-v3-turbo", device="cpu"):
    global _ASR_MODEL, _ASR_CONFIG
    current_config = (model_path, device)
    with _ASR_LOAD_LOCK:
        if _ASR_MODEL is not None and _ASR_CONFIG == current_config:
            return _ASR_MODEL
        if _ASR_MODEL is not None:
            # 类型/设备发生改变，释放旧模型显存
            _ASR_MODEL = None
            _ASR_CONFIG = None
            gc.collect()

        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError('请安装 ASR 依赖：pip install -e ".[app-asr]"') from exc

        model = WhisperModel(
            model_path,
            device=device,
            compute_type="int8" if device == "cpu" else "float16",
        )
        _ASR_MODEL = model
        _ASR_CONFIG = current_config
        return model


MOSS_TO_WHISPER_LANGUAGE = {
    "chinese": "zh", "cantonese": "yue", "english": "en",
    "arabic": "ar", "czech": "cs", "danish": "da", "dutch": "nl",
    "finnish": "fi", "french": "fr", "german": "de", "greek": "el",
    "hebrew": "he", "hindi": "hi", "hungarian": "hu", "italian": "it",
    "japanese": "ja", "korean": "ko", "macedonian": "mk", "malay": "ms",
    "persian (farsi)": "fa", "polish": "pl", "portuguese": "pt",
    "romanian": "ro", "russian": "ru", "spanish": "es", "swahili": "sw",
    "swedish": "sv", "tagalog": "tl", "thai": "th", "turkish": "tr",
    "vietnamese": "vi",
}


def whisper_language(language_tag):
    """Translate a MOSS UI language name to a faster-whisper language code."""
    tag = (language_tag or "").strip().lower()
    if tag in {"", "auto", "auto (omit)"}:
        return None
    if tag in MOSS_TO_WHISPER_LANGUAGE:
        return MOSS_TO_WHISPER_LANGUAGE[tag]
    if tag in MOSS_TO_WHISPER_LANGUAGE.values():
        return tag
    raise ValueError(f"不支持的 MOSS 语言标签：{language_tag}")


def transcribe_reference(path, model_path="large-v3-turbo", device="cpu", language_tag=None):
    """ASR 转录时强制使用指定语言，避免模型自己乱跳语言"""
    language = whisper_language(language_tag)
    options = {
        "language": language,
        "beam_size": 5,
        "vad_filter": True,
        "condition_on_previous_text": False,
        "initial_prompt": "这是一个中文句子，带标点。" if language == "zh" else None
    }
    segments, _ = load_asr(model_path, device).transcribe(str(path), **options)
    text = "".join(segment.text for segment in segments).strip()
    if not text:
        raise ValueError("参考音频未识别出文本，请使用包含清晰语音的参考音频。")
    if not re.search(r"[。？！，、；：…—～.?!,;:~\-\"'\)\]\}\>”’）》】〉]$", text):
        text += "。" if re.search(r"[\u4e00-\u9fff]", text) else "."
    return text
