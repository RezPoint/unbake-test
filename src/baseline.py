"""Baseline ASR runner: faster-whisper large-v3 with word_timestamps=True.

This is the strawman to beat. Runs end-to-end on a single htdemucs vocal m4a,
emits a Transcript ready for src.eval.

Usage (locally with CPU, slow but works):
    python -m src.baseline path/to/vocals.m4a --language ru --model large-v3

In Colab (T4):
    !pip install -q faster-whisper
    from src.baseline import transcribe
    t = transcribe("vocals.m4a", language="ru")
"""

from __future__ import annotations

import argparse
import json
import time

from .eval.schema import Transcript, Word


def transcribe(
    audio_path: str,
    language: str | None = None,
    model_size: str = "large-v3",
    device: str = "auto",
    compute_type: str = "default",
    vad_filter: bool = True,
    beam_size: int = 5,
) -> tuple[Transcript, dict]:
    """Returns (transcript, telemetry).

    Telemetry: {load_s, infer_s, audio_s, rtf} for cost estimation.
    """
    from faster_whisper import WhisperModel  # imported lazily so eval has no GPU dep

    t0 = time.perf_counter()
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    t_load = time.perf_counter() - t0

    t1 = time.perf_counter()
    segments, info = model.transcribe(
        audio_path,
        language=language,
        word_timestamps=True,
        vad_filter=vad_filter,
        beam_size=beam_size,
        condition_on_previous_text=False,  # cuts hallucination cascades
    )
    words: list[Word] = []
    for seg in segments:
        if not seg.words:
            continue
        for w in seg.words:
            if w.word is None or w.start is None or w.end is None:
                continue
            words.append(
                Word(
                    text=w.word.strip(),
                    start=float(w.start),
                    end=float(w.end),
                    confidence=float(w.probability) if w.probability is not None else None,
                )
            )
    t_infer = time.perf_counter() - t1

    transcript = Transcript(language=info.language, words=words)
    telemetry = {
        "load_s": round(t_load, 2),
        "infer_s": round(t_infer, 2),
        "audio_s": round(info.duration, 2),
        "rtf": round(t_infer / max(info.duration, 0.01), 3),
        "detected_lang": info.language,
        "lang_prob": round(info.language_probability, 3),
    }
    return transcript, telemetry


def _cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--language", default=None)
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--compute-type", default="default")
    ap.add_argument("--out", default=None, help="path to write transcript JSON")
    args = ap.parse_args()

    transcript, telemetry = transcribe(
        args.audio,
        language=args.language,
        model_size=args.model,
        device=args.device,
        compute_type=args.compute_type,
    )
    print(json.dumps(telemetry, indent=2, ensure_ascii=False))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(transcript.model_dump_json(indent=2))
        print(f"\nwrote {args.out}")
    else:
        print()
        print(transcript.model_dump_json(indent=2)[:2000])


if __name__ == "__main__":
    _cli()
