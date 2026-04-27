"""Run baseline + alignment on every track in data/raw/, write results CSV.

Usage:
    python -m src.eval.batch_runner

Per-language alignment model: jonatasgrosman/wav2vec2-large-xlsr-53-<lang>.
For languages where we don't have a per-lang fine-tune, fall back to
facebook/mms-1b-all (1100+ langs, единый чекпоинт).
"""

from __future__ import annotations

import csv
import dataclasses
import json
import time
import traceback
import unicodedata
from pathlib import Path

from ..align import align
from ..baseline import transcribe
from .schema import Transcript
from .text_wer import compute as text_wer

ALIGN_MODELS = {
    "ru": "jonatasgrosman/wav2vec2-large-xlsr-53-russian",
    "es": "jonatasgrosman/wav2vec2-large-xlsr-53-spanish",
    "en": "jonatasgrosman/wav2vec2-large-xlsr-53-english",
    "fr": "jonatasgrosman/wav2vec2-large-xlsr-53-french",
    "it": "facebook/mms-1b-all",
    "pt": "facebook/mms-1b-all",
    "ja": "facebook/mms-1b-all",
    "pl": "facebook/mms-1b-all",
}


def _stem(name: str) -> str:
    return name.removesuffix(".m4a").removesuffix(".wav")


def _find_lyrics(lyrics_root: Path, lang: str, stem: str) -> Path | None:
    """Find lyrics file robustly across Unicode normalization forms.

    Yandex API may return filenames in NFD while lyrics writer used NFC
    (or vice versa). Try NFC, NFD, then case-insensitive glob fallback."""
    lang_dir = lyrics_root / lang
    if not lang_dir.exists():
        return None
    for form in ("NFC", "NFD"):
        candidate = lang_dir / f"{unicodedata.normalize(form, stem)}.txt"
        if candidate.exists():
            return candidate
    target_nfc = unicodedata.normalize("NFC", stem)
    for f in lang_dir.glob("*.txt"):
        if unicodedata.normalize("NFC", f.stem) == target_nfc:
            return f
    return None


def run(
    raw_root: Path = Path("data/raw"),
    lyrics_root: Path = Path("data/lyrics"),
    out_root: Path = Path("data/transcripts"),
    csv_path: Path = Path("docs/results.csv"),
) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for lang_dir in sorted(raw_root.iterdir()):
        if not lang_dir.is_dir():
            continue
        lang = lang_dir.name
        align_model = ALIGN_MODELS.get(lang, "facebook/mms-1b-all")

        for audio in sorted(lang_dir.glob("*.m4a")):
            stem = _stem(audio.name)
            row: dict = {"lang": lang, "track": stem}
            print(f"\n=== {lang} / {stem} ===")

            # --- BASELINE ---
            base_json = out_root / f"baseline_{lang}_{stem}.json"
            base_tele = out_root / f"baseline_{lang}_{stem}.telemetry.json"
            try:
                if base_json.exists() and base_tele.exists():
                    print("baseline: cached")
                    tele_b = json.loads(base_tele.read_text(encoding="utf-8"))
                else:
                    t0 = time.perf_counter()
                    transcript_b, tele_b = transcribe(str(audio), language=lang)
                    base_json.write_text(transcript_b.model_dump_json(indent=2), encoding="utf-8")
                    base_tele.write_text(json.dumps(tele_b, ensure_ascii=False, indent=2), encoding="utf-8")
                    print(f"baseline done in {time.perf_counter()-t0:.1f}s, rtf={tele_b['rtf']}")
                row["baseline_rtf"] = tele_b["rtf"]
                row["baseline_audio_s"] = tele_b["audio_s"]
                row["baseline_lang_prob"] = tele_b.get("lang_prob")
            except Exception as e:
                print(f"baseline FAIL: {e}")
                traceback.print_exc()
                row["baseline_error"] = str(e)

            # --- WER vs lyrics ---
            lyr = _find_lyrics(lyrics_root, lang, stem)
            if lyr is not None and base_json.exists():
                try:
                    wer = text_wer(lyr, base_json)
                    row["baseline_wer"] = round(wer["wer"], 4)
                    row["baseline_cer"] = round(wer["cer"], 4)
                    row["n_ref_words"] = wer["n_ref"]
                except Exception as e:
                    row["wer_error"] = str(e)
            else:
                row["baseline_wer"] = None

            # --- ALIGNMENT ---
            align_json = out_root / f"align_{lang}_{stem}.json"
            align_tele = out_root / f"align_{lang}_{stem}.telemetry.json"
            if lyr is None:
                print("alignment: SKIP (no lyrics)")
                row["align_status"] = "no_lyrics"
            else:
                try:
                    if align_json.exists() and align_tele.exists():
                        print("alignment: cached")
                        tele_a = json.loads(align_tele.read_text(encoding="utf-8"))
                    else:
                        t0 = time.perf_counter()
                        transcript_a, tele_a_obj = align(
                            str(audio), lyr.read_text(encoding="utf-8"),
                            language=lang, model_name=align_model,
                        )
                        align_json.write_text(transcript_a.model_dump_json(indent=2), encoding="utf-8")
                        tele_a = dataclasses.asdict(tele_a_obj)
                        align_tele.write_text(json.dumps(tele_a, ensure_ascii=False, indent=2), encoding="utf-8")
                        print(f"alignment done in {time.perf_counter()-t0:.1f}s, rtf={tele_a['rtf']}")
                    row["align_rtf"] = tele_a["rtf"]
                    row["align_coverage"] = tele_a["coverage"]
                    row["align_mean_conf"] = tele_a["mean_confidence"]
                    row["align_model"] = tele_a["model"]
                    row["align_status"] = "ok"
                except Exception as e:
                    print(f"alignment FAIL: {e}")
                    traceback.print_exc()
                    row["align_status"] = f"error: {e}"

            rows.append(row)

    if not rows:
        print("no rows")
        return
    fields = sorted({k for r in rows for k in r.keys()})
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {csv_path} ({len(rows)} rows)")


if __name__ == "__main__":
    run()
