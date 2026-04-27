"""Text-only WER between lyrics .txt and Transcript JSON.

Lyrics from Genius / LRCLib often contain ad-libs in parentheses
(пау-пау-пау), section labels [Chorus], structural cues. We strip
those out before WER so the metric reflects content recognition,
not annotation differences.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import jiwer

from .schema import Transcript

_PAREN = re.compile(r"\([^)]*\)|\[[^\]]*\]")
_PUNCT = re.compile(r"[^\w\s'-]", re.UNICODE)
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("ё", "е")
    text = _PAREN.sub(" ", text)
    text = _PUNCT.sub(" ", text)
    text = _WS.sub(" ", text).strip()
    return text


def lyrics_words(path: Path) -> list[str]:
    return normalize(path.read_text(encoding="utf-8")).split()


def transcript_words(t: Transcript) -> list[str]:
    return normalize(" ".join(w.text for w in t.words)).split()


def compute(lyrics_path: Path, transcript_path: Path) -> dict:
    ref = lyrics_words(lyrics_path)
    hyp = transcript_words(Transcript.model_validate_json(transcript_path.read_text(encoding="utf-8")))
    ref_s, hyp_s = " ".join(ref), " ".join(hyp)
    out = jiwer.process_words(ref_s, hyp_s)
    return {
        "wer": out.wer,
        "cer": jiwer.cer(ref_s, hyp_s),
        "n_ref": len(ref),
        "n_hyp": len(hyp),
        "substitutions": out.substitutions,
        "deletions": out.deletions,
        "insertions": out.insertions,
        "hits": out.hits,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        sys.exit("usage: python -m src.eval.text_wer <lyrics.txt> <transcript.json>")
    res = compute(Path(sys.argv[1]), Path(sys.argv[2]))
    print(json.dumps(res, ensure_ascii=False, indent=2))
