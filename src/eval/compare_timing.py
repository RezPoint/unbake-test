"""Compare two Transcripts by token-equality + timing offset.

Use case: alignment (silver standard) vs ASR baseline. For words
present in both, report the per-word start-time offset distribution.
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path
from statistics import mean, median

import numpy as np

from .schema import Transcript


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower().replace("ё", "е")
    return re.sub(r"[^\w'-]", "", s, flags=re.UNICODE)


def compare(ref_path: Path, hyp_path: Path) -> dict:
    ref = Transcript.model_validate_json(ref_path.read_text(encoding="utf-8"))
    hyp = Transcript.model_validate_json(hyp_path.read_text(encoding="utf-8"))

    # Greedy match: for each ref word in order, find nearest hyp word with
    # same normalized text within ±5s and not yet consumed.
    used = [False] * len(hyp.words)
    offsets = []
    matched = 0
    for r in ref.words:
        rn = _norm(r.text)
        if not rn:
            continue
        best = -1
        best_dt = 5.0
        for j, h in enumerate(hyp.words):
            if used[j]:
                continue
            if _norm(h.text) != rn:
                continue
            dt = abs(h.start - r.start)
            if dt < best_dt:
                best_dt = dt
                best = j
        if best >= 0:
            used[best] = True
            offsets.append(hyp.words[best].start - r.start)
            matched += 1

    abs_off = [abs(o) for o in offsets]
    return {
        "n_ref": len(ref.words),
        "n_hyp": len(hyp.words),
        "matched": matched,
        "match_rate": round(matched / len(ref.words), 4) if ref.words else 0,
        "mean_abs_offset_s": round(mean(abs_off), 3) if abs_off else None,
        "median_abs_offset_s": round(median(abs_off), 3) if abs_off else None,
        "p95_abs_offset_s": round(float(np.percentile(abs_off, 95)), 3) if abs_off else None,
        "mean_signed_offset_s": round(mean(offsets), 3) if offsets else None,
    }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python -m src.eval.compare_timing <ref.json> <hyp.json>")
    print(json.dumps(compare(Path(sys.argv[1]), Path(sys.argv[2])), ensure_ascii=False, indent=2))
