from __future__ import annotations

import re
from dataclasses import dataclass

import jiwer

from .schema import Reference, Transcript, Word


_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)


def _normalize(text: str) -> str:
    return _PUNCT.sub(" ", text.lower()).split().__str__()


def _normalize_words(words: list[Word]) -> list[str]:
    out: list[str] = []
    for w in words:
        cleaned = _PUNCT.sub("", w.text.lower()).strip()
        if cleaned:
            out.append(cleaned)
    return out


@dataclass
class TextMetrics:
    wer: float
    cer: float
    n_ref_words: int
    n_hyp_words: int


def text_metrics(ref: Reference, hyp: Transcript) -> TextMetrics:
    ref_tokens = _normalize_words(ref.words)
    hyp_tokens = _normalize_words(hyp.words)
    ref_str = " ".join(ref_tokens)
    hyp_str = " ".join(hyp_tokens) if hyp_tokens else " "
    return TextMetrics(
        wer=jiwer.wer(ref_str, hyp_str),
        cer=jiwer.cer(ref_str, hyp_str),
        n_ref_words=len(ref_tokens),
        n_hyp_words=len(hyp_tokens),
    )


@dataclass
class TimingMetrics:
    """How well do hypothesis word boundaries match reference word boundaries.

    Computed only on words that align (Levenshtein-matched) between ref and hyp.
    """

    mean_abs_offset: float
    p50_offset: float
    p95_offset: float
    coverage: float  # fraction of ref words that found a hyp match


def timing_metrics(ref: Reference, hyp: Transcript, max_offset_s: float = 5.0) -> TimingMetrics:
    ref_norm = _normalize_words(ref.words)
    hyp_norm = _normalize_words(hyp.words)

    # Pair ref words to hyp words by token equality + nearest-in-time, single pass.
    used_hyp: set[int] = set()
    offsets: list[float] = []
    matched = 0
    for i, ref_w in enumerate(ref.words):
        token = _normalize_words([ref_w])
        if not token:
            continue
        token = token[0]
        ref_mid = (ref_w.start + ref_w.end) / 2
        best_j = -1
        best_d = max_offset_s
        for j, h in enumerate(hyp.words):
            if j in used_hyp:
                continue
            if j >= len(hyp_norm):
                break
            if hyp_norm[j] != token:
                continue
            d = abs((h.start + h.end) / 2 - ref_mid)
            if d < best_d:
                best_d = d
                best_j = j
        if best_j >= 0:
            used_hyp.add(best_j)
            offsets.append(best_d)
            matched += 1

    if not offsets:
        return TimingMetrics(0.0, 0.0, 0.0, coverage=0.0)
    offsets.sort()
    n = len(offsets)
    return TimingMetrics(
        mean_abs_offset=sum(offsets) / n,
        p50_offset=offsets[n // 2],
        p95_offset=offsets[min(n - 1, int(0.95 * n))],
        coverage=matched / max(1, len([w for w in ref.words if _normalize_words([w])])),
    )


@dataclass
class HallucinationMetrics:
    """How much of hyp falls inside reference silence regions."""

    spurious_word_rate: float  # spurious_words / total_hyp_words
    spurious_words: int


def hallucination_metrics(ref: Reference, hyp: Transcript) -> HallucinationMetrics:
    if not hyp.words:
        return HallucinationMetrics(0.0, 0)
    spurious = 0
    for w in hyp.words:
        mid = (w.start + w.end) / 2
        for s, e in ref.silence_intervals:
            if s <= mid <= e:
                spurious += 1
                break
    return HallucinationMetrics(
        spurious_word_rate=spurious / len(hyp.words),
        spurious_words=spurious,
    )


@dataclass
class TrackResult:
    track_id: str
    language: str
    text: TextMetrics
    timing: TimingMetrics
    halluc: HallucinationMetrics

    def as_row(self) -> dict:
        return {
            "track_id": self.track_id,
            "lang": self.language,
            "wer": round(self.text.wer, 4),
            "cer": round(self.text.cer, 4),
            "mean_offset_s": round(self.timing.mean_abs_offset, 3),
            "p95_offset_s": round(self.timing.p95_offset, 3),
            "coverage": round(self.timing.coverage, 3),
            "halluc_rate": round(self.halluc.spurious_word_rate, 4),
            "ref_words": self.text.n_ref_words,
            "hyp_words": self.text.n_hyp_words,
        }


def evaluate(ref: Reference, hyp: Transcript) -> TrackResult:
    return TrackResult(
        track_id=ref.track_id,
        language=ref.language,
        text=text_metrics(ref, hyp),
        timing=timing_metrics(ref, hyp),
        halluc=hallucination_metrics(ref, hyp),
    )
