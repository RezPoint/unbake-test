from __future__ import annotations

from pydantic import BaseModel, Field


class Word(BaseModel):
    text: str
    start: float = Field(ge=0, description="seconds")
    end: float = Field(ge=0, description="seconds")
    confidence: float | None = None


class Transcript(BaseModel):
    """Output of an ASR/alignment system. Word-level granularity required."""

    language: str
    words: list[Word]


class Reference(BaseModel):
    """Ground truth for a single track."""

    track_id: str
    language: str
    words: list[Word]
    silence_intervals: list[tuple[float, float]] = Field(
        default_factory=list,
        description="Regions where no vocal is present — used to score hallucinations.",
    )
