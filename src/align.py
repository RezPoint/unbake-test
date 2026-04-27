"""Forced alignment of known lyrics on htdemucs vocal.

Pipeline:
  audio (m4a) → 16 kHz mono → wav2vec2-CTC emissions → forced_align
  lyrics text → normalize → tokenize via the model's char vocab → token ids
  → torchaudio.functional.forced_align → frame indices → word spans

Output: a Transcript whose `words` carry the *known* lyric tokens
with timestamps and per-word confidence (mean of frame probs along
the aligned path). WER vs lyrics is 0 by construction; the useful
metrics are coverage (fraction of lyric words placed) and mean
confidence — low confidence flags artefacts / cover lyrics.

Default model: jonatasgrosman/wav2vec2-large-xlsr-53-russian.
For other languages swap to e.g. facebook/mms-1b-all.
"""

from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import torch
import torchaudio.functional as F
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from .eval.schema import Transcript, Word

SR = 16_000


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower().replace("ё", "е")
    text = re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", text)
    text = re.sub(r"[^\w\s'-]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _to_token_ids(words: list[str], processor: Wav2Vec2Processor) -> tuple[list[int], list[tuple[int, int]]]:
    """Return flat list of token ids and (start, end) span per word in that list."""
    vocab = processor.tokenizer.get_vocab()
    word_delim = processor.tokenizer.word_delimiter_token  # usually "|"
    delim_id = vocab[word_delim]
    unk_id = vocab[processor.tokenizer.unk_token]
    ids: list[int] = []
    spans: list[tuple[int, int]] = []
    for i, w in enumerate(words):
        if i > 0:
            ids.append(delim_id)
        start = len(ids)
        for ch in w:
            ids.append(vocab.get(ch, unk_id))
        spans.append((start, len(ids)))
    return ids, spans


@dataclass
class AlignTelemetry:
    audio_s: float
    infer_s: float
    rtf: float
    n_lyric_words: int
    n_aligned: int
    coverage: float
    mean_confidence: float
    model: str


def align(
    audio_path: str | Path,
    lyrics_text: str,
    language: str = "ru",
    model_name: str = "jonatasgrosman/wav2vec2-large-xlsr-53-russian",
    device: str = "auto",
) -> tuple[Transcript, AlignTelemetry]:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    audio, _ = librosa.load(str(audio_path), sr=SR, mono=True)
    audio_s = len(audio) / SR

    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device).eval()

    inputs = processor(audio, sampling_rate=SR, return_tensors="pt").input_values.to(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        logits = model(inputs).logits
        log_probs = torch.log_softmax(logits, dim=-1)  # (1, T, V)
    infer_s = time.perf_counter() - t0

    words = _normalize(lyrics_text).split()
    token_ids, word_spans = _to_token_ids(words, processor)
    targets = torch.tensor([token_ids], dtype=torch.int32, device=device)

    blank_id = model.config.pad_token_id or 0
    aligned, scores = F.forced_align(log_probs, targets, blank=blank_id)
    aligned = aligned[0].cpu().numpy()  # (T,)
    scores = scores[0].cpu().exp().numpy()  # frame-level prob along path

    # Map frame index → seconds. wav2vec2 stride ≈ 320 samples = 20 ms.
    n_frames = log_probs.shape[1]
    frame_s = audio_s / n_frames

    # For each token id index k in flat token list, find frame range where
    # forced_align placed it. Build map: frame_t → token_index_in_targets.
    # forced_align returns the path over the target sequence with blanks;
    # we recover token positions by scanning where the label changes.
    token_frames: dict[int, tuple[int, int, list[float]]] = {}
    last_tok = -1
    cur_start = 0
    cur_probs: list[float] = []
    target_pos = -1
    for t, lab in enumerate(aligned):
        if lab == blank_id:
            cur_probs.append(float(scores[t]))
            continue
        if lab != last_tok:
            if last_tok != -1:
                token_frames[target_pos] = (cur_start, t, cur_probs)
            target_pos += 1
            cur_start = t
            cur_probs = [float(scores[t])]
            last_tok = lab
        else:
            cur_probs.append(float(scores[t]))
    if last_tok != -1:
        token_frames[target_pos] = (cur_start, len(aligned), cur_probs)

    out_words: list[Word] = []
    aligned_count = 0
    confidences: list[float] = []
    for w_idx, (s, e) in enumerate(word_spans):
        char_frames = [token_frames[k] for k in range(s, e) if k in token_frames]
        if not char_frames:
            continue
        start_frame = char_frames[0][0]
        end_frame = char_frames[-1][1]
        probs = [p for cf in char_frames for p in cf[2]]
        conf = float(np.mean(probs)) if probs else 0.0
        out_words.append(
            Word(
                text=words[w_idx],
                start=round(start_frame * frame_s, 3),
                end=round(end_frame * frame_s, 3),
                confidence=round(conf, 4),
            )
        )
        aligned_count += 1
        confidences.append(conf)

    transcript = Transcript(language=language, words=out_words)
    tele = AlignTelemetry(
        audio_s=round(audio_s, 2),
        infer_s=round(infer_s, 2),
        rtf=round(infer_s / audio_s, 4),
        n_lyric_words=len(words),
        n_aligned=aligned_count,
        coverage=round(aligned_count / len(words), 4) if words else 0.0,
        mean_confidence=round(float(np.mean(confidences)), 4) if confidences else 0.0,
        model=model_name,
    )
    return transcript, tele
