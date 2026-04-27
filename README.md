# unbake-test

Тестовое для [Unbake AI](https://unbake.net/): API → лирика с word-level таймстемпами на htdemucs v4 vocal, до 8 языков, cost ≤ $0.05/3-min трек.

→ **Design doc: [`docs/design.md`](docs/design.md)**
→ **Сырые цифры бенчмарков: [`docs/bench_log.md`](docs/bench_log.md)**

## TL;DR

Two-pass Shazam-hybrid pipeline:
1. **faster-whisper large-v3** для language detect + ASR fallback.
2. **wav2vec2-CTC + `torchaudio.functional.forced_align`** для выравнивания известной лирики на vocal — главный путь, когда лирика есть.
3. **Cover-detector**: `mean_alignment_confidence > 0.55 AND asr_lyric_overlap > 0.5`. Один сигнал не работает — показано экспериментом (см. §3.2 design doc'а).

Замеры на 1 RU-треке (Pharaoh — Дико, например, htdemucs v4 vocal, 168s):

| | baseline whisper | alignment (XLSR-53-ru) |
|---|---|---|
| RTF на T4 (free Colab) | 0.118 | 0.038 |
| Words placed | 223 | 247 / 248 (coverage 99.6%) |
| WER vs Genius lyrics | 0.347 | 0 by construction |
| Mean confidence | разброс | 0.82 |

Combined RTF на A10G ≈ 0.045 → **$0.00076 за 3-min трек = 65× ниже потолка $0.05**. Cost — не активный constraint, бюджет тратится на accuracy.

## Структура

```
src/eval/        Pydantic-схема, WER/CER, timing-offset, hallucination, set-overlap
src/baseline.py  faster-whisper wrapper с телеметрией
src/align.py     wav2vec2-CTC + torchaudio.forced_align, per-word confidence
notebooks/       Colab T4 — 01_baseline / 02_alignment / 02b_alignment_control
docs/            design doc + bench_log
data/            датасет, лирики, transcripts (gitignored)
```

## Воспроизвести бенчмарки

GPU: бесплатный Colab T4 (или Kaggle T4×2).

```
notebooks/01_baseline.ipynb           # → baseline_pharaoh_ru.json + telemetry
notebooks/02_alignment.ipynb          # → align_pharaoh_ru.json + telemetry
notebooks/02b_alignment_control.ipynb # → align_control.json
```

Все три качают датасет с Yandex Disk автоматически. Результаты складываются в `/content/`, оттуда забираются в `data/transcripts/` локально.

Метрики локально:
```bash
python -m src.eval.text_wer data/lyrics/ru/track.txt data/transcripts/track.json
python -m src.eval.compare_timing data/transcripts/align.json data/transcripts/baseline.json
```

## Что не делается

Платные ASR API, Modal/Replicate, on-device inference, скейл-инфра, iOS-клиент, диаризация, streaming. Подробнее — §5 / §11 design doc'а.
