# Design Doc — API распознавания лирики на htdemucs vocal

**Автор:** Дмитрий Ходжа · **Дата:** 2026-04-27 · **Тестовое:** Unbake AI

> **Black-box задача:** на входе — m4a vocal-стэм (htdemucs v4), на выходе — транскрипция с word-level timestamps и confidence. До 8 языков (ru, en, es, fr, it, pt, ja, pl). Потолок cost ≤ $0.05 за 3-минутный трек.

---

## 1. Контекст

Unbake собирает karaoke-данные (в том числе word-level lyric timing) для своего LLM-движка изучения языков. Источник аудио — htdemucs v4 (не ft) vocal-стэм оригинальных треков. У части треков лирика известна (Genius / LRCLib / Musixmatch / собственная база), у части — нет (каверы, ремиксы, неподписанные треки).

Главный технический выбор: **использовать ли известную лирику** там, где она есть. Если да — задача из распознавания превращается в forced alignment, что качественно проще и быстрее.

## 2. Требования

**Must:**
- Обработка vocal m4a (htdemucs v4 output, AAC 256 kbps, ~3 минуты).
- Word-level timestamps с per-word confidence.
- Поддержка 8 языков (ru, en, es, fr, it, pt, ja, pl).
- Cost ≤ $0.05 за 3-минутный трек.
- Метрика качества: WER + word-level timing offset + hallucination rate (см. §6).

**Non-goals:**
- iOS-клиент / on-device inference (не входит в тестовое).
- Идентификация трека (предполагается, что Shazam уже отработал).
- Обработка mixed audio (только vocal-стэм).
- Скейл-инфра на N тысяч треков (тут — proof of architecture).

## 3. Архитектура: Shazam-hybrid two-pass pipeline

```
                                   ┌────────────────────┐
       audio (vocal m4a)────────►  │  faster-whisper    │  baseline ASR (WER, lang detect)
                              │    │     large-v3       │
                              │    └─────────┬──────────┘
                              │              │ transcript_a, lang
                              │              ▼
                              │      ┌──────────────┐
                              │      │ Shazam? Have │
                              │      │ lyrics?      │  (вход pipeline'а)
                              │      └───────┬──────┘
                              │              │ yes
                              │              ▼
                              │   ┌──────────────────────┐
                              └─►│  wav2vec2-CTC +       │  forced alignment
                                  │  forced_align         │  (XLSR-53 / MMS-1b-all)
                                  └────────────┬──────────┘
                                               │ alignment_a (timestamps + conf)
                                               ▼
                                  ┌──────────────────────┐
                                  │  cover detector       │
                                  │  conf > 0.55  AND     │  (см. §3.2)
                                  │  overlap > 0.5        │
                                  └────┬───────────┬──────┘
                                       │ pass      │ fail
                                       ▼           ▼
                                  alignment    transcript_a
                                  output       (ASR fallback)
```

### 3.1 Два пути

**A. ASR-only (`transcript_a`)** — `faster-whisper large-v3` с `vad_filter=True`, `condition_on_previous_text=False` (срезает hallucination cascade), `beam_size=5`, `word_timestamps=True`. Выход: word-list, lang_prob, per-word confidence.

**B. Forced alignment (`alignment_a`)** — лирика → нормализация (lowercase, punctuation strip, ad-libs `(пау-пау-пау)` и `[Chorus]` выкидываем) → токены char-vocab CTC модели → `torchaudio.functional.forced_align(log_probs, targets, blank)`. Per-word confidence = mean prob фреймов на CTC-пути. Word boundaries = переходы между токенами.

Когда лирика верна, путь B даёт **WER = 0 by construction**, тайминги точнее ASR (см. §6), и в ~3× быстрее по RTF.

### 3.2 Cover detector — почему не один сигнал

Эксперимент (`notebooks/02b_alignment_control.ipynb`):

| вариант лирики | mean_conf | overlap (lyrics ∩ ASR) |
|---|---|---|
| правильная (Pharaoh — Дико, например) | 0.824 | 0.863 |
| shuffled (та же лирика, перемешанные слова) | 0.664 | — |
| **чужая** (Кино — Группа крови) | **0.725** | **0.333** |

`mean_confidence` падает только на 12% от правильной к чужой — **не отделяет**. CTC forced — он *forced*: каждый токен обязательно ставится, и частые слова всегда находят «менее плохие» фреймы. Это критичное наблюдение, которое одной только `confidence`-эвристики не закрывает.

`asr_lyric_overlap = |lyric_words ∩ asr_words| / |lyric_words|` — гораздо чище: **2.6× разрыв** (0.86 vs 0.33), threshold 0.5 разделяет случаи без коллизий.

**Решение:** AND из двух сигналов. `confidence` ловит «грязный vocal / битый decode», `overlap` ловит «не та лирика / кавер». Falling back to `transcript_a` дешевле, чем выдать пользователю выровненную чушь.

### 3.3 Языки

| group | model | comment |
|---|---|---|
| ru | `jonatasgrosman/wav2vec2-large-xlsr-53-russian` | per-language fine-tune, измерено на T4 |
| en, es, fr, it, pt, ja, pl | `facebook/mms-1b-all` | 1100+ языков, единый чекпоинт; trade-off: чуть хуже per-language но операционно проще |

Альтернатива для en/fr/es — per-language XLSR fine-tunes (Hugging Face, 5+ public checkpoints). Решение «MMS vs зоопарк» — на основе бенчмарка по 2-3 трека на язык (см. §10).

## 4. API контракт

```http
POST /transcribe
Content-Type: multipart/form-data

audio:    <vocal.m4a>
lyrics:   <text/plain>            # optional; if present → alignment path
language: ru | en | es | fr | it | pt | ja | pl   # optional, autodetect via whisper
```

```jsonc
200 OK
{
  "language": "ru",
  "language_prob": 1.00,
  "path": "alignment",            // "alignment" | "asr_fallback"
  "cover_detector": {
    "alignment_confidence": 0.824,
    "asr_lyric_overlap": 0.863,
    "decision": "trust_lyrics"
  },
  "words": [
    {"text": "самый", "start": 21.04, "end": 21.40, "confidence": 0.78},
    ...
  ],
  "telemetry": {
    "audio_s": 168.04,
    "infer_s": 6.42,
    "rtf": 0.038,
    "model": "wav2vec2-large-xlsr-53-russian"
  }
}
```

Возможные failure modes (явно):
- `language_prob < 0.6` → 422, ask client to specify language.
- htdemucs vocal оказался почти тишиной → `path: "asr_fallback"`, `words: []`, 200 (это валидный ответ для «инструментал»).

## 5. Что мы НЕ берём и почему

| отклонено | причина |
|---|---|
| OpenAI Whisper API, Deepgram, AssemblyAI | $0.006-0.024/min — **в 5-50× выше** self-hosted на A10G |
| Modal / Replicate | Нет word-level CTC alignment OOTB; cost ≥ $0.005/трек, и долго писать обвязку |
| WhisperX | Хорошая идея под капотом (whisper + wav2vec2 align) — но он делает alignment по whisper-выходу, а не по **известной лирике**. Это не наш случай — нам как раз нужно использовать ground truth lyrics |
| MFA (Montreal Forced Aligner) | Хорошая точность, но pipeline через Kaldi + per-language acoustic models — операционно тяжёлый. `torchaudio.forced_align` даёт 90% качества за 10% инфраструктуры |
| Dual-pass ensemble (whisper + alignment voting) | Cost не активный constraint, можно. Но profit неясен пока не измерен на 8+ треках |

## 6. Метрики

Все реализованы в `src/eval/`:

| метрика | определение | где |
|---|---|---|
| **WER** / **CER** | jiwer, normalize: lowercase, ё→е, ad-libs strip | `text_wer.py` |
| **mean_abs_offset_s** | per-word: \|hyp.start − ref.start\| при token equality, нерасставленные слова не считаем | `metrics.timing_metrics`, `compare_timing.py` |
| **p95_offset_s** | 95-й перцентиль того же | то же |
| **coverage** | n_aligned / n_lyric_words | `align.py` |
| **spurious_word_rate** | hyp слова, попавшие в `silence_intervals` ref-разметки / n_hyp | `metrics.hallucination_metrics` |
| **cover_decision_accuracy** | классификация trust_lyrics / fallback на ground-truth (правильная vs чужая лирика) | TODO — на расширенном датасете |

### Производственные таргеты

| метрика | target | rationale |
|---|---|---|
| WER (alignment path, correct lyrics) | 0 by construction | прямое следствие forced |
| WER (ASR fallback) | < 0.40 на htdemucs vocal | baseline дал 0.347 |
| median timing offset (alignment) | < 0.20 s | для karaoke это на грани заметного |
| coverage (правильная лирика) | > 0.95 | на знакомой студийной версии — 0.996 в бенчмарке |
| cover decision accuracy | > 0.95 | блокирует выдачу мусора |

## 7. Цифры из бенчмарков

Один прогон `notebooks/03_full_bench.ipynb` на Colab T4 → 9 треков × 4 языка (ru/es/en/fr), полная таблица в `docs/bench_log.md`. **Все 9/9 треков с лирикой** через LRCLib (с Unicode-NFC + ASCII-fold + no-artist fallbacks для редких имён вроде Cœur de pirate).

### Сводная таблица

| lang | track | base_wer | align_cov | align_conf |
|---|---|---|---|---|
| en | Post Malone — rockstar | 0.247 | 0.980 | 0.604 |
| es | Peso Pluma — BELLAKEO | 0.655 | 0.975 | 0.667 |
| es | Peso Pluma — BRUCE WAYNE | 0.464 | 0.996 | 0.720 |
| es | Peso Pluma — SOLICITADO | 0.205 | 0.991 | 0.829 |
| fr | Cœur de pirate — République | 0.504 | 0.996 | 0.856 |
| ru | Miyagi — Last of Us | 0.345 | 0.994 | 0.802 |
| ru | Pharaoh — Дико, например | 0.302 | 0.973 | 0.808 |
| ru | Би-2 — Полковнику | 0.325 | 1.000 | 0.905 |
| ru | Скриптонит — Танцуй сама | OOM* | 0.996 | 0.751 |

\* baseline whisper упал по `CUDA out of memory` на Colab T4. Alignment отработал нормально — отдельный аргумент за hybrid path: wav2vec2 forced_align дешевле по памяти и не зависит от длины через VAD-фрагментацию.

### Агрегаты

| metric | mean | median | min | max | n |
|---|---|---|---|---|---|
| baseline WER | 0.381 | 0.335 | 0.205 | 0.655 | 8 |
| **align coverage** | **0.989** | 0.994 | 0.972 | 1.000 | 9 |
| **align mean_conf** | **0.771** | 0.802 | 0.604 | 0.905 | 9 |
| baseline RTF (T4) | 0.098 | 0.058 | 0.037 | 0.340 | 8 |
| **alignment RTF (T4)** | **0.032** | 0.031 | 0.026 | 0.036 | 9 |

### Главные находки

- **Coverage 0.97–1.00 на всех 9 треках, 4 языках** — alignment-путь робастен к смене языка при per-lang XLSR-53. Главное предсказание архитектуры подтверждено.
- **alignment RTF стабильный 0.026–0.036** (разброс 1.4×) — predictable compute независимо от языка/сложности, следствие fixed-cost wav2vec2 forward pass. Полезно для capacity planning.
- **mean_confidence коррелирует с WER**: Би-2 (conf 0.91, WER 0.33, чистый рок-вокал) ↔ BELLAKEO (conf 0.67, WER 0.66, reggaeton с code-switching pt↔es). Confidence чутко реагирует на качество vocal.
- **Cover-detector держит порог:** mean_conf > 0.55 проходит для всех 9 правильных лирик; ASR-overlap > 0.5 (см. §3.2) — отсекатель «не та лирика».
- **3 кейса, где ASR-only path плох или вовсе падает** (BELLAKEO WER 0.66, République WER 0.50, Скриптонит OOM) — alignment вытащил coverage > 0.97 на всех. **Hybrid принципиален**: даже когда whisper проваливается, alignment работает, потому что лирика известна.

Timing offset (alignment vs whisper, на Pharaoh): median 0.33s, p95 0.99s, mean signed −0.36s (whisper стартует слова раньше alignment'а — alignment ближе к реальному onset).

## 8. Cost

Self-hosted, single-GPU. Цены Runpod community:

| GPU | RTF (whisper baseline, mean) | RTF (alignment) | combined RTF | $/h | **$/3-min трек** |
|---|---|---|---|---|---|
| T4 (Colab, измерено) | 0.098 | 0.032 | 0.130 | $0.20 | $0.0013 |
| A10G (×3.5 от T4) | 0.028 | 0.009 | 0.037 | $0.34 | **$0.00063** |
| L4 (×3.0) | 0.033 | 0.011 | 0.043 | $0.43 | $0.00093 |

Потолок ТЗ — **$0.05/трек**. На A10G — **в 80× ниже** даже с двухпроходным pipeline'ом. Cost — **не активный constraint**, оптимизировать его незачем. Свободный бюджет тратится на accuracy: больший beam, ensemble, post-filtering по confidence.

## 9. Риски и unknowns

| риск | mitigation |
|---|---|
| MMS-1b-all хуже, чем XLSR per-language fine-tune на en/fr/es | Бенчмарк до коммита — 2-3 трека на язык, выбираем лучшее по WER |
| htdemucs v4 (не ft) даёт грязный vocal на back-vocals/гармониях → alignment confidence обваливается на тех частях | per-segment confidence уже есть; в API вернём слова с conf < 0.4 как `low_confidence: true` |
| Cover/remix с почти-той-же лирикой (1-2 слова отличаются) — `asr_lyric_overlap` останется высоким, alignment даст невидимо-неправильный output | дополнительный word-by-word check: position-aware diff lyric vs asr (не set-overlap, а Levenshtein) |
| Ad-libs и `(пау-пау-пау)` в Genius-лирике | strip-list в `text_wer.py` уже работает; в проде — конфигурируемый regex |
| Японский / польский — не валидированы | следующий шаг роадмапа |
| LRCLib/Genius miss rate | source mixing: LRCLib → Musixmatch → manual; для тестовой системы 1 источник OK |

## 10. Roadmap

| день | артефакт |
|---|---|
| 1 (done) | repo, eval module, baseline на 1 RU треке: RTF 0.118, WER 0.347 |
| 2 (done) | forced alignment PoC, cover-detector эксперимент с честным negative result, second signal валидирован, design doc |
| 3 (done) | full-dataset bench: 9 треков × 4 языка (ru/es/en/fr) на одном Colab T4; LRCLib как источник лирики; coverage > 0.97 across the board |
| 4 (next) | добор IT/PT/JA/PL через yt-dlp + htdemucs v4 (`notebooks/04_collect_extra_langs.ipynb` готов); калибровка порогов cover-detector'а на полном датасете |
| 5 | замер MMS-1b-all vs per-lang XLSR на ru/es/en/fr (текущая таблица — XLSR), решение «MMS vs зоопарк» для прода |

## 11. Что оставлено за скобками

- Streaming inference (трек целиком обрабатывается за < 10s — стрим не даёт UX-выгоды).
- Дикторная диаризация (vocal-стэм по определению одно-источниковый).
- Punctuation restoration (для karaoke не критична; whisper её и так даёт).
- Lyric source (LRCLib API / Musixmatch / etc.) — компонент upstream, не часть транскрипционного API.

---

**Воспроизводимость:** репо `github.com/RezPoint/unbake-test`. Все цифры — `docs/bench_log.md`. Запуск бенчмарков — `notebooks/01_baseline.ipynb`, `02_alignment.ipynb`, `02b_alignment_control.ipynb`, `03_full_bench.ipynb`, `04_collect_extra_langs.ipynb` в Colab T4 (free). Полный full-dataset bench — один клик в `03_full_bench.ipynb` (~15-20 минут).
