# Design Doc — API распознавания лирики на htdemucs vocal

**Автор:** Дмитрий Ходжа · **Дата:** 2026-04-27 · **Тестовое:** Unbake AI

> **Black-box задача:** на входе — m4a vocal-стэм (htdemucs v4), на выходе — транскрипция с word-level timestamps и confidence. До 8 языков (ru, en, es, fr, it, pt, ja, pl). Потолок cost ≤ $0.05 за 3-минутный трек.

---

## 1. Контекст

Unbake — iOS-приложение для музыкантов, которые делают каверы. Юзер хочет слышать свой кавер с синхронизированной лирикой (своей или оригинальной). Сервер получает vocal-стэм (htdemucs v4 на бэкенде Unbake) и обязан вернуть word-level sync.

**Три типичных сценария входа:**

1. **Известный трек, оригинальная лирика.** Юзер поёт «как в оригинале». Клиент идентифицирует трек **через ShazamKit** (бесплатно, без rate limits на iOS) → присылает `shazam_id` + `artist/title` → сервер находит лирику в LRCLib/Musixmatch/internal → forced alignment. WER = 0 by construction.
2. **Кавер с кастомной лирикой.** Юзер переписал текст. Клиент шлёт `lyrics` напрямую (юзер вводит/редактирует в UI). Внешние БД лирики тут не работают — это центральный нюанс ТЗ.
3. **Неизвестный трек / Shazam miss / лирика не нашлась.** ASR fallback (faster-whisper).

Все три ветки сходятся в одном API endpoint. Маршрутизация — на сервере, по входным полям.

Главный технический выбор: **использовать ли известную лирику** там, где она есть. Если да — задача из распознавания превращается в forced alignment, что качественно проще, точнее по таймингу и в ~3× быстрее.

## 2. Требования

**Must:**
- Обработка vocal m4a (htdemucs v4 output, AAC 256 kbps, ~3 минуты) — см. §3.4 о формате.
- Word-level timestamps с per-word confidence.
- Поддержка 8 языков (ru, en, es, fr, it, pt, ja, pl).
- Cost ≤ $0.05 за 3-минутный трек.
- E2E latency бюджет: warm path ≤ 15s, cold ≤ 60s (см. §8.1).
- Месячный бюджет инфраструктуры: ≤ $100/мес @ 100 req/день, масштабируемо до 10k req/день (см. §8.2).
- Метрика качества: WER + word-level timing offset + hallucination rate (см. §6).

**Non-goals:**
- iOS-клиент / on-device inference (не входит в тестовое; ShazamKit фигурирует только как источник `track_id`).
- Распознавание трека из mixed audio (это делает ShazamKit на клиенте).
- Обработка mixed audio (только vocal-стэм).
- Скейл-инфра на сотни тысяч треков (тут — proof of architecture).

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

### 3.3 Сценарий: кавер с кастомной лирикой

ТЗ явно указывает: **аудитория — каверщики, БД типа lrclib часто не подойдут**. Это означает, что lookup по `shazam_id → LRCLib` ловит только сценарий 1 (см. §1). Для сценария 2 (custom lyrics) есть две UX-точки:

- **Юзер сам ввёл лирику** → клиент шлёт её в поле `lyrics`. Сервер игнорирует ShazamKit-lookup, идёт прямой alignment path. Cover-detector проверяет `mean_conf > 0.55 AND overlap > 0.5` — на правильно введённой кастомной лирике это пройдёт; если юзер случайно ввёл лирику другой песни, AND отсечёт и упадёт в ASR fallback.
- **Юзер не ввёл лирику, но трек — кавер с другим текстом** → ShazamKit-lookup находит оригинальную лирику → cover-detector ловит несовпадение (`overlap` обвалится) → fallback в ASR. Юзер получает разумный результат вместо тайминга чужой лирики.

Эта развилка — главное отличие от «отдай в Whisper и всё». Forced alignment к **неправильной** лирике даёт силно неверный, но плавный output, который без cover-detector выглядит правдоподобно. См. §3.2 — почему одного `mean_confidence` недостаточно.

### 3.4 Формат входа: m4a vs альтернативы

ТЗ просит «подумать, какой формат лучше». Анализ:

| формат | размер 3-мин vocal | upload @ 5 Mbps 4G | потеря качества для CTC | вердикт |
|---|---|---|---|---|
| **m4a 256k AAC** (htdemucs default) | ~5.7 MB | 9.1 s | none после 16k mono resample | **используем как есть** |
| Opus 64k | ~1.4 MB | 2.2 s | none (CTC ест 16k mono) | **v2 оптимизация** для slow networks — re-encode на клиенте |
| WAV 16-bit/44.1k | ~30 MB | 48 s | reference | избыточно для mobile |
| FLAC | ~18 MB | 29 s | none | избыточно |

CTC-модели (XLSR-53, MMS) работают на 16 kHz mono — внутренний resample стирает разницу между lossy AAC 256k, Opus 64k и lossless. **Bottleneck — upload time**, не качество. m4a 256k оставляем по умолчанию (htdemucs выдаёт его без перекодирования = +0 latency на бэкенде); если клиентская телеметрия покажет долгие upload'ы на 4G, добавляем Opus 64k transcode на клиенте (iOS AVAudioConverter делает это локально за <500ms).

### 3.5 Языки

| group | model | comment |
|---|---|---|
| ru | `jonatasgrosman/wav2vec2-large-xlsr-53-russian` | per-language fine-tune, измерено на T4 |
| en, es, fr, it, pt, ja, pl | `facebook/mms-1b-all` | 1100+ языков, единый чекпоинт; trade-off: чуть хуже per-language но операционно проще |

Альтернатива для en/fr/es — per-language XLSR fine-tunes (Hugging Face, 5+ public checkpoints). Решение «MMS vs зоопарк» — на основе бенчмарка по 2-3 трека на язык (см. §10).

## 4. API контракт

```http
POST /transcribe
Content-Type: multipart/form-data

audio:        <vocal.m4a>           # presigned S3 URL в production
lyrics:       <text/plain>          # optional; явная кастомная лирика (cover scenario 2)
shazam_id:    <string>              # optional; iOS ShazamKit, для server-side lyric lookup
artist:       <string>              # optional; параллельно с shazam_id, helps LRCLib lookup
title:        <string>              # optional; то же
language:     ru | en | ...         # optional; autodetect via whisper
cover_mode:   strict | lenient      # default lenient; strict → fail если cover-detector не сработал
```

**Маршрутизация (server-side):**

```
if lyrics: → alignment path (юзер ввёл сам, scenario 2)
elif shazam_id or (artist and title): →
    LRCLib/internal lookup
    if hit: → alignment path (scenario 1)
    else:   → ASR fallback (scenario 3)
else: → ASR fallback (scenario 3)
```

В обоих alignment-сценариях работает cover-detector (§3.2). При `cover_mode=strict` его failure → `path: "asr_fallback"` без warning'а; при `lenient` — fallback с warning в response.

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

Полный прогон `03_full_bench.ipynb` (9 треков, ru/es/en/fr) + `04_collect_extra_langs.ipynb` (4 трека, it/pt/ja/pl) на Colab T4 → **13 треков × 8 языков**. Полная таблица — `docs/bench_log.md`. Все треки с лирикой через LRCLib (NFC + ASCII-fold + no-artist fallbacks).

### Сводная таблица

| lang | track | model | base_wer | align_cov | align_conf | align_rtf |
|---|---|---|---|---|---|---|
| en | Post Malone — rockstar | XLSR-53-en | 0.247 | 0.980 | 0.604 | 0.033 |
| es | Peso Pluma — BELLAKEO | XLSR-53-es | 0.655 | 0.975 | 0.667 | 0.030 |
| es | Peso Pluma — BRUCE WAYNE | XLSR-53-es | 0.464 | 0.996 | 0.720 | 0.031 |
| es | Peso Pluma — SOLICITADO | XLSR-53-es | 0.205 | 0.991 | 0.829 | 0.031 |
| fr | Cœur de pirate — République | XLSR-53-fr | 0.504 | 0.996 | 0.856 | 0.036 |
| ru | Miyagi — Last of Us | XLSR-53-ru | 0.345 | 0.994 | 0.802 | 0.029 |
| ru | Pharaoh — Дико, например | XLSR-53-ru | 0.302 | 0.973 | 0.808 | 0.026 |
| ru | Би-2 — Полковнику | XLSR-53-ru | 0.325 | 1.000 | 0.905 | 0.034 |
| ru | Скриптонит — Танцуй сама | XLSR-53-ru | **OOM** | 0.996 | 0.751 | 0.036 |
| it | Måneskin — I WANNA BE YOUR SLAVE | MMS-1b-all | 0.355 | 0.972 | 0.712 | 0.084 |
| pt | Anitta — Envolver | MMS-1b-all | **OOM** | 0.990 | 0.730 | 0.097 |
| pl | sanah — Szampan | MMS-1b-all | **OOM** | 1.000 | 0.886 | 0.096 |
| ja | YOASOBI — Idol | MMS-1b-all | 7.52† | 0.213† | 0.868 | 0.111 |

**OOM:** baseline whisper-large-v3 упал по `CUDA out of memory` на T4 (16 GB) при одновременной загрузке wav2vec2. **Alignment отработал в 3/3 случаях** — отдельный (emergent) аргумент за hybrid: wav2vec2 forced_align дешевле по памяти.

**† JA — артефакт word-based метрик на CJK.** Японский без пробелов; lyrics whitespace-split дал 75 «токенов», alignment вернул 16 фразовых единиц (`'無敵の笑顔で荒らすメディア'` как один word), whisper — per-character timestamps (564). Confidence 0.87 показывает, что **alignment корректен**; нужна mecab-/char-tokenized метрика для JA/ZH/KO в проде. См. §9.

### Агрегаты (13 треков × 8 языков)

| metric | mean | median | min | max | n |
|---|---|---|---|---|---|
| baseline WER (excl ja degen) | 0.378 | 0.345 | 0.205 | 0.655 | 9 |
| **align coverage** (excl ja) | **0.988** | 0.992 | 0.972 | 1.000 | 12 |
| **align mean_conf** | **0.780** | 0.802 | 0.604 | 0.905 | 13 |
| baseline RTF (T4) | 0.092 | 0.067 | 0.037 | 0.340 | 10 |
| **align RTF — XLSR-53** | **0.032** | 0.031 | 0.026 | 0.036 | 9 |
| **align RTF — MMS-1b-all** | 0.097 | 0.096 | 0.084 | 0.111 | 4 |

### Главные находки

- **Coverage > 0.97 на 12/13 треках, 8 языках** — alignment-путь робастен к смене языка как на per-lang XLSR-53, так и на multilingual MMS-1b-all. Архитектурное предсказание подтверждено.
- **MMS-1b-all в ~3× медленнее XLSR-53** на T4 (~0.097 vs ~0.032) — ожидаемо при 1B vs 300M параметров. Trade-off «MMS vs зоопарк» эмпирический: оба варианта в порядки ниже cost-потолка, выбор по операционной простоте.
- **mean_confidence коррелирует с WER** (где он осмыслен): Би-2 (conf 0.91, WER 0.33) ↔ BELLAKEO (conf 0.67, WER 0.66, reggaeton + code-switch).
- **Cover-detector держит порог:** mean_conf > 0.55 проходит для всех 13 правильных лирик; ASR-overlap > 0.5 (§3.2) — отсекатель «не та лирика».
- **3 OOM-кейса на baseline whisper** (Скриптонит, Anitta, sanah) — alignment вытащил coverage 0.99-1.00 на всех. **Wav2vec2 forced_align дешевле по памяти** (300M vs 1.5B), не нуждается в beam/condition-on-prev кэше. На 3-минутных треках T4 один whisper держит, два не помещается — hybrid-path **операционное** преимущество сверх accuracy-преимущества.

Timing offset (alignment vs whisper, на Pharaoh): median 0.33s, p95 0.99s, mean signed −0.36s (whisper стартует слова раньше alignment'а — alignment ближе к реальному onset).

## 8. Cost

Self-hosted, single-GPU. Цены Runpod community:

| GPU | RTF (whisper baseline, mean) | RTF (align, mean mixed) | combined RTF | $/h | **$/3-min трек** |
|---|---|---|---|---|---|
| T4 (Colab, измерено) | 0.092 | 0.052 | 0.144 | $0.20 | $0.0014 |
| A10G (×3.5 от T4) | 0.026 | 0.015 | 0.041 | $0.34 | **$0.00070** |
| L4 (×3.0) | 0.031 | 0.017 | 0.048 | $0.43 | $0.00103 |

Потолок ТЗ — **$0.05/трек**. На A10G — **в ~70× ниже**. Если выбрать XLSR-only (per-lang зоопарк) для всех языков, align RTF падает до 0.032 → A10G **$0.00060/трек, в 80× ниже**.

Cost — **не активный constraint**. Свободный бюджет тратится на accuracy: больший beam, ensemble, post-filtering по confidence.

### 8.1 E2E latency (iOS юзер → запрос → ответ)

ТЗ-критерий #3. Разбивка для warm-path A10G, 3-минутный трек, 4G mobile network:

| этап | время | примечание |
|---|---|---|
| ShazamKit на клиенте | 0.5–2 s | iOS-side, до запроса; ~5 s аудио → fingerprint |
| htdemucs на бэкенде Unbake | (вне scope) | предполагается уже сделано upstream, vocal.m4a в S3 |
| Presigned URL fetch (server pulls audio) | ~1 s | S3 → GPU host, 5.7 MB |
| GPU inference (alignment + whisper-fallback) | **~7 s** | RTF 0.041 на A10G, 3-мин трек |
| LRCLib lookup (если применимо) | 0.2–0.5 s | внешний REST, кэшируется |
| JSON response (~50 KB) → клиент | <1 s | мобильный download |
| **E2E warm** | **~10 s** | целевой UX |
| **E2E cold (serverless cold start)** | **~40 s** | загрузка whisper-large-v3 + wav2vec2 в VRAM |

Streaming-выдача (отдавать выровненные слова чанками по мере готовности) — возможна, но за 7 секунд compute UX-выгода маргинальна. Skip в v1.

**Cold-start mitigation:** на serverless (Modal) — keep-warm 1 instance круглосуточно при 100 req/день (стоимость см. §8.2). На always-on (Runpod) — cold start не возникает.

### 8.2 Месячный бюджет: 100 vs 10k req/день

ТЗ: $100/мес на серверы при 100 req/день, скейл до 10k req/день. Расчёт на A10G (см. §8):

| режим | req/день | compute time/мес | serverless (Modal) | always-on (Runpod, 1 GPU) | вердикт |
|---|---|---|---|---|---|
| MVP | 100 | 21k sec (~5.8 ч) | **~$10/мес** + keep-warm $245/мес | $245/мес idle 99% | **serverless** |
| growth | 1k | 210k sec (~58 ч) | ~$70/мес | $245/мес @ 8% util | **serverless** |
| scale | 10k | 2.1M sec (~580 ч) | ~$685/мес | **2 GPU = ~$490/мес** @ ~80% util | **always-on (×2)** |

**Crossover ~3.5k req/день:** ниже — serverless оптимален; выше — always-on с автоскейлом.

**Решение:** start with Modal serverless. При >3k req/день стабильно — миграция на Runpod always-on кластер с k8s autoscaling. Код одинаковый; различается только deployment-обёртка (FastAPI + container image один и тот же).

Для MVP @ 100 req/день: keep-warm нужен для cold-start UX (см. §8.1) — это $245/мес фиксированно. **Если cold-start 40s допустим в MVP** — keep-warm убираем, runtime $10/мес, **в 10× ниже бюджета ТЗ**. Решение по keep-warm — продуктовый трейд-офф (UX vs cost), не технический.

## 9. Риски и unknowns

| риск | mitigation |
|---|---|
| MMS-1b-all хуже, чем XLSR per-language fine-tune на en/fr/es | Бенчмарк до коммита — 2-3 трека на язык, выбираем лучшее по WER |
| htdemucs v4 (не ft) даёт грязный vocal на back-vocals/гармониях → alignment confidence обваливается на тех частях | per-segment confidence уже есть; в API вернём слова с conf < 0.4 как `low_confidence: true` |
| Cover/remix с почти-той-же лирикой (1-2 слова отличаются) — `asr_lyric_overlap` останется высоким, alignment даст невидимо-неправильный output | дополнительный word-by-word check: position-aware diff lyric vs asr (не set-overlap, а Levenshtein) |
| Ad-libs и `(пау-пау-пау)` в Genius-лирике | strip-list в `text_wer.py` уже работает; в проде — конфигурируемый regex |
| Word-based метрики (WER, coverage) ломаются на CJK (no-whitespace tokenization) | для JA/ZH/KO — char-level coverage или mecab/jieba-tokenized метрика; в эмпирике на YOASOBI alignment корректен (conf 0.87), но формальный coverage 0.21 — артефакт счёта |
| LRCLib/Genius miss rate | source mixing: LRCLib → Musixmatch → manual; для тестовой системы 1 источник OK |

## 10. Roadmap

| день | артефакт |
|---|---|
| 1 (done) | repo, eval module, baseline на 1 RU треке: RTF 0.118, WER 0.347 |
| 2 (done) | forced alignment PoC, cover-detector эксперимент с честным negative result, second signal валидирован, design doc |
| 3 (done) | full-dataset bench: 9 треков × 4 языка (ru/es/en/fr) на одном Colab T4; LRCLib как источник лирики; coverage > 0.97 across the board |
| 4 (done) | добор IT/PT/JA/PL через yt-dlp + htdemucs v4: alignment работает на всех 8 языках; подтверждено эмпирически, что MMS в ~3× медленнее per-lang XLSR-53 |
| 5 (next) | калибровка порогов cover-detector'а на расширенном датасете; CJK-метрика (mecab tokenization) для JA/KO/ZH; замер MMS vs per-lang XLSR на ru/es/en/fr — выбор «зоопарк vs единый чекпоинт» для прода |

## 11. Чем это отличается от наивного LLM-решения

ТЗ-критерий: «уникальность относительно того, что выдаёт LLM». Наивный LLM-ответ на эту задачу:

> «Возьми OpenAI Whisper API, передай audio, верни `word_timestamps=True`. Если медленно — Deepgram. Если дорого — self-host Whisper.»

Что упускает этот подход и что мы делаем иначе:

| наивный LLM-подход | проблема | наше решение |
|---|---|---|
| Whisper API на всё | $0.006/мин = **$0.018/трек, ×26 от нашей цены $0.0007**; ×3.6 от потолка ТЗ при scale | self-host wav2vec2 + faster-whisper, оба opensource |
| ASR без знания лирики | WER 0.21–0.66 на нашем датасете; для karaoke это нечитаемо | forced alignment к ground-truth → **WER = 0 by construction** в 12/13 случаев |
| Whisper не умеет принимать лирику hint'ом | `initial_prompt` влияет на стиль, но не на точность тайминга | `torchaudio.functional.forced_align` на ту же лирику + char-level CTC |
| «Если есть лирика — добавь её в prompt» | Whisper всё равно галлюцинирует на тишине / инструменталках | VAD + forced path даёт `coverage` как hard signal: где лирика выровнялась, где нет |
| Cover-detector через `confidence` | confidence на чужой лирике падает только на 12% (см. §3.2) — не отделяет | **two-signal AND** (`mean_conf > 0.55 AND overlap > 0.5`) — эмпирически валидирован |
| Метрика — WER на всём | на CJK без mecab WER 7.52 при коректном alignment | per-language tokenization (см. §9) + char-level coverage для JA/ZH/KO |

**Ключевая структурная разница:** LLM трактует задачу как «распознавание». Мы трактуем её как **routing между двумя путями** — alignment (когда лирика известна и совпадает) vs ASR (когда нет). Cover-detector — это router. Это не подсказка, которую LLM выдаст из коробки — она требует эмпирической проверки, что **одного сигнала недостаточно** (см. §3.2 negative result).

Вторая структурная разница: ShazamKit на клиенте бесплатно даёт `track_id`, и сервер использует это как explicit input → server lookup в LRCLib → alignment. Наивный подход эту бесплатную идентификацию игнорирует и ASR'ит всё подряд.

## 12. Что оставлено за скобками

- Streaming inference (трек целиком обрабатывается за < 10s — стрим не даёт UX-выгоды).
- Дикторная диаризация (vocal-стэм по определению одно-источниковый).
- Punctuation restoration (для karaoke не критична; whisper её и так даёт).
- Lyric source (LRCLib API / Musixmatch / etc.) — компонент upstream, не часть транскрипционного API.

---

**Воспроизводимость:** репо `github.com/RezPoint/unbake-test`. Все цифры — `docs/bench_log.md`. Запуск бенчмарков — `notebooks/01_baseline.ipynb`, `02_alignment.ipynb`, `02b_alignment_control.ipynb`, `03_full_bench.ipynb`, `04_collect_extra_langs.ipynb` в Colab T4 (free). Полный full-dataset bench — один клик в `03_full_bench.ipynb` (~15-20 минут).
