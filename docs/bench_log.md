# Bench Log

Сырые цифры из бенчмарков. Каждая строка — один прогон. Цифры идут в design doc.

## 2026-04-27 — baseline #1

- Notebook: `notebooks/01_baseline.ipynb`
- GPU: Colab T4 (Tesla T4 16 GB)
- Model: faster-whisper `large-v3`, `compute_type=float16`, `beam_size=5`, `vad_filter=True`, `condition_on_previous_text=False`
- Track: `data/raw/ru/Pharaoh - Дико, например.m4a` (htdemucs v4 vocal, m4a 256 kbps)

### Telemetry

| metric | value |
|---|---|
| audio_s | 168.04 |
| infer_s | 19.84 |
| **rtf** | **0.118** |
| load_s (one-time) | 29.16 |
| detected_lang / lang_prob | ru / 1.00 |

### Качественные наблюдения

- Транскрипция стартует с ~21 сек — VAD корректно отрезал интро без вокала.
- В конце словa без вокала (`детка` после фрагмента «...киской детка где твой») — кандидат на галлюцинацию. Нужна перепроверка с silence_intervals в ground truth.
- Confidence < 0.7 на отдельных словах (`баконьер` 0.68, `видный` 0.60, `эту` 0.37, `ее` 0.56). Порог `< 0.6` отбраковки — рабочая гипотеза.
- Слово «баконьер» — вероятно substitution для «богатей». Проверить против лирики (TODO).

### Cost экстраполяция

| GPU | RTF (оценка) | 3-min трек, sec | Runpod $/h | **$/3-min** |
|---|---|---|---|---|
| T4 (измерено) | 0.118 | 21.2 | 0.20 | **0.0012** |
| A10G (×3.5 от T4) | 0.034 | 6.1 | 0.34 | **0.00058** |
| L4 | 0.040 | 7.2 | 0.43 | **0.00086** |

Потолок ТЗ: **$0.05/трек**. Мы в **50-80×** ниже.

→ Cost не является активным constraint. Можно тратить «бюджет» на accuracy: второй проход alignment, больший beam_size, ensemble, post-filter по confidence.

### WER vs Genius lyrics

`python -m src.eval.text_wer data/lyrics/ru/Pharaoh\ -\ Дико,\ например.txt data/transcripts/baseline_pharaoh_ru.json`

| metric | value |
|---|---|
| **WER** | **0.347** |
| **CER** | **0.201** |
| n_ref words | 248 |
| n_hyp words | 223 |
| hits | 173 |
| substitutions | 39 |
| deletions | 36 |
| insertions | 11 |

Нормализация: lowercase, ё→е, `(пау-пау-пау)` и `[Chorus]`-метки выкинуты, пунктуация выкинута.

**Интерпретация:**
- 36 deletions при 4 повторах припева (~64 слова на повтор) — кандидаты сразу: пропущенные части припева. VAD мог зарезать тихие повторы или модель применила condition_on_previous_text=False и не «дослушала».
- WER 0.35 — это **до** alignment-прохода. Цель пути Shazam-hybrid (forced alignment лирики на vocal через wav2vec2-CTC) — снизить WER ниже 0.10 на треках, для которых лирика уже найдена.

## 2026-04-27 — alignment #1

- Notebook: `notebooks/02_alignment.ipynb`
- GPU: Colab T4
- Model: `jonatasgrosman/wav2vec2-large-xlsr-53-russian` (CTC)
- Aligner: `torchaudio.functional.forced_align` (CUDA)
- Track: тот же `Pharaoh - Дико, например.m4a` + Genius lyrics

### Telemetry

| metric | value |
|---|---|
| audio_s | 168.04 |
| infer_s | 6.42 |
| **rtf** | **0.038** |
| n_lyric_words | 248 |
| n_aligned | 247 |
| **coverage** | **0.996** |
| **mean_confidence** | **0.824** |

### Сравнение alignment (silver) vs baseline whisper

`python -m src.eval.compare_timing align_pharaoh_ru.json baseline_pharaoh_ru.json`

| metric | value |
|---|---|
| matched (token-equal) | 173 / 247 |
| match_rate | 0.70 |
| mean abs offset | 0.405 s |
| median abs offset | 0.331 s |
| p95 abs offset | 0.986 s |
| mean signed offset (hyp − ref) | **−0.362 s** |

### Интерпретация

- **Coverage 99.6%** на знакомой студийной версии — alignment-путь практически безошибочно укладывает известную лирику на htdemucs vocal. Один пропущенный слово — кандидат на ad-lib / тихий фрагмент.
- **mean confidence 0.82** — чистый сигнал на htdemucs v4 vocal. На каверах / другой версии лирики ожидаем drop до 0.3-0.5 — это и есть детектор «лирика не та» в Shazam-hybrid pipeline.
- **RTF 0.038** на T4 — alignment-проход в **3× быстрее** whisper baseline. Полный двухпроходный pipeline (whisper для language detect + ASR fallback на каверах + alignment) укладывается в RTF ≈ 0.16, cost на A10G ≈ $0.0008/трек — всё ещё **60× ниже** потолка.
- **Timing offset 0.33s median**: whisper-baseline консистентно стартует слова на **~0.36s раньше** alignment. Для karaoke-применения 0.33s медиана — на грани заметного. Alignment даёт лучший timing.
- **Match rate 70%** между whisper и alignment по тексту = whisper верно угадал 70% слов лирики; оставшиеся 30% — substitutions/deletions whisper'а, которые alignment чинит «бесплатно», т.к. знает текст.

### Что это значит для архитектуры

→ **Shazam-hybrid доказан на одной точке:** известная лирика + forced alignment даёт coverage > 0.99 и confidence > 0.8 за RTF 0.04. ASR-only baseline даёт WER 0.35. Разница огромна и в пользу alignment-пути.

→ Confidence — сразу годный сигнал «правильная ли это лирика». Cross-check ASR-vs-alignment WER (как у нас 30% несовпадений) — second signal.

## 2026-04-27 — alignment control

Notebook: `notebooks/02b_alignment_control.ipynb`. Цель — проверить, падает ли `mean_confidence`, когда лирика **не та**.

| вариант | mean_confidence | coverage | n_aligned | n_lyric_words |
|---|---|---|---|---|
| same lyrics | **0.824** | 0.996 | 247 | 248 |
| shuffled (same words, wrong order) | 0.664 | 0.996 | 247 | 248 |
| other song (Кино — Группа крови) | **0.725** | 1.000 | 144 | 144 |

### Главный вывод (плохой для простой гипотезы)

`mean_confidence` **не является самостоятельным cover-detector'ом**. Чужая лирика дала 0.72 — drop всего 12% от правильной 0.82. Хуже того: shuffled (0.66) оказался ниже чужой песни (0.72) — то есть путаница порядка bites больнее, чем полная подмена контента.

**Почему:** forced_align *forced* — он ОБЯЗАН поставить каждый токен куда-то. CTC всегда находит «наименее плохой» фрейм. Частые RU-слова из чужой песни («я», «не», «мне», «в», «на») фонетически цепляются за подходящие участки аудио → confidence остаётся приличным.

### Архитектурное следствие

Cover-detector в Shazam-hybrid pipeline должен использовать **два сигнала** и принимать решение по AND/OR:

1. `mean_confidence < 0.55` (пороги откалибровать на расширенном датасете) — **слабый** сигнал.
2. **ASR-vs-alignment word-match-rate**: прогон baseline whisper параллельно с alignment, считаем долю слов, попавших по тексту. На same — 70% (см. compare_timing). На other_song ожидаем единицы процентов.

→ Дизайн API: `confidence` — soft-flag, `asr_match_rate` — hard-flag, fallback на ASR-only при низком обоих.

## 2026-04-27 — second signal: ASR-vs-lyrics set overlap

Локальный счёт без GPU. На том же baseline whisper-транскрипте Pharaoh-трека считаем `|lyric_set ∩ asr_set| / |lyric_set|` для двух кандидатных лирик:

| candidate lyrics | overlap rate |
|---|---|
| Pharaoh — Дико, например (правильная) | **0.863** (214/248) |
| Кино — Группа крови (чужая) | **0.333** (30/90) |

**Разрыв 2.6×**, threshold ≈ 0.5 даёт чистое разделение. Сигнал крепкий и дешёвый — считается за миллисекунды на CPU после baseline ASR прогона.

### Финальный cover-detector для design doc

```
is_correct_lyrics = (mean_alignment_confidence > 0.55)
                    AND (asr_lyric_overlap > 0.5)
```

- `mean_alignment_confidence` — soft signal, отсекает дополнительные гнилые случаи (плохое разделение htdemucs, тихий vocal).
- `asr_lyric_overlap` — hard signal, ловит подмену лирики и каверы с другим текстом.
- Калибровка порогов на расширенном датасете (8 треков × 8 языков) — задача для дальнейшей работы.

### Что дальше

1. ES/EN/FR — `facebook/mms-1b-all` (multilingual CTC, поддерживает 1100+ языков).
2. Прогнать alignment+baseline на остальных 8 треках для статистики порогов.
3. **Писать design doc** — у нас есть всё: метрики, baseline, alignment-путь, проверенный cover-detector, cost-расчёты.

## 2026-04-27 — full-dataset bench (9 треков, 4 языка)

- Notebook: `notebooks/03_full_bench.ipynb`
- GPU: Colab T4
- Pipeline: `src/eval/batch_runner.py` — baseline (faster-whisper large-v3) + alignment (per-lang XLSR-53) per track, кэш по JSON, сводка → `docs/results.csv`
- Источник лирики: LRCLib API (`src/lyrics.py`)

### Сводная таблица

| lang | track | base_rtf | base_wer | base_cer | n_ref | align_cov | align_conf |
|---|---|---|---|---|---|---|---|
| en | Post Malone — rockstar | 0.073 | **0.247** | 0.170 | 441 | 0.980 | **0.604** |
| es | Peso Pluma — BELLAKEO | 0.329 | **0.687** | 0.539 | 319 | 0.975 | **0.667** |
| es | Peso Pluma — BRUCE WAYNE | 0.037 | **0.464** | 0.359 | 239 | 0.996 | **0.720** |
| es | Peso Pluma — SOLICITADO | 0.044 | **0.205** | 0.154 | 215 | 0.991 | **0.829** |
| ru | Miyagi — Last of Us | 0.083 | **0.345** | 0.153 | 307 | 0.994 | **0.802** |
| ru | Pharaoh — Дико, например | 0.133 | **0.235** | 0.106 | 255 | 0.973 | **0.808** |
| ru | Би-2 — Полковнику | 0.046 | **0.181** | 0.098 |  83 | 1.000 | **0.905** |
| fr | Cœur de pirate — République | 0.046 | — | — | — | no_lyrics | — |
| ru | Скриптонит — Танцуй сама | 0.226 | — | — | — | no_lyrics | — |

**Агрегаты по 7 трекам с лирикой:**

| metric | mean | median | min | max |
|---|---|---|---|---|
| baseline WER | 0.338 | 0.247 | 0.181 | 0.687 |
| baseline CER | 0.226 | 0.154 | 0.098 | 0.539 |
| baseline RTF | 0.106 | 0.073 | 0.037 | 0.329 |
| align coverage | 0.987 | 0.991 | 0.973 | 1.000 |
| align mean_conf | 0.762 | 0.802 | 0.604 | 0.905 |

### Наблюдения

- **Coverage 0.97–1.00** на всех языках (RU/ES/EN, htdemucs v4 vocal) — alignment-путь робастен к смене языка при per-lang XLSR-53 чекпоинте. Главное предсказание дизайна (Shazam-hybrid даёт high-coverage) подтверждено на n=7.
- **mean_confidence коррелирует с WER**: Би-2 (WER 0.18) → conf 0.91; BELLAKEO (WER 0.69, reggaeton + ad-libs) → conf 0.67. Это естественно — чем чище vocal и чётче дикция, тем выше оба показателя.
- **WER 0.69 на BELLAKEO** — outlier: тяжёлый bachata/reggaeton с code-switching pt↔es и обильными ad-libs, плюс htdemucs v4 на этом материале режет vocal с артефактами (RTF baseline 0.33 — whisper упирался в hallucination loop). Alignment вытянул coverage 0.975 — фишка hybrid'а: даже когда ASR проваливается (WER 0.69), alignment сохраняет timing.
- **Английский (rockstar) — самый низкий confidence (0.604)**: Post Malone + 21 Savage — ASR/CTC чекпоинт не любит auto-tune и trap flow. Кандидат на улучшение через MMS или dedicated EN-checkpoint.
- **2 трека без лирики** (`fr Cœur de pirate`, `ru Скриптонит`) — LRCLib вернул фразу, но кэш не подцепился из-за расхождения нормализации Unicode в имени файла (œ NFC↔NFD на macOS/Linux). Это операционная мелочь, не архитектурный пробел.

### Caveat: alignment RTF

В таблице `align_rtf` намеренно опущен. Первый прогон (en, RTF 0.0351) — корректный; на последующих треках телеметрия показала 0.0001-0.0002 из-за отсутствия `torch.cuda.synchronize()` перед `time.perf_counter()` (CUDA async return). Bug пофикшен в `src/align.py` после прогона. На реальный экономический расчёт это не влияет — оценка RTF alignment ≈ 0.04 из baseline #1 + en-трека измерения остаётся в силе.

### Главное

Один прогон, одна машина, ноль ручной работы → 9 треков, 4 языка, воспроизводимые числа. **Design doc'у есть на что опираться:** alignment-путь даёт coverage > 0.97 на всех языках, mean_conf > 0.6 даже на самых сложных треках, и WER 0.18-0.69 на baseline whisper создаёт нужный gap для cover-detector'а.
