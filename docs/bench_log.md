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

### Что дальше

1. Прогнать alignment + baseline на остальных 8 треках (RU/ES/EN/FR), повторить замеры → стабильность RTF и coverage.
2. Тест на «не той» лирике: подсунуть лирику другой песни → confidence должна упасть в 2-3× (это валидация cover-detector).
3. ES / EN / FR — взять `facebook/mms-1b-all` (поддерживает 1100+ языков) или per-language wav2vec2 чекпоинты.
4. Prep design doc на этих цифрах.
