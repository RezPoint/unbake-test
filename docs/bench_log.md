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

### Что дальше

1. Найти лирику Pharaoh — Дико, например (Genius/LRCLib) → построить Reference → WER/timing.
2. Прогнать ещё 2-3 трека на T4 → подтвердить RTF ≈ 0.12 как стабильный.
3. Прогнать `large-v3-turbo` для сравнения (быстрее, но что с accuracy на htdemucs-vocal с артефактами?).
4. Подключить wav2vec2-CTC alignment на найденную лирику → главный кандидат уникального угла.
