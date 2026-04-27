# План работ

## Дни (примерные)

### День 1 (сегодня) — фундамент
- [x] Скелет репо, eval-модуль (`src/eval/`), smoke-тесты.
- [x] Baseline runner (`src/baseline.py`).
- [ ] Push на GitHub (приватный или публичный — решить).
- [ ] Скачать их датасет с Yandex Disk локально, посмотреть структуру.
- [ ] Прогнать baseline на 1 треке в Colab T4 → есть ли GPU вообще, какие RTF/cost.

### День 2 — alignment-путь
- [ ] Поднять wav2vec2-CTC alignment (или WhisperX) на 2-3 RU треках с известной лирикой.
- [ ] Сравнить vs baseline: WER, timing offset, hallucination rate.
- [ ] Понять fail-modes htdemucs vocal (back-vocals, артефакты на согласных).

### День 3 — сборка датасета и автоматизация
- [ ] 5 треков × 8 языков с лириками (LRCLib + ручной поиск).
- [ ] Ground truth: либо 30-сек ручной разметки на трек, либо WhisperX на чистом вокале как silver standard.
- [ ] Bench-runner: один CLI прогоняет всех кандидатов и пишет CSV.

### День 4 — Shazam-hybrid и cost
- [ ] Прототип forced-alignment пути: лирика → wav2vec2-CTC → word timestamps.
- [ ] Cross-check WER ASR vs alignment для детекции каверов с другим текстом.
- [ ] Расчёт cost/req на A10G/L4 на основе RTF из бенчмарков.

### День 5 — design doc
- [ ] Google Doc на цифрах из бенчмарков.
- [ ] Архитектура, alternatives, риски, железо, метрики.
- [ ] Отправка @vkrot.

## Кандидаты для бенчмарка

| Подход | Где запустить | Что проверяем |
|---|---|---|
| faster-whisper large-v3 + word_timestamps | Colab T4 | базовая accuracy, RTF |
| WhisperX (whisper + wav2vec2 align) | Colab T4 | улучшение timing |
| MMS-1b-all (Meta) | Colab T4 | мультиязычка, особенно JP/PT/PL |
| NeMo Canary-1b | Kaggle T4×2 | accuracy на EN/FR/ES |
| forced alignment лирики (wav2vec2-CTC + ctc-segmentation) | Colab T4 | путь Shazam-hybrid |

## Что НЕ делаем

- Платные API (OpenAI Whisper, Deepgram) — берём цифры из публичных бенчмарков.
- Modal/Replicate.
- On-device inference.
- Скейл-инфра на будущее.
- iOS клиент (не входит в тестовое).
