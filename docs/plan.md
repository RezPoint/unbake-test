# План работ

## Дни (примерные)

### День 1 (сегодня) — фундамент
- [x] Скелет репо, eval-модуль (`src/eval/`), smoke-тесты.
- [x] Baseline runner (`src/baseline.py`).
- [x] Скачать их датасет с Yandex Disk локально, посмотреть структуру.
- [x] Push на GitHub (публичный — без auth-возни в Colab).
- [x] Прогнать baseline на 1 треке в Colab T4. **RTF=0.118, $0.0012/трек на T4** → cost ниже потолка в 50×.

### Главный сдвиг по цифрам

Cost ($0.05/3min трек) **не является активным constraint** — реалистичный self-hosted setup на T4/A10G даёт $0.001-0.0006 за трек. Это значит:
- Нет смысла оптимизировать cost. Основной приоритет — accuracy.
- Можно тратить compute на двухпроходные схемы: ASR → forced alignment cross-check, ensemble, бóльший beam.
- В design doc этот вывод нужно проиллюстрировать таблицей.

## Что в их датасете

`unbake-vocals/` на Yandex Disk (~50 МБ, 9 треков, только m4a-вокал, лирик нет):

| Lang | Tracks |
|---|---|
| ru (4) | Miyagi & Эндшпиль — Last of Us; Pharaoh — Дико, например; Би-2 — Полковнику никто не пишет; Скриптонит — Танцуй сама |
| es (3) | Peso Pluma & Anitta — BELLAKEO; Peso Pluma — BRUCE WAYNE; Peso Pluma — SOLICITADO |
| en (1) | Post Malone & 21 Savage — rockstar |
| fr (1) | Cœur de pirate — Place de la République |

**Покрыто только 4 языка из заявленных 8.** IT, PT, JP, PL — собирать самим: взять студийные треки → прогнать через htdemucs v4 (htdemucs, не ft) → получить аналогичный вокал. По 2-3 трека на язык хватит.

Лирик в архиве нет — искать вручную (Genius / LRCLib / Musixmatch). Это упомянуть в design doc как ожидаемое.

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
