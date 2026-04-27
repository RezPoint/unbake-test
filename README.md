# unbake-test

Тестовое задание для [Unbake AI](https://unbake.net/): API → текст песни с word-level таймстемпами на вокале после htdemucs v4.

## Структура

```
src/eval/      — метрики и бенчмарк-раннер (запускается локально и в Colab)
notebooks/     — Colab/Kaggle ноутбуки для прогонов на GPU
data/          — датасет (gitignored), лирики, ground-truth таймкоды
docs/          — design doc, заметки по решениям
```

## Constraints

- Self-hosted only, без платных API (для финального решения).
- Бенчмарки гоняются на бесплатных Colab T4 / Kaggle T4×2.
- Цель cost: < $0.05 / 3-min трек на проде (A10G / L4).
- Целевые языки: FR, IT, RU, EN, PT, ES, JP, PL.

## Подход (черновик)

Двухступенчатая схема:
1. iOS клиент шлёт ShazamKit-ID + presigned URL.
2. Бэк по ID достаёт лирику → forced alignment (wav2vec2-CTC / MFA) на htdemucs-вокал.
3. Если лирики нет / WER cross-check высокий → fallback на faster-whisper large-v3 + alignment.

Подробнее — `docs/design.md`.
