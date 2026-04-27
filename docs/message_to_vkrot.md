# Сообщение @vkrot

Скопировать-вставить в Telegram.

---

Привет! Это Дмитрий Ходжа, по тестовому от Unbake AI.

Сделал — design doc + рабочий PoC с воспроизводимыми бенчмарками:

→ **Design doc:** https://github.com/RezPoint/unbake-test/blob/master/docs/design.md
→ **Repo:** https://github.com/RezPoint/unbake-test
→ **Сырые цифры бенчмарков:** https://github.com/RezPoint/unbake-test/blob/master/docs/bench_log.md

Коротко:

1. Two-pass Shazam-hybrid: faster-whisper baseline + wav2vec2-CTC forced alignment известной лирики на vocal.
2. Замеры на 1 RU-треке dataset'а (Pharaoh — Дико, например, htdemucs v4 vocal): baseline whisper WER 0.347 → alignment coverage 0.996, mean confidence 0.82, RTF 0.038 на free Colab T4. Combined cost на A10G — $0.0008/3-min трек, в 60× ниже потолка $0.05.
3. Cover-detector: проверил эмпирически, что одного `mean_confidence` мало — чужая лирика даёт 0.72 vs правильная 0.82, drop всего 12%. Решение через AND с lyric/ASR set-overlap (там разрыв 2.6×, чистый порог 0.5). Это, кажется, основной нетривиальный момент в архитектуре.
4. Что НЕ сделано: расширение до остальных 8 треков и до ES/EN/FR (план — §10 design doc'а). Я сознательно выбрал глубину: довести один RU-кейс до проверенного cover-detector'а вместо широкого, но поверхностного прохода.

Все цифры воспроизводимы на free Colab T4 — 3 ноутбука в `notebooks/`. Готов обсуждать на следующем шаге.
