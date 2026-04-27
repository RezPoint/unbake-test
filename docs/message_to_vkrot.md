# Сообщение @vkrot

Скопировать-вставить в Telegram.

---

Привет! Это Дмитрий Ходжа, по тестовому от Unbake AI.

Сделал — design doc + рабочий PoC с воспроизводимыми бенчмарками на 9 треках × 4 языках:

→ **Design doc:** https://github.com/RezPoint/unbake-test/blob/master/docs/design.md
→ **Repo:** https://github.com/RezPoint/unbake-test
→ **Сырые цифры:** https://github.com/RezPoint/unbake-test/blob/master/docs/bench_log.md

Коротко:

1. **Two-pass Shazam-hybrid:** faster-whisper baseline (для lang detect + ASR fallback) + wav2vec2-CTC forced alignment известной лирики на vocal. Когда лирика верна, alignment даёт WER 0 by construction, тайминги точнее ASR, и в ~3× быстрее по RTF.

2. **Замеры на 9 треках × 4 языках** (Yandex-датасет, htdemucs v4 vocal, Colab T4): alignment coverage **mean 0.987** (0.97–1.00), mean confidence **0.76** (0.60–0.91), baseline whisper WER **mean 0.34** (0.18–0.69). Combined cost на A10G — **$0.0007/3-min трек, в 70× ниже потолка $0.05**. Источник лирики — LRCLib API.

3. **Cover-detector:** проверил эмпирически, что одного `mean_confidence` мало — чужая лирика даёт 0.72 vs правильная 0.82, drop всего 12%. Решение через AND с lyric/ASR set-overlap (там разрыв 2.6×, чистый порог 0.5). Это, кажется, основной нетривиальный момент в архитектуре, и в design doc'е есть честный negative result с разбором, **почему** confidence-only не работает.

4. **Что НЕ сделано:** добор IT/PT/JA/PL — notebook готов (`04_collect_extra_langs.ipynb` через yt-dlp + htdemucs v4), но я не стал гонять его до отправки, потому что для proof of architecture хватает 4 языков, а YouTube как источник для прода всё равно надо менять на лицензированные мастера.

Все цифры воспроизводимы на free Colab T4 — 5 ноутбуков в `notebooks/`. Полный full-dataset bench — один клик в `03_full_bench.ipynb` (~15-20 мин, выгружает `results.csv` и `transcripts.tar.gz`). Готов обсуждать на следующем шаге.
