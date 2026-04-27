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

2. **Замеры на 13 треках × 8 языков** (Yandex + добор yt-dlp/htdemucs v4 для it/pt/ja/pl, Colab T4): alignment coverage **mean 0.988** (0.97–1.00, n=12 без JA-degen), mean confidence **0.78** (0.60–0.91), align RTF **0.032 на per-lang XLSR-53 / 0.097 на MMS-1b-all** (3× — эмпирический trade-off «зоопарк vs единый чекпоинт»). Combined cost на A10G — **$0.0007/3-min трек, в ~70× ниже потолка $0.05**. Источник лирики — LRCLib API.

   Бонус: **3 трека (Скриптонит, Anitta, sanah) — baseline whisper упал по CUDA OOM на T4, а alignment отработал нормально** (coverage 0.99-1.00). Wav2vec2 дешевле по памяти; на T4 один whisper держит, два — нет. Это операционный аргумент за hybrid сверх запланированного.

   JA: word-based метрики (WER, coverage) ломаются на CJK без пробелов — формальный coverage 0.21 при реально корректном alignment (conf 0.87). В design'е выписано, в проде нужна mecab/char-level метрика.

3. **Cover-detector:** проверил эмпирически, что одного `mean_confidence` мало — чужая лирика даёт 0.72 vs правильная 0.82, drop всего 12%. Решение через AND с lyric/ASR set-overlap (там разрыв 2.6×, чистый порог 0.5). Это, кажется, основной нетривиальный момент в архитектуре, и в design doc'е есть честный negative result с разбором, **почему** confidence-only не работает.

Все цифры воспроизводимы на free Colab T4 — 5 ноутбуков в `notebooks/`. `03_full_bench.ipynb` (~15-20 мин) даёт основные 9 треков из Yandex-датасета; `04_collect_extra_langs.ipynb` (~10 мин в той же сессии) добирает it/pt/ja/pl через yt-dlp + htdemucs v4 (для прода YouTube заменяется на лицензированные мастера). Оба выгружают `results.csv` и `transcripts.tar.gz`. Готов обсуждать на следующем шаге.
