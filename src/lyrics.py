"""LRCLib lyric fetcher.

LRCLib (lrclib.net) — публичный API без ключа, без rate-limit'ов в
рамках вежливого использования. Отлично подходит для тестового; в
проде в плеере источников должно быть несколько (Musixmatch, Genius
fallback).

API: https://lrclib.net/docs
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

import requests

API = "https://lrclib.net/api/get"
SEARCH = "https://lrclib.net/api/search"
UA = "unbake-test/0.1 (https://github.com/RezPoint/unbake-test)"


@dataclass
class LyricResult:
    track_name: str
    artist_name: str
    plain_lyrics: str
    synced_lyrics: str | None
    duration: float | None


def _strip_synced_marks(lyrics: str) -> str:
    """LRCLib synced format: '[mm:ss.xx] line'. Strip timestamps."""
    return re.sub(r"\[\d{1,2}:\d{1,2}\.\d{1,3}\]", "", lyrics).strip()


def search(track_name: str, artist_name: str) -> LyricResult | None:
    params = {"track_name": track_name, "artist_name": artist_name}
    r = requests.get(SEARCH, params=params, headers={"User-Agent": UA}, timeout=15)
    r.raise_for_status()
    items = r.json()
    if not items:
        return None
    it = items[0]
    plain = it.get("plainLyrics") or _strip_synced_marks(it.get("syncedLyrics", ""))
    if not plain.strip():
        return None
    return LyricResult(
        track_name=it.get("trackName", track_name),
        artist_name=it.get("artistName", artist_name),
        plain_lyrics=plain,
        synced_lyrics=it.get("syncedLyrics"),
        duration=it.get("duration"),
    )


# Manual mapping: filename → (artist, track) для треков из Yandex-датасета.
DATASET_QUERIES: dict[str, tuple[str, str]] = {
    # ru
    "Miyagi & Эндшпиль - Last of Us.m4a": ("Miyagi & Эндшпиль", "Last of Us"),
    "Pharaoh - Дико, например.m4a": ("Pharaoh", "Дико, например"),
    "Би-2 - Полковнику никто не пишет.m4a": ("Би-2", "Полковнику никто не пишет"),
    "Скриптонит - Танцуй сама.m4a": ("Скриптонит", "Танцуй сама"),
    # es
    "Peso Pluma & Anitta - BELLAKEO.m4a": ("Peso Pluma", "BELLAKEO"),
    "Peso Pluma - BRUCE WAYNE.m4a": ("Peso Pluma", "BRUCE WAYNE"),
    "Peso Pluma - SOLICITADO.m4a": ("Peso Pluma", "SOLICITADO"),
    # en
    "Post Malone & 21 Savage - rockstar.m4a": ("Post Malone", "rockstar"),
    # fr
    "Cœur de pirate - Place de la République.m4a": ("Cœur de pirate", "Place de la République"),
}


def fetch_dataset_lyrics(out_root: Path = Path("data/lyrics")) -> dict[str, str]:
    """Lookup all dataset tracks via LRCLib, save to data/lyrics/<lang>/<track>.txt.

    Returns {filename: status}. Status ∈ {"ok", "miss", "error: ..."}.
    """
    LANG_MAP = {  # filename → lang dir
        "Miyagi & Эндшпиль - Last of Us.m4a": "ru",
        "Pharaoh - Дико, например.m4a": "ru",
        "Би-2 - Полковнику никто не пишет.m4a": "ru",
        "Скриптонит - Танцуй сама.m4a": "ru",
        "Peso Pluma & Anitta - BELLAKEO.m4a": "es",
        "Peso Pluma - BRUCE WAYNE.m4a": "es",
        "Peso Pluma - SOLICITADO.m4a": "es",
        "Post Malone & 21 Savage - rockstar.m4a": "en",
        "Cœur de pirate - Place de la République.m4a": "fr",
    }
    status: dict[str, str] = {}
    for fname, (artist, track) in DATASET_QUERIES.items():
        lang = LANG_MAP[fname]
        out = out_root / lang / fname.replace(".m4a", ".txt")
        if out.exists() and out.stat().st_size > 100:
            status[fname] = "skip (have)"
            continue
        try:
            res = search(track, artist)
        except Exception as e:
            status[fname] = f"error: {e}"
            continue
        if res is None:
            status[fname] = "miss"
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(res.plain_lyrics, encoding="utf-8")
        status[fname] = "ok"
        time.sleep(0.5)
    return status


if __name__ == "__main__":
    import json
    print(json.dumps(fetch_dataset_lyrics(), ensure_ascii=False, indent=2))
