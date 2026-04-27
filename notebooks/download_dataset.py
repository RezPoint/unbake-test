"""Download Unbake's Yandex-Disk dataset to data/raw/<lang>/<file>.m4a.

Idempotent: skips files that already exist with matching size.
Run once locally; commit nothing (data/ is gitignored).
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import quote

import requests

PUBLIC_KEY = "https://disk.yandex.com/d/aGtcKCVEnii2bw"
API = "https://cloud-api.yandex.net/v1/disk/public/resources"
DOWNLOAD_API = "https://cloud-api.yandex.net/v1/disk/public/resources/download"


def list_dir(path: str) -> list[dict]:
    r = requests.get(API, params={"public_key": PUBLIC_KEY, "path": path, "limit": 200})
    r.raise_for_status()
    return r.json()["_embedded"]["items"]


def get_download_url(path: str) -> str:
    r = requests.get(DOWNLOAD_API, params={"public_key": PUBLIC_KEY, "path": path})
    r.raise_for_status()
    return r.json()["href"]


def download(out_root: Path) -> None:
    languages = [item["name"] for item in list_dir("/") if item["type"] == "dir"]
    print(f"languages: {languages}")
    for lang in languages:
        items = list_dir(f"/{lang}")
        out_dir = out_root / lang
        out_dir.mkdir(parents=True, exist_ok=True)
        for it in items:
            if it["type"] != "file":
                continue
            name = it["name"]
            size = it.get("size", 0)
            dst = out_dir / name
            if dst.exists() and dst.stat().st_size == size:
                print(f"skip {lang}/{name} ({size} B, already on disk)")
                continue
            href = get_download_url(f"/{lang}/{name}")
            with requests.get(href, stream=True) as resp:
                resp.raise_for_status()
                with open(dst, "wb") as f:
                    for chunk in resp.iter_content(64 * 1024):
                        f.write(chunk)
            print(f"got  {lang}/{name} ({dst.stat().st_size} B)")


if __name__ == "__main__":
    root = Path(os.environ.get("DATA_DIR", "data/raw"))
    download(root)
    print("\ndone")
