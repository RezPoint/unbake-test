# Colab / Kaggle setup

Бесплатные T4 — рабочая лошадка для всех бенчмарков.

## Cell 1 — клон + зависимости

```python
!git clone https://github.com/<USER>/unbake-test.git
%cd unbake-test
!pip install -q faster-whisper jiwer pydantic
```

## Cell 2 — скачивание датасета (Yandex Disk)

```python
# публичная Yandex-ссылка → прямой URL через API
import requests, os
PUBLIC = "https://disk.yandex.com/d/aGtcKCVEnii2bw"
api = "https://cloud-api.yandex.net/v1/disk/public/resources/download"
r = requests.get(api, params={"public_key": PUBLIC}).json()
# r['href'] — одноразовый прямой URL (если это zip всего датасета)
!wget -q -O dataset.zip "{r['href']}"
!unzip -q dataset.zip -d data/
!ls data/ | head
```

(Если на верхнем уровне директория, а не один файл — нужно листать `/v1/disk/public/resources` рекурсивно. Допишем при первом прогоне.)

## Cell 3 — baseline на одном треке

```python
from src.baseline import transcribe
t, tele = transcribe("data/<file>.m4a", language="ru")
print(tele)
print(t.words[:10])
```

## Cell 4 — eval против лирики

```python
from src.eval.schema import Reference, Word
from src.eval.metrics import evaluate
# собрать Reference из заранее размеченной лирики (см. data/refs/<track>.json)
```

## GPU sanity check

```python
import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

Должно быть `Tesla T4`.

## Лимиты

- Colab Free: ~12ч непрерывно, могут отключить если идле или слишком долго.
- Kaggle: 30ч/нед, более стабильно для длинных прогонов.
- Для длинного прогона (десятки треков) — Kaggle.
