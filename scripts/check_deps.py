#!/usr/bin/env python3
"""Ліцензії ТРАНЗИТИВНИХ залежностей Python — вимога п.6.

Чому окремо від `check_licenses.py`: той перевіряє ваги моделей за
`models/manifest.lock`. Ваги й код постачаються разом, але ризик у них різний
за природою. Вагу ми обираємо свідомо, по одній, і кожна записана в маніфест.
Залежність приходить сама, транзитивно, і GPL третього рівня не питає дозволу
— вона просто зʼявляється при оновленні чогось зовсім іншого.

Читається `importlib.metadata`, а не `pip-licenses`: дані ті самі (обидва
беруть їх із метаданих пакета), але без зайвої залежності в ланцюгу
постачання. Для проєкту, де кожна вага закріплена sha256, додавати пакет
заради читання поля `License` було б непослідовно.

Політика:

* **Сильний копілефт** (GPL, AGPL, LGPL, SSPL, EUPL) — збірка червона.
  Мережевий копілефт AGPL робить постачання замовнику неможливим без
  відкриття модифікацій; саме через нього відхилено Ultralytics YOLO
  (ADR-011) і ParadeDB/Citus (ADR-001).
* **Некомерційні** (CC-BY-NC) — червона. Через це відхилено ваги InsightFace,
  NLLB і jina-clip-v2.
* **Невідома ліцензія** — червона. «Не вдалося перевірити» це не «можна»:
  саме мовчазний дозвіл і є тим класом помилки, що коштує найдорожче.
* **MPL-2.0** — дозволено з поясненням. Копілефт файловий: обовʼязок
  поширюється на самі файли MPL, а не на роботу, що їх використовує.
"""

from __future__ import annotations

import re
import sys
from importlib.metadata import distributions

#: Сильний копілефт і некомерційні — постачання стає неможливим або
#: обмеженим. Порівнюється за словом цілком, інакше «MIT» ловилося б у
#: «PERMITTED», а «LGPL» ховалося б усередині «GPL».
FORBIDDEN = (
    r"\bA?GPL\b", r"\bLGPL\b", r"\bGPL-?[23]\b", r"\bSSPL\b", r"\bEUPL\b",
    r"\bCC-BY-NC\b", r"\bElastic License\b", r"\bBUSL\b", r"\bnon-?commercial\b",
)
#: Дозволено, але називається вголос: слабкий копілефт має бути видимим
#: рішенням, а не непоміченим рядком у звіті.
NOTED = (r"\bMPL\b", r"\bMozilla\b")


def license_of(dist) -> str:
    """Ліцензія пакета з метаданих, у порядку надійності джерела."""
    meta = dist.metadata
    # `License-Expression` (PEP 639) — машиночитний і однозначний.
    for field in ("License-Expression", "License"):
        value = (meta.get(field) or "").strip()
        # Деякі пакети кладуть у `License` увесь текст ліцензії.
        if value and value.upper() != "UNKNOWN" and "\n" not in value:
            return value
    # Класифікатори — запасний шлях: «License :: OSI Approved :: MIT License».
    tags = [c for c in meta.get_all("Classifier") or [] if c.startswith("License ::")]
    if tags:
        return "; ".join(t.rsplit(" :: ", 1)[-1] for t in tags)
    return "UNKNOWN"


def main() -> int:
    bad: list[tuple[str, str]] = []
    noted: list[tuple[str, str]] = []
    unknown: list[str] = []
    total = 0

    for dist in sorted(distributions(), key=lambda d: (d.metadata.get("Name") or "").lower()):
        name = dist.metadata.get("Name") or "?"
        text = license_of(dist)
        total += 1
        if text == "UNKNOWN":
            unknown.append(name)
        elif any(re.search(p, text, re.I) for p in FORBIDDEN):
            bad.append((name, text))
        elif any(re.search(p, text, re.I) for p in NOTED):
            noted.append((name, text))

    print(f"перевірено пакетів: {total}")
    if noted:
        print("\nслабкий копілефт (дозволено, обовʼязок лише на самі файли MPL):")
        for name, text in noted:
            print(f"  {name:24} {text}")
    if unknown:
        print("\nЛІЦЕНЗІЮ НЕ ВИЗНАЧЕНО — перевірити вручну:")
        for name in unknown:
            print(f"  {name}")
    if bad:
        print("\nЗАБОРОНЕНІ ЛІЦЕНЗІЇ:")
        for name, text in bad:
            print(f"  {name:24} {text}")

    if bad or unknown:
        print("\nвимога п.6 порушена")
        return 1
    print("\nсильного копілефту й невизначених ліцензій немає")
    return 0


if __name__ == "__main__":
    sys.exit(main())
