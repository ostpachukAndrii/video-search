#!/usr/bin/env python
"""Обумовлена детекція: рамка за словом запиту (ADR-012).

Загальні пропозиції Florence-2 успадковують те саме зміщення в бік помітного,
що й ембединг: модель пропонує те, що вважає головним у кадрі. Детектор, якому
сказали, ЩО шукати, цього зміщення не має.

Скрипт міряє різницю на конкретному знімку: скільки рамок дає кожен режим,
якої вони площі й наскільки впевнено кроп відповідає запиту.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from PIL import Image  # noqa: E402

from vsearch.config import get_profile  # noqa: E402
from vsearch.represent.embed import Siglip2Embedder  # noqa: E402
from vsearch.represent.regions import (  # noqa: E402
    TASK_OPEN_VOCAB,
    TASK_REGION_PROPOSAL,
    Florence2Proposer,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--term", action="append", default=[],
                    help="слово або фраза запиту; можна кілька разів")
    args = ap.parse_args()

    path = Path(args.image)
    image = Image.open(path).convert("RGB")
    profile = get_profile("balanced")
    embedder = Siglip2Embedder(profile=profile)
    scale, bias = embedder.calibration

    def prob(cos: float) -> float:
        import math
        return 1.0 / (1.0 + math.exp(-(scale * float(cos) + bias)))

    print(f"{path.name}  {image.width}×{image.height}\n")

    generic = Florence2Proposer(profile, task=TASK_REGION_PROPOSAL)
    started = time.perf_counter()
    proposals = generic.propose(image)
    print(f"загальні пропозиції <REGION_PROPOSAL>: {len(proposals)} рамок "
          f"за {time.perf_counter()-started:.1f} с")

    for term in args.term:
        detector = Florence2Proposer(profile, task=TASK_OPEN_VOCAB, term=term)
        started = time.perf_counter()
        found = detector.propose(image)
        elapsed = time.perf_counter() - started
        print(f"\n<OPEN_VOCABULARY_DETECTION> «{term}»: {len(found)} рамок "
              f"за {elapsed:.1f} с")
        if not found:
            continue
        crops = [r.crop(image) for r in found]
        vectors = embedder.embed_images(crops)
        text = embedder.embed_texts([term])[0]
        for region, cos in zip(found, vectors @ text):
            print(f"    {region.area_ratio * 100:6.2f}% кадру  "
                  f"впевненість {prob(cos):6.1%}  «{region.label}»  "
                  f"bbox={[round(v, 3) for v in region.bbox]}")

        # Для порівняння: те саме слово на цілому кадрі.
        whole = embedder.embed_images([image])[0]
        print(f"    ── цілий кадр:            впевненість {prob(whole @ text):6.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
