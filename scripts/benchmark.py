#!/usr/bin/env python3
"""Замір затримки та якості по профілях і пристроях (п.10).

Профіль "fast" завжди виграв би за часом, просто повертаючи гірші результати,
тому затримка й якість міряються разом і виводяться поруч. Поки пошуковий
бекенд не готовий (M1), скрипт чесно повідомляє про це, а не малює нулі.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from eval.metrics import percentile  # noqa: E402
from vsearch.backends import device  # noqa: E402
from vsearch.config import PROFILES  # noqa: E402


def load_search_backend():
    """Повернути виклик пошуку або None, якщо бекенда ще немає."""
    try:
        from vsearch.search.retrieve import search  # type: ignore[attr-defined]
    except (ImportError, AttributeError):
        return None
    return search


def measure(search, queries, profile: str, repeats: int) -> dict[str, float]:
    latencies: list[float] = []
    for _ in range(repeats):
        for query in queries:
            start = time.perf_counter()
            search(query, profile=profile)
            latencies.append((time.perf_counter() - start) * 1000)
    return {
        "n": len(latencies),
        "p50": percentile(latencies, 0.50),
        "p95": percentile(latencies, 0.95),
        "max": max(latencies),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="cpu|mps|cuda|all|auto")
    parser.add_argument("--golden", default="people_glasses", help="набір запитів")
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()

    targets = (
        device.available_devices()
        if args.device in ("all", "auto")
        else [args.device]
    )
    print(f"Пристрої для заміру: {', '.join(targets)}")
    print(f"Профілі: {', '.join(PROFILES)}\n")

    search = load_search_backend()
    if search is None:
        print(
            "Пошуковий бекенд ще не реалізований (заплановано на M1).\n"
            "Гарнес готовий: щойно зʼявиться vsearch.search.retrieve.search,\n"
            "заміри підуть без змін у цьому скрипті.",
            file=sys.stderr,
        )
        return 2

    from vsearch import goldenset

    queries = [q.text for q in goldenset.load(args.golden).queries]
    for target in targets:
        for profile in PROFILES:
            stats = measure(search, queries, profile, args.repeats)
            print(
                f"{target:5} {profile:9} n={stats['n']:4} "
                f"p50={stats['p50']:7.1f}мс  p95={stats['p95']:7.1f}мс  "
                f"max={stats['max']:7.1f}мс"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
