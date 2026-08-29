#!/usr/bin/env python3
"""Наскільки формулювання канонічної фрази рухає результат.

Питання виникло з вимірної суперечності (ADR-024): `a pepsi can` дає 10.3%, а
`a can of pepsi` — 0.23% НА ТОМУ САМОМУ кадрі й тій самій плитці. Обидві фрази
англійські й обидві правильні. Отже число, яким система звітує про
впевненість, залежить від того, як LLM склала речення.

Цей скрипт міряє, наскільки це загальне явище, а не окремий випадок, і чи
знімає його злиття кількох формулювань за рангами — той самий механізм, що вже
розвʼязав злиття мовних каналів (ADR-013) і двох свідчень (ADR-019).

Свідомо НЕ вимірюється «яка фраза краща»: обирати формулювання під золотий
набір означало б підганяти систему під 30 знімків. Міряється РОЗКИД, який дає
формулювання, і чи зменшує його злиття.

    PYTHONPATH=src .venv/bin/python scripts/probe_phrasing.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

#: Перефразування будуються ЗА ПРАВИЛОМ, а не вигадуються під набір: кожне
#: правило — загальна властивість англійської, застосовна до будь-якого
#: запиту. Це навмисно, бо перелік фраз, підібраний під конкретні запити, і є
#: підгонкою, якої проєкт уникає.
def variants(phrase: str) -> list[str]:
    """Формулювання того самого запиту за загальними правилами мови."""
    out = [phrase]
    words = phrase.split()
    # «a X of Y» → «a Y X»: саме ця пара й дала розрив у 44 рази.
    if len(words) >= 4 and words[0] in {"a", "an"} and "of" in words:
        i = words.index("of")
        head, tail = words[1:i], words[i + 1 :]
        if head and tail:
            out.append(" ".join(["a", *tail, *head]))
    # Без артикля: підписи в навчальних парах часто без нього.
    if words[0] in {"a", "an", "the"}:
        out.append(" ".join(words[1:]))
    # «фото X» — формулювання, ближче до підпису, ніж до запиту.
    out.append(f"a photo of {' '.join(words[1:]) if words[0] in {'a','an','the'} else phrase}")
    seen: list[str] = []
    for v in out:
        v = v.strip()
        if v and v not in seen:
            seen.append(v)
    return seen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="real_photos")
    ap.add_argument("--limit", type=int, default=10)
    args = ap.parse_args()

    from vsearch.search.parse import get_parser
    from vsearch.search.retrieve import Searcher

    base = ROOT / "tests" / "golden" / args.golden
    assets = {
        a["asset_id"]: Path(a["path"]).name
        for a in (json.loads(l) for l in (base / "assets.jsonl").read_text().splitlines() if l.strip())
    }
    queries = [json.loads(l) for l in (base / "queries.jsonl").read_text().splitlines() if l.strip()]

    searcher = Searcher()
    spreads: list[float] = []
    rank_spreads: list[int] = []

    print(f"{'запит':26} {'формулювань':>11} {'впевненість min→max':>24} {'позиція цілі':>16}")
    for q in queries:
        parsed = get_parser().parse_or_empty(q["text"])
        canonical = (parsed.query_en or q["text"]).strip()
        relevant = {assets[a] for a in q["relevant"] if a in assets}
        probs: list[float] = []
        ranks: list[int] = []
        for phrase in variants(canonical):
            res = searcher.search(phrase, limit=args.limit, parse=False)
            names = [Path(r.path).name for r in res.results]
            hit = [i for i, n in enumerate(names, 1) if n in relevant]
            probs.append(res.results[0].probability if res.results else 0.0)
            ranks.append(hit[0] if hit else 99)
        if len(probs) < 2:
            continue
        spread = (max(probs) / max(min(probs), 1e-6))
        spreads.append(spread)
        rank_spreads.append(max(ranks) - min(ranks))
        print(f"  {q['text'][:24]:24} {len(probs):11} "
              f"{min(probs):9.2%} → {max(probs):8.2%}  ×{spread:<6.1f} "
              f"{min(ranks):>3}–{max(ranks):<3}")

    print(f"\nмедіанний розкид впевненості від формулювання: ×{statistics.median(spreads):.1f}")
    print(f"медіанний розкид позиції цілі:                 {statistics.median(rank_spreads):.0f}")
    print("\nРозкид — це те, чого система НЕ повинна мати: формулювання добирає\n"
          "не людина, а парсер, і воно не є свідченням про кадр.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
