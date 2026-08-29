#!/usr/bin/env python3
"""A/B-порівняння конфігурацій регіонів на будь-якому золотому наборі.

Існує заради двох питань, відкладених на M2 до появи реальних фото:

  1. Чи окупаються пропозиції Florence-2 порівняно з самими лише плитками?
  2. Чи справді дрібніша сітка гірша, чи це артефакт синтетики?

На синтетичному наборі відповіді були «ні» і «гірша», але там Florence-2
свідомо в невигідних умовах: він тренований на реальних знімках і бачить
намальоване захаращення як одну текстуру. Тому висновок відкладено, а не
зроблено — і ця програма дає його однією командою, щойно зʼявляться дані.

    python scripts/compare_configs.py --golden real_photos
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from eval.metrics import ndcg_at_k, recall_at_k, reciprocal_rank  # noqa: E402
from vsearch import goldenset  # noqa: E402
from vsearch.config import TilingConfig, get_profile  # noqa: E402
from vsearch.index.catalog import Catalog  # noqa: E402
from vsearch.index.store import VectorStore  # noqa: E402
from vsearch.ingest.images import index_paths  # noqa: E402
from vsearch.represent.embed import Siglip2Embedder  # noqa: E402
from vsearch.search.retrieve import Searcher  # noqa: E402


def variants(base):
    """Конфігурації для порівняння. Кожна відрізняється від базової одним."""
    return {
        "лише кадр": dataclasses.replace(
            base, tiling=TilingConfig(tile_size=None), use_region_proposals=False
        ),
        "плитка 384": dataclasses.replace(
            base, tiling=TilingConfig(tile_size=384), use_region_proposals=False
        ),
        "плитка 288": dataclasses.replace(
            base, tiling=TilingConfig(tile_size=288), use_region_proposals=False
        ),
        "плитка 224": dataclasses.replace(
            base, tiling=TilingConfig(tile_size=224), use_region_proposals=False
        ),
        "288 + Florence-2": dataclasses.replace(
            base, tiling=TilingConfig(tile_size=288), use_region_proposals=True
        ),
    }


def evaluate(searcher, golden, queries, k: int) -> dict[str, float]:
    mapping = {Path(a.path).name: a.asset_id for a in golden.assets.values()}
    recalls, mrrs, ndcgs, latencies = [], [], [], []
    for query in queries:
        response = searcher.search(query.text, limit=k)
        latencies.append(response.latency_ms)
        ranked: list[str] = []
        for result in response.results:
            asset = mapping.get(Path(result.path).name)
            if asset and asset not in ranked:
                ranked.append(asset)
        if query.relevant:
            recalls.append(recall_at_k(ranked, query.relevant, k))
            mrrs.append(reciprocal_rank(ranked, query.relevant))
        if query.gains:
            ndcgs.append(ndcg_at_k(ranked, query.gains, k))

    def mean(values):
        return sum(values) / len(values) if values else float("nan")

    return {
        "recall": mean(recalls),
        "mrr": mean(mrrs),
        "ndcg": mean(ndcgs),
        "latency": mean(latencies),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", default="clutter", help="імʼя золотого набору")
    parser.add_argument("--profile", default="balanced")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--only-small",
        action="store_true",
        help="лише запити на дрібні цілі — саме вони вимірюють п.12",
    )
    args = parser.parse_args()

    golden = goldenset.load(args.golden)
    if missing := golden.missing_files():
        print(f"У наборі {args.golden} немає {len(missing)} медіафайлів.", file=sys.stderr)
        return 1

    queries = golden.queries
    if args.only_small:
        queries = [q for q in queries if "tiny" in q.query_id or "small" in q.query_id]
        if not queries:
            print("У наборі немає запитів на дрібні цілі.", file=sys.stderr)
            return 1

    base = get_profile(args.profile)
    embedder = Siglip2Embedder(profile=base)

    print(f"Набір {args.golden}: {len(golden)} активів, {len(queries)} запитів")
    print(f"Ембедер {base.embed_model}, max_num_patches={base.max_num_patches}\n")
    header = f"{'конфігурація':20} {'рег/кадр':>9} {'індекс':>8} {'R@k':>7} {'MRR':>7} {'nDCG':>7} {'пошук':>8}"
    print(header)
    print("-" * len(header))

    rows = []
    for label, profile in variants(base).items():
        store = VectorStore(prefix=f"cmp_{abs(hash(label)) % 10000}_")
        catalog = Catalog(f"/tmp/vsearch_cmp_{abs(hash(label)) % 10000}.db")
        started = time.perf_counter()
        stats = index_paths(
            golden.root / "media",
            profile=profile, store=store, catalog=catalog,
            embedder=embedder, recreate=True,
        )
        index_s = time.perf_counter() - started
        searcher = Searcher(profile=profile, store=store, catalog=catalog, embedder=embedder)
        metrics = evaluate(searcher, golden, queries, args.k)
        rows.append((label, metrics))
        print(
            f"{label:20} {stats.regions_per_frame:9.1f} {index_s:7.1f}с "
            f"{metrics['recall']:7.3f} {metrics['mrr']:7.3f} {metrics['ndcg']:7.3f} "
            f"{metrics['latency']:7.0f}мс"
        )
        store.drop_collection("frames")
        store.drop_collection("regions")

    best = max(rows, key=lambda r: (r[1]["mrr"], r[1]["recall"]))
    print(f"\nНайкраща за MRR: {best[0]}")
    print("Оновіть PROFILES у src/vsearch/config.py, якщо висновок змінився.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
