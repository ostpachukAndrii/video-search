#!/usr/bin/env python3
"""Стенд масштабу: чи тримаються властивості пошуку на мільйонах векторів.

Навіщо. Усі пороги цього проєкту виведені з корпусу в 116 знімків (ризик 13), а
чотири механізми коректні лише поки вибірка накриває майже весь індекс
(ADR-015). Жоден із них не ЗЛАМАЄТЬСЯ на мільйонах — вони дадуть гірші
відповіді. Тому стенд важливіший за самі виправлення: без нього ми перевіряємо
здогади, а не поведінку.

Звідки беруться вектори. НЕ з випадкового шуму. Справжні ембединги згруповані,
і ANN на рівномірно розподілених векторах поводиться зовсім інакше: сусіди
рівновіддалені, обхід графа вироджується, а латентність виходить оптимістичною.
Стенд бере НАЯВНІ вектори індексу й розмножує їх із шумом навколо кожного —
геометрія розподілу зберігається, а розмір росте. Це не «справжні дані», і
стенд цього не стверджує; він стверджує лише, що структура сусідства
правдоподібна.

Фасети розподіляються за ЧАСТОТАМИ реального індексу, а не рівномірно:
селективність фільтра — головне, від чого залежить поведінка filterable HNSW.
`cat_person` на 47% і рідкісна категорія на 0.3% — різні режими обходу, і
міряти треба обидва.

    PYTHONPATH=src .venv/bin/python scripts/scale_bench.py --regions 100_000 --report
    PYTHONPATH=src .venv/bin/python scripts/scale_bench.py --regions 1_000_000 --check-negation
    PYTHONPATH=src .venv/bin/python scripts/scale_bench.py --drop
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

#: Стенд живе в ОКРЕМИХ колекціях. Роздувати робочий індекс не можна: його
#: доведеться перебудовувати цілком, а це години на реальному корпусі.
SUFFIX = "_scalebench"

#: Шум навколо взірця. Досить малий, щоб клон лишався в тому ж кластері, і
#: досить великий, щоб клони не злилися в одну точку — інакше HNSW отримав би
#: мільйон копій однієї вершини й показав неправдиво швидкий обхід.
NOISE = 0.08


def _batches(total: int, size: int):
    done = 0
    while done < total:
        step = min(size, total - done)
        yield done, step
        done += step


def inflate(store, schema, source: str, target: int, batch: int, seed: int) -> int:
    """Наростити колекцію-стенд до `target` точок, зберігши структуру сусідства."""
    from qdrant_client import models

    rng = np.random.default_rng(seed)
    points, _ = store.client.scroll(
        collection_name=store.name(source), limit=4096,
        with_payload=True, with_vectors=True,
    )
    if not points:
        raise SystemExit(f"колекція {source} порожня — стенду немає з чого рости")

    def dense_of(vector):
        if isinstance(vector, dict):
            return vector.get("") or next(iter(vector.values()))
        return vector

    seeds = np.asarray([dense_of(p.vector) for p in points], dtype=np.float32)
    payloads = [dict(p.payload or {}) for p in points]
    dim = seeds.shape[1]
    name = f"{source}{SUFFIX}"
    store.ensure_collection(name, dim)

    written = 0
    for offset, step in _batches(target, batch):
        pick = rng.integers(0, len(seeds), size=step)
        block = seeds[pick] + rng.normal(0.0, NOISE, size=(step, dim)).astype(np.float32)
        block /= np.linalg.norm(block, axis=1, keepdims=True)
        store.client.upsert(
            collection_name=store.name(name),
            points=[
                models.PointStruct(
                    id=str(__import__("uuid").uuid5(
                        __import__("uuid").NAMESPACE_URL, f"{name}:{offset + i}")),
                    vector=block[i].tolist(),
                    # Payload копіюється від того самого взірця, від якого взято
                    # вектор: інакше фасети розійшлися б із геометрією, і фільтр
                    # відбирав би випадкову підмножину простору.
                    payload={**payloads[int(pick[i])],
                             "frame_id": f"bench:{offset + i}",
                             "asset_id": f"bench-{(offset + i) // 8}"},
                )
                for i in range(step)
            ],
            wait=False,
        )
        written += step
        if written % (batch * 20) == 0 or written >= target:
            print(f"  залито {written:>9,} / {target:,}", flush=True)
    print()
    return written


def measure(store, schema, name: str, dim: int, trials: int, facet: str | None) -> dict:
    """Латентність ANN на стенді: без фільтра і з фільтром за фасетом."""
    from vsearch.index.store import build_filter

    rng = np.random.default_rng(0)
    queries = rng.normal(size=(trials, dim)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    def run(flt) -> list[float]:
        times = []
        for q in queries:
            started = time.perf_counter()
            store.search(name, q, limit=10, query_filter=flt)
            times.append((time.perf_counter() - started) * 1000)
        return times

    result = {"плоский": run(None)}
    if facet:
        result[f"фільтр {facet}"] = run(build_filter(must=[(facet, True)]))
    return result


def facet_frequencies(store, name: str, sample: int = 4096) -> list[tuple[str, float]]:
    """Частоти фасетів у стенді — щоб міряти фільтри РІЗНОЇ селективності."""
    points, _ = store.client.scroll(
        collection_name=store.name(name), limit=sample, with_payload=True
    )
    counts: dict[str, int] = {}
    for p in points:
        for key, value in (p.payload or {}).items():
            if key.startswith(("cat_", "attr_")) and value is True:
                counts[key] = counts.get(key, 0) + 1
    total = max(len(points), 1)
    return sorted(((k, v / total) for k, v in counts.items()), key=lambda kv: -kv[1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", type=int, default=100_000,
                    help="скільки точок нарощувати у стенді")
    # Межа не в памʼяті, а в HTTP: Qdrant відкидає тіло понад 32 МБ, а 1152
    # float у JSON — це ~25 КБ на точку. 512 лишає запас.
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", action="store_true", help="показати частоти фасетів")
    ap.add_argument("--check-negation", action="store_true",
                    help="чи лишається заперечення ТОЧНИМ на масштабі")
    ap.add_argument("--drop", action="store_true", help="видалити колекції стенду")
    args = ap.parse_args()

    from vsearch.index import schema
    from vsearch.index.store import VectorStore

    store = VectorStore()
    name = f"{schema.REGIONS}{SUFFIX}"

    if args.drop:
        for base in (schema.REGIONS, schema.FRAMES):
            try:
                store.client.delete_collection(store.name(f"{base}{SUFFIX}"))
                print(f"видалено {base}{SUFFIX}")
            except Exception as exc:  # noqa: BLE001
                print(f"{base}{SUFFIX}: {exc}")
        return 0

    print(f"нарощую стенд до {args.regions:,} точок (взірці з {schema.REGIONS})")
    started = time.perf_counter()
    written = inflate(store, schema, schema.REGIONS, args.regions, args.batch, args.seed)
    store.ensure_payload_indexes(name)
    elapsed = time.perf_counter() - started
    total = store.count(name, exact=False)
    print(f"залито {written:,} за {elapsed:.0f} с ({written / max(elapsed, 1e-9):,.0f} точок/с); "
          f"у колекції {total:,}")

    freqs = facet_frequencies(store, name)
    if args.report and freqs:
        print("\nчастоти фасетів (селективність фільтра):")
        for key, share in freqs[:8]:
            print(f"  {key:28} {share:6.1%}")

    dim = store.client.get_collection(store.name(name)).config.params.vectors.size
    # Міряємо ДВА режими: найчастіший фасет (фільтр майже не звужує) і
    # найрідкісніший (звужує різко). Саме на другому pgvector і ламався б —
    # див. ADR-001, блокер 1.
    picks = [freqs[0][0]] if freqs else []
    if len(freqs) > 3:
        picks.append(freqs[-1][0])
    print(f"\nлатентність, {args.trials} запитів, розмір {dim}:")
    for facet in picks or [None]:
        for label, times in measure(store, schema, name, dim, args.trials, facet).items():
            share = dict(freqs).get(facet, 1.0) if facet else 1.0
            print(f"  {label:28} медіана {statistics.median(times):6.1f} мс   "
                  f"p95 {sorted(times)[int(len(times) * 0.95) - 1]:6.1f} мс"
                  + (f"   (частка {share:.1%})" if label.startswith("фільтр") else ""))

    if args.check_negation:
        print("\nзаперечення на масштабі:")
        print(_negation_report(store, name, freqs))
    return 0


def _negation_report(store, name: str, freqs) -> str:
    """Чи виключає заперечення кадр, що лежить ПОЗА межею вибірки.

    Механізм `must_not` будує заборонені кадри як top-N найсхожіших серед тих,
    що підпадають під заборону. Поки N перевищує колекцію, заперечення точне.
    На мільйонах воно стає ймовірнісним — а пропущене виключення дає ВПЕВНЕНО
    ХИБНИЙ результат, найгірший різновид помилки в розслідуванні (п.11).
    """
    from vsearch.index.store import build_filter
    from vsearch.search.retrieve import STRICT_FETCH_LIMIT

    if not freqs:
        return "  фасетів немає — нічого перевіряти"
    key, share = freqs[0]
    banned = store.count_filtered(name, build_filter(must=[(key, True)])) \
        if hasattr(store, "count_filtered") else int(share * store.count(name, exact=False))
    verdict = "ТОЧНЕ" if banned <= STRICT_FETCH_LIMIT else "ЙМОВІРНІСНЕ"
    return (
        f"  заборонених точок за {key}: ~{banned:,}\n"
        f"  межа вибірки STRICT_FETCH_LIMIT: {STRICT_FETCH_LIMIT:,}\n"
        f"  вердикт: заперечення {verdict}"
        + ("" if verdict == "ТОЧНЕ" else
           "\n  → кадр із забороненою ознакою за межами вибірки НЕ буде виключений (M7c)")
    )


if __name__ == "__main__":
    sys.exit(main())
