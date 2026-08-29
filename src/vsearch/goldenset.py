"""Золотий набір: розмітка + запити з очікуваннями.

Виконує подвійну роль і тому окуповує роботу двічі:
  1. еталон для метрик у BDD-сценаріях;
  2. калібрувальна вибірка для порогів фасетів (`glasses`, `headwear`, …),
     бо фасет — це косинус із текстовим прототипом, і поріг звідкись треба взяти.

Формат — два JSONL поруч у теці набору:
  assets.jsonl   — що насправді зображено (ground truth)
  queries.jsonl  — запит і які активи він має (і не має) повернути

JSONL, а не один JSON, свідомо: розмітка дописується рядками руками й
переглядається в diff по одному активу.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GOLDEN_DIR = REPO_ROOT / "tests" / "golden"

ASSETS_FILE = "assets.jsonl"
QUERIES_FILE = "queries.jsonl"


class GoldenSetError(RuntimeError):
    """Набір відсутній або його розмітка не тримається схеми."""


@dataclass(frozen=True)
class ObjectAnnotation:
    """Обʼєкт на зображенні. `area_ratio` потрібен для сценаріїв п.12."""

    label: str
    bbox: tuple[float, float, float, float]  # x, y, w, h у частках кадру
    area_ratio: float

    @property
    def is_small(self) -> bool:
        """Поріг, нижче якого обʼєкт зникає з ембедингу цілого кадру."""
        return self.area_ratio < 0.02


@dataclass(frozen=True)
class Asset:
    asset_id: str
    path: str
    media_type: str  # "image" | "video"
    #: Булеві та категорійні мітки: {"glasses": True, "gender": "male"}.
    labels: dict[str, Any] = field(default_factory=dict)
    objects: tuple[ObjectAnnotation, ...] = ()
    ts_ms: int | None = None  # для кадру з відео
    notes: str = ""

    def has_label(self, key: str, value: Any) -> bool:
        return self.labels.get(key) == value


@dataclass(frozen=True)
class Query:
    """Запит із очікуваннями.

    `forbidden` — не «нерелевантні», а ті, поява яких означає провал заперечення.
    Саме ця множина робить п.11 вимірюваним.
    """

    query_id: str
    text: str
    lang: str
    relevant: frozenset[str] = frozenset()
    forbidden: frozenset[str] = frozenset()
    #: Градуйована релевантність для nDCG: {asset_id: gain}.
    gains: dict[str, float] = field(default_factory=dict)
    #: Очікуваний розбір запиту (must / must_not) для сценаріїв парсера.
    expected_parse: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    @property
    def is_negation(self) -> bool:
        return bool(self.forbidden) or bool(self.expected_parse.get("must_not"))


@dataclass
class GoldenSet:
    name: str
    root: Path
    assets: dict[str, Asset]
    queries: list[Query]

    def __len__(self) -> int:
        return len(self.assets)

    def assets_with(self, key: str, value: Any) -> list[str]:
        return [a.asset_id for a in self.assets.values() if a.has_label(key, value)]

    def missing_files(self) -> list[str]:
        """Активи, розмічені в JSONL, але відсутні на диску."""
        return [a.asset_id for a in self.assets.values() if not (self.root / a.path).exists()]

    def validate(self) -> list[str]:
        """Перевірити цілісність розмітки. Порожній список = все гаразд."""
        problems: list[str] = []
        seen_queries: set[str] = set()

        for query in self.queries:
            if query.query_id in seen_queries:
                problems.append(f"запит {query.query_id}: дублікат ідентифікатора")
            seen_queries.add(query.query_id)

            unknown = (query.relevant | query.forbidden) - self.assets.keys()
            if unknown:
                problems.append(f"запит {query.query_id}: невідомі активи {sorted(unknown)}")

            both = query.relevant & query.forbidden
            if both:
                problems.append(
                    f"запит {query.query_id}: активи {sorted(both)} одночасно "
                    f"очікувані й заборонені"
                )

            unknown_gains = query.gains.keys() - self.assets.keys()
            if unknown_gains:
                problems.append(
                    f"запит {query.query_id}: gains для невідомих активів {sorted(unknown_gains)}"
                )

            if not query.relevant and not query.forbidden:
                problems.append(
                    f"запит {query.query_id}: немає ні очікуваних, ні заборонених активів — "
                    f"такий запит нічого не перевіряє"
                )

        return problems


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        raise GoldenSetError(f"Немає {path}")
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise GoldenSetError(f"{path}:{lineno}: {exc}") from exc


def load(name: str, golden_dir: Path | str | None = None) -> GoldenSet:
    """Завантажити золотий набір за іменем теки."""
    root = Path(golden_dir or DEFAULT_GOLDEN_DIR) / name
    if not root.is_dir():
        raise GoldenSetError(
            f"Немає золотого набору {name!r} у {root}. "
            f"Доступні: {', '.join(available(golden_dir)) or '—'}"
        )

    assets: dict[str, Asset] = {}
    for row in _read_jsonl(root / ASSETS_FILE):
        objects = tuple(
            ObjectAnnotation(
                label=o["label"],
                bbox=tuple(o["bbox"]),  # type: ignore[arg-type]
                area_ratio=float(o["area_ratio"]),
            )
            for o in row.get("objects", [])
        )
        asset = Asset(
            asset_id=row["asset_id"],
            path=row["path"],
            media_type=row.get("media_type", "image"),
            labels=row.get("labels", {}),
            objects=objects,
            ts_ms=row.get("ts_ms"),
            notes=row.get("notes", ""),
        )
        if asset.asset_id in assets:
            raise GoldenSetError(f"дублікат asset_id {asset.asset_id!r} у {root / ASSETS_FILE}")
        assets[asset.asset_id] = asset

    queries = [
        Query(
            query_id=row["query_id"],
            text=row["text"],
            lang=row.get("lang", "uk"),
            relevant=frozenset(row.get("relevant", [])),
            forbidden=frozenset(row.get("forbidden", [])),
            gains={k: float(v) for k, v in row.get("gains", {}).items()},
            expected_parse=row.get("expected_parse", {}),
            notes=row.get("notes", ""),
        )
        for row in _read_jsonl(root / QUERIES_FILE)
    ]

    return GoldenSet(name=name, root=root, assets=assets, queries=queries)


def available(golden_dir: Path | str | None = None) -> list[str]:
    base = Path(golden_dir or DEFAULT_GOLDEN_DIR)
    if not base.is_dir():
        return []
    return sorted(p.name for p in base.iterdir() if (p / ASSETS_FILE).exists())
