"""Спільна підготовка для тестів.

Пакет ще не встановлюється в оточення на M0 (важкі залежності попереду), тому
шляхи додаються тут. Коли зʼявиться `pip install -e .`, цей блок зникне.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def context() -> dict:
    """Кошик стану між кроками одного сценарію.

    pytest-bdd прокидає фікстури між кроками, але заводити окрему фікстуру на
    кожну проміжну величину — зайвий шум. Один явний словник читабельніший.
    """
    return {}


# ── фікстури для сценаріїв пошуку ───────────────────────────────────────────
#
# Тести працюють на тому самому Qdrant, що й розробник, тому живуть у власному
# просторі імен колекцій і з власним каталогом у tmp. Прогін тестів не має
# зачіпати робочий індекс.

TEST_PREFIX = "test_"


def _skip_unless_ready() -> None:
    """Пропустити сценарій, якщо ваг або Qdrant немає.

    Пропуск, а не падіння: відсутність 4.5 ГБ ваг на чужій машині — це не
    поламаний код. А от якщо ваги є, а Qdrant лежить — це вже привід сказати
    прямо, що саме треба підняти.
    """
    from vsearch.backends.registry import get_registry
    from vsearch.config import get_profile
    from vsearch.index.store import VectorStore

    model = get_profile().embed_model
    if not get_registry().is_fetched(model):
        pytest.skip(f"немає ваг {model}: python scripts/fetch_models.py --only {model}")
    if not VectorStore().is_alive():
        pytest.skip(
            "Qdrant недоступний: docker run -d -p 6333:6333 "
            "-v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant"
        )


@pytest.fixture(scope="session")
def embedder():
    _skip_unless_ready()
    from vsearch.config import get_profile
    from vsearch.represent.embed import Siglip2Embedder

    return Siglip2Embedder(profile=get_profile("balanced"))


@dataclass
class IndexedSet:
    """Золотий набір разом із пошуком, привʼязаним саме до нього."""

    golden: object
    searcher: object
    store: object
    catalog: object

    def __getattr__(self, item):
        # Кроки звертаються до набору напряму (`golden.queries`, `golden.root`),
        # тож прозоро делегуємо все, чого немає тут.
        return getattr(self.golden, item)


@pytest.fixture(autouse=True)
def _release_gpu_memory():
    """Звільняти памʼять прискорювача після кожного тесту.

    Не гігієна, а необхідність. У повному прогоні одночасно живуть SigLIP
    (4.5 ГБ), Florence-2 і Qwen3, а кожен BDD-сценарій ще й індексує набір
    сотнями кропів. Кеш MPS доростав до 23.8 ГБ при межі 30.2 — і починав
    падати з `MPS backend out of memory` на випадкових тестах.

    Симптом був оманливий: ті самі файли поодинці зелені, у повному прогоні
    червоні. Це виглядало як взаємний вплив тестів, хоча фікстури ізольовані
    коректно, — і коштувало кількох хибних гіпотез, перш ніж у трасуванні
    знайшлося справжнє повідомлення.

    На Apple Silicon памʼять єдина, тож у RSS процесу цього не видно взагалі.
    """
    yield
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 — прибирання не мусить валити тест
        pass


@pytest.fixture(scope="session")
def indexed_sets(embedder, tmp_path_factory):
    """Ліниво індексує золотий набір у ВЛАСНУ колекцію.

    Кожен набір отримує окремий префікс колекцій і окремий каталог. Спільна
    колекція здавалася економією, але давала гірше: `smoke` і `clutter`
    лягали разом, чужі кадри потрапляли у видачу, і сценарії зеленіли лише
    тоді, коли модуль запускали окремо. Тест, що проходить лише наодинці,
    гірший за відсутній.

    Індексація коштує секунд, тому фікстура session-scoped: один раз на набір.
    """
    from vsearch import goldenset
    from vsearch.config import get_profile
    from vsearch.index.catalog import Catalog
    from vsearch.index.store import VectorStore
    from vsearch.ingest.images import index_paths
    from vsearch.search.retrieve import Searcher

    done: dict[str, IndexedSet] = {}
    root = tmp_path_factory.mktemp("catalogs")

    def ensure(name: str) -> IndexedSet:
        if name in done:
            return done[name]
        gs = goldenset.load(name)
        if gs.missing_files():
            pytest.skip(
                f"у наборі {name} немає медіафайлів "
                f"({len(gs.missing_files())} з {len(gs)}). "
                f"Згенерувати синтетичні: python scripts/make_smoke_set.py"
            )
        profile = get_profile("balanced")
        store = VectorStore(prefix=f"{TEST_PREFIX}{name}_")
        catalog = Catalog(root / f"{name}.db")
        index_paths(
            gs.root / "media",
            profile=profile, store=store, catalog=catalog,
            embedder=embedder, recreate=True,
        )
        done[name] = IndexedSet(
            golden=gs,
            searcher=Searcher(profile=profile, store=store, catalog=catalog, embedder=embedder),
            store=store,
            catalog=catalog,
        )
        return done[name]

    return ensure


@pytest.fixture
def searcher(golden):
    """Пошук, привʼязаний до набору з передумови сценарію."""
    return golden.searcher


# ── спільні кроки BDD ───────────────────────────────────────────────────────
#
# pytest-bdd не ділиться кроковими визначеннями між тестовими модулями, але
# бачить їх у conftest. Передумова «проіндексовано золотий набір» потрібна
# кільком функціоналам, тож живе тут, а не дублюється.

from pytest_bdd import given, parsers  # noqa: E402


@given(parsers.parse('проіндексовано золотий набір "{name}"'), target_fixture="golden")
def _given_indexed_golden_set(indexed_sets, name):
    return indexed_sets(name)
