"""Крокові визначення для invariants.feature.

Жодного очікуваного результату: перевіряються лише властивості. Тому ці
сценарії не старіють разом із розміткою і їх неможливо «підкрутити» —
підганяти тут просто нема що.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pytest_bdd import given, scenarios, then, when

scenarios("invariants.feature")

#: Запити навмисно поза золотим набором і поза тим, що колись лагодили:
#: перевірка не має залежати від того, що ми вже бачили.
PROBE_QUERIES = (
    "нічна вулиця",
    "a wooden table",
    "людина біля води",
    "щось червоне",
    "документи на столі",
)


@given("індекс із будь-яким наповненням", target_fixture="probe")
def _probe():
    from vsearch.config import get_profile
    from vsearch.index import schema
    from vsearch.index.store import VectorStore
    from vsearch.search.retrieve import Searcher

    store = VectorStore()
    if not store.is_alive():
        pytest.skip("Qdrant недоступний")
    if not store.count(schema.FRAMES):
        pytest.skip("індекс порожній — властивості нема на чому перевіряти")
    return {"searcher": Searcher(), "store": store, "profile": get_profile()}


@when("я виконую довільні запити")
def _run_probe(probe, context):
    context["runs"] = {q: probe["searcher"].search(q, limit=8) for q in PROBE_QUERIES}


@when("я виконую довільні запити з різними лімітами")
def _run_limits(probe, context):
    # Ліміти навмисно ПО ОБИДВА боки від внутрішніх глибин. Раніше тут стояли
    # 3 і 10 — обидва нижчі за `RESCORE_DEPTH`, тож глибина відбору в обох
    # випадках виходила однакова, і властивість перевірялася вхолосту.
    #
    # Дефект вилізе лише на великому корпусі: поки глибина накриває майже весь
    # індекс, будь-який ліміт дає ту саму множину кандидатів. На 384 кадрах
    # `limit=60` уже давав ІНШИЙ порядок верхівки, ніж `limit=30`.
    context["pairs"] = {
        q: (probe["searcher"].search(q, limit=3), probe["searcher"].search(q, limit=60))
        for q in PROBE_QUERIES
    }


@when("я виконую довільні запити двічі")
def _run_twice(probe, context):
    context["twice"] = {
        q: (probe["searcher"].search(q, limit=5), probe["searcher"].search(q, limit=5))
        for q in PROBE_QUERIES
    }


@when("я шукаю з умовою на неіснуючий фасет")
def _unknown_facet(probe, context):
    context["unknown"] = probe["searcher"].search(
        "будь-що", limit=5, category="zzz_такої_категорії_немає"
    )


@then("жоден результат не має програвати сусіду за всіма вимірами")
def _pareto(context):
    """Домінування: якщо кадр гірший за ОБОМА числами, він не може стояти вище.

    Сильніша властивість, ніж монотонність одного числа, і застосовна за
    будь-якої кількості свідчень. При одному вимірі зводиться до звичайної
    монотонності, тож нічого не втрачається.
    """
    eps = 1e-9
    failures = []
    for query, response in context["runs"].items():
        rows = [
            (r.probability, r.entity_confidence if r.entity_confidence is not None else 1.0)
            for r in response.results
        ]
        bad = [
            (i + 1, i + 2)
            for i, (hi, lo) in enumerate(zip(rows, rows[1:]))
            if lo[0] > hi[0] + eps and lo[1] > hi[1] + eps
        ]
        if bad:
            failures.append(f"{query!r}: позиції {bad[:3]}")
    assert not failures, (
        "результат стоїть вище за той, що перевершує його за ОБОМА вимірами — "
        "тобто порядок не пояснюється жодним із показаних чисел:\n  "
        + "\n  ".join(failures)
    )


@then("верхівка коротшої видачі має збігатися з початком довшої")
def _limit_stable(context):
    failures = []
    for query, (short, long) in context["pairs"].items():
        head_short = [Path(r.path).name for r in short.results]
        head_long = [Path(r.path).name for r in long.results][: len(head_short)]
        if head_short and head_short != head_long:
            failures.append(f"{query!r}: {head_short} проти {head_long}")
    assert not failures, (
        "ліміт змінює ПОРЯДОК, а не лише глибину видачі — отже кількість "
        "запитаних результатів мовчки перемикає стратегію пошуку:\n  "
        + "\n  ".join(failures)
    )


@then("обидві видачі мають збігтися")
def _deterministic(context):
    failures = []
    for query, (first, second) in context["twice"].items():
        a = [(Path(r.path).name, round(r.probability, 6)) for r in first.results]
        b = [(Path(r.path).name, round(r.probability, 6)) for r in second.results]
        if a != b:
            failures.append(f"{query!r}")
    assert not failures, (
        "той самий запит дає різні результати — доказ, який неможливо "
        f"відтворити: {failures}"
    )


@then("індекс має містити фасети, які обіцяє профіль")
def _facets_present(probe):
    from vsearch.index import schema

    profile = probe["profile"]
    if not profile.use_region_proposals and not profile.tiling.enabled:
        pytest.skip("профіль не рахує фасети на регіонах")

    found: set[str] = set()
    for payload in probe["store"].scroll_payloads(schema.REGIONS, limit=512):
        found |= {
            key for key in payload
            if key.startswith(("cat_", "attr_")) and not key.endswith("_score")
        }
    assert found, (
        "в індексі НЕМАЄ жодного фасета, хоча профіль їх обіцяє. Щільний пошук "
        "при цьому працює, тож збій непомітний — а заперечення й фільтри за "
        "атрибутами тихо не діють."
    )


@then("система має назвати відкинуту умову")
def _unknown_reported(context):
    response = context["unknown"]
    assert response.notice, (
        "умова на неіснуючий фасет не збігається ніколи, і система повернула "
        "порожнечу без пояснення. Мовчазний нуль неможливо відрізнити від "
        "чесної відсутності матеріалу."
    )


@then("кожен обчислюваний фасет має бути проіндексований")
def _facets_indexed(probe):
    """Перелік індексованих полів звіряється з реєстром прототипів.

    Обидва переліки колись писалися руками окремо й розійшлися непомітно.
    Тепер вони з одного джерела, і ця перевірка стежить, щоб так і лишалося.
    """
    from vsearch.index import schema

    store = probe["store"]
    needed = {f.name for f in schema.facet_fields()}
    missing: dict[str, list[str]] = {}
    for collection in (schema.REGIONS, schema.FRAMES):
        name = store.name(collection)
        if not store.client.collection_exists(name):
            continue
        indexed = set(store.client.get_collection(name).payload_schema or {})
        gap = sorted(needed - indexed)
        if gap:
            missing[collection] = gap
    assert not missing, (
        "фасети без payload-індексу — фільтр за ними виконується повним "
        f"перебором: { {k: v[:6] for k, v in missing.items()} }"
    )
