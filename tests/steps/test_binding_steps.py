"""Крокові визначення для binding.feature.

Звʼязування ознак з обʼєктом — місце, де вимоги п.11 і п.12 сходяться: фасет
має сенс лише разом із тим, до чого він прикріплений.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest
from pytest_bdd import parsers, scenarios, then, when

from eval.metrics import binding_clean_rate, binding_inverted
from vsearch.represent.prototypes import KIND_ATTRIBUTE

scenarios("binding.feature")

FETCH_LIMIT = 10


def _rank(searcher, golden, scope: str) -> list[tuple]:
    mapping = {Path(a.path).name: a.asset_id for a in golden.assets.values()}
    runs = []
    for query in golden.queries:
        # parse=False принципово: сценарій порівнює РЕЖИМИ ПОШУКУ, а з
        # розбором запит пішов би в жорсткий прохід по регіонах і scope
        # перестав би на щось впливати — тест мовчки перестав би перевіряти
        # те, що заявляє.
        response = searcher.search(
            query.text, limit=FETCH_LIMIT, scope=scope, parse=False
        )
        ranked: list[str] = []
        for result in response.results:
            asset = mapping.get(Path(result.path).name)
            if asset and asset not in ranked:
                ranked.append(asset)
        runs.append((query, ranked))
    return runs


# ── дії ─────────────────────────────────────────────────────────────────────


@when("я виконую запити на звʼязування")
def _run_default(context, searcher, golden):
    context["runs"] = _rank(searcher, golden, "auto")


@when(parsers.parse('я виконую запити на звʼязування у режимі "{scope}"'))
def _run_scope(context, searcher, golden, scope):
    context.setdefault("by_scope", {})[scope] = _rank(searcher, golden, scope)


@when("я читаю фасети кадрів і регіонів")
def _read_both(context, golden):
    context["frames"] = golden.store.scroll_payloads("frames", limit=50)
    context["regions"] = golden.store.scroll_payloads("regions", limit=200)
    assert context["frames"] and context["regions"], "індекс порожній"


@when("я будую звʼязаний фільтр з двох умов")
def _build_bound(context):
    from vsearch.index.store import build_bound_filter

    context["filter"] = build_bound_filter(
        [("attr_gender", "male"), ("attr_clothing_color", "red")]
    )


# ── перевірки ───────────────────────────────────────────────────────────────


@then("жодна пастка не має стояти вище за релевантний результат")
def _no_inversions(context):
    failures = [
        f"{query.text}: пастка обігнала правильну відповідь (видача {ranked[:4]})"
        for query, ranked in context["runs"]
        if binding_inverted(ranked, query.relevant, query.forbidden)
    ]
    assert not failures, (
        f"інверсій звʼязування {len(failures)} із {len(context['runs'])}:\n  "
        + "\n  ".join(failures)
    )


@then(parsers.parse('режим "{better}" має бути кращим за режим "{worse}"'))
def _scope_comparison(context, better, worse):
    rates = {
        scope: binding_clean_rate(
            (ranked, query.relevant, query.forbidden) for query, ranked in runs
        )
        for scope, runs in context["by_scope"].items()
    }
    assert rates[better] > rates[worse], (
        f"режим {better!r} ({rates[better]:.2f}) не кращий за {worse!r} "
        f"({rates[worse]:.2f}). Якщо вони рівні — набір не перевіряє звʼязування, "
        f"і сценарій нічого не доводить."
    )


@then("на рівні кадру не має бути жодного атрибута людини")
def _no_frame_attributes(context):
    leaked = {
        key
        for payload in context["frames"]
        for key in payload
        if key.startswith("attr_")
    }
    assert not leaked, (
        f"атрибути протекли на рівень кадру: {sorted(leaked)}. "
        f"Разом із cat_person вони утворюють конʼюнкцію, яка стверджує те, "
        f"чого в кадрі немає."
    )


@then("категорії на рівні кадру мають лишитися")
def _frame_categories_kept(context, golden):
    """Категорії обчислюються для кадрів — на відміну від атрибутів людини.

    Перевіряється РЕЄСТР, а не наявність ключів у payload: категорії
    зберігаються розріджено, тобто ключ зʼявляється лише там, де категорія
    спрацювала. На кольорових фігурах не спрацьовує жодна, і це коректно —
    відсутність ключа означає «ні», а не «не рахували».
    """
    from vsearch.represent.prototypes import KIND_CATEGORY

    registered = [r for r in golden.catalog.prototypes() if r["kind"] == KIND_CATEGORY]
    assert registered, (
        "категорії взагалі не обчислювалися для індексу — «це нічна вулиця» "
        "осмислене саме на рівні кадру"
    )


@then("обидві умови мають застосуватися до однієї точки")
def _filter_is_conjunctive(context):
    query_filter = context["filter"]
    assert query_filter is not None
    conditions = query_filter.must or []
    assert len(conditions) == 2, f"очікувалося дві умови must, отримано {len(conditions)}"
    keys = {c.key for c in conditions}
    assert keys == {"attr_gender", "attr_clothing_color"}
    assert not (query_filter.should or []), (
        "should зробив би умови альтернативними — звʼязування зникло б"
    )


def test_атрибути_мають_тип_який_виключає_рівень_кадру():
    """Страховка на рівні даних, а не лише конвеєра.

    Якщо колись хтось перепризначить тип атрибута на категорію, він знову
    почне проставлятися на кадрах — і мовчки поверне ту саму помилку.
    """
    from vsearch.represent.categories import DEFAULT_ATTRIBUTES

    for prototype in DEFAULT_ATTRIBUTES:
        assert prototype.kind == KIND_ATTRIBUTE, (
            f"{prototype.name} має тип {prototype.kind!r}: як категорія він "
            f"потрапить на рівень кадру і зламає звʼязування"
        )


# ── усі знахідки кадру, а не лише найкраща ──────────────────────────────────


@then("результати мають містити більше однієї ділянки на кадр")
def _multiple_regions(context, searcher, golden):
    responses = [
        searcher.search(query.text, limit=5, scope="auto", parse=False)
        for query in golden.queries[:3]
    ]
    counts = [len(r.regions) for response in responses for r in response.results]
    assert counts, "порожня видача нічого не доводить"
    assert max(counts) > 1, (
        f"на жодному кадрі не показано більше однієї ділянки (максимум {max(counts)}). "
        f"У наборі binding кожен кадр містить два обʼєкти, тож щонайменше на "
        f"частині запитів мають знайтися обидва."
    )
    context["responses"] = responses


@then("ділянки мають бути впорядковані за оцінкою")
def _regions_sorted(context):
    for response in context["responses"]:
        for result in response.results:
            scores = [region.score for region in result.regions]
            assert scores == sorted(scores, reverse=True), (
                f"ділянки не за спаданням оцінки: {scores}"
            )
            if result.regions:
                assert result.bbox == result.regions[0].bbox, (
                    "головна рамка має збігатися з найкращою ділянкою"
                )


@then("ділянки не мають дублювати одна одну")
def _regions_deduplicated(context):
    from vsearch.represent.tiling import Region

    for response in context["responses"]:
        for result in response.results:
            regions = [Region(*r.bbox, kind=r.kind) for r in result.regions]
            for first, second in itertools.combinations(regions, 2):
                assert first.iou(second) < 0.9, (
                    f"дві майже однакові рамки в одному кадрі: "
                    f"{first.bbox} і {second.bbox}. Florence-2 пропонує вкладені "
                    f"рамки (людина, потім її обличчя) — для показу вони мають зливатися."
                )
