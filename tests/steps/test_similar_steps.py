"""Кроки для `similar.feature` — зворотний звʼязок за Rocchio."""

from __future__ import annotations

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

scenarios("similar.feature")


@given("проіндексовано реальні фото з розміткою", target_fixture="real_set")
def _real_set():
    from vsearch import goldenset
    from vsearch.search.retrieve import Searcher

    try:
        golden = goldenset.load("real_photos")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"немає набору real_photos: {exc}")
    if golden.missing_files():
        pytest.skip("немає медіафайлів набору real_photos")
    return {"golden": golden, "searcher": Searcher()}


@when(
    parsers.parse('я шукаю "{query}" і позначаю найкращий кадр як зразок'),
    target_fixture="probe",
)
def _search_with_seed(real_set, query):
    searcher = real_set["searcher"]
    base = searcher.search(query, limit=5)
    assert base.results, f"запит {query!r} нічого не дав — сценарій нічого не міряє"
    seed = base.results[0]
    return {
        "seed": seed,
        "response": searcher.search(query, limit=5, similar_to=[seed.frame_id]),
    }


@when(
    parsers.parse('я шукаю "{query}" зі зразком, якого немає в індексі'),
    target_fixture="probe",
)
def _search_missing_seed(real_set, query):
    return {
        "seed": None,
        "response": real_set["searcher"].search(
            query, limit=5, similar_to=["немає-такого-кадру:0"]
        ),
    }


@then("зразок має лишитися у видачі")
def _seed_present(probe):
    ids = [r.frame_id for r in probe["response"].results]
    assert probe["seed"].frame_id in ids, (
        "кадр, позначений як зразок, зник із відповіді на власну позначку"
    )


@then("впевненість не має бути однаковою в усіх результатів")
def _not_saturated(probe):
    values = {round(r.probability, 4) for r in probe["response"].results}
    assert len(values) > 1, (
        f"усі результати мають однакову впевненість {values} — шкала насичена, "
        "тобто число нічого не розрізняє"
    )


@then("у поясненні має бути сказано про неврахований зразок")
def _missing_named(probe):
    notice = probe["response"].notice or ""
    assert "зразок" in notice.lower() or "позначен" in notice.lower(), (
        f"позначку не враховано, і система про це змовчала: {notice!r}"
    )
