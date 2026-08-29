"""Кроки для `adversarial.feature`.

Тут навмисно немає жодного очікуваного кадру. Кожна перевірка — властивість,
яку можна сформулювати, не знаючи, що саме лежить в індексі: система або
називає свою невпевненість, або мовчить про неї, і саме це й перевіряється.
"""

from __future__ import annotations

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

scenarios("adversarial.feature")


@given("проіндексовано реальні фото з розміткою", target_fixture="real_set")
def _real_set():
    # Той самий набір, що й у `real_photos`: сценарії тут корпусно-незалежні,
    # але їм потрібен НЕПОРОЖНІЙ індекс, інакше вони пройшли б на порожньому.
    from vsearch import goldenset
    from vsearch.search.retrieve import Searcher

    try:
        golden = goldenset.load("real_photos")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"немає набору real_photos: {exc}")
    if golden.missing_files():
        pytest.skip("немає медіафайлів набору real_photos")
    return {"golden": golden, "searcher": Searcher()}


@when(parsers.parse('я шукаю "{query}"'), target_fixture="probe")
def _search(real_set, query):
    return {"response": real_set["searcher"].search(query, limit=10), "query": query}


@when(parsers.parse('я розбираю "{query}"'), target_fixture="probe")
def _parse(query):
    from vsearch.search.parse import get_parser

    return {"parsed": get_parser().parse_or_empty(query), "query": query}


@then("розбіжність перекладу має бути названа у поясненні")
def _translation_named(probe):
    notice = probe["response"].notice or ""
    assert "переклад" in notice.lower(), (
        "переклад не узгоджений з оригіналом, але система про це змовчала; "
        f"пояснення: {notice!r}"
    )


@then("порядок не має спиратися на неузгоджений переклад")
def _translation_not_used(probe):
    # Відкинутий переклад підмінюється оригіналом — саме за ним і йде пошук.
    parsed = probe["response"].parsed
    assert parsed.query_en.strip() == probe["query"].strip(), (
        f"пошук іде за {parsed.query_en!r}, хоча переклад визнано неузгодженим"
    )


@then(parsers.parse('означення "{word}" має лишитися і в тексті, і в ознаці'))
def _modifier_kept(probe, word):
    parsed = probe["parsed"]
    assert word in (parsed.query_en or "").lower(), (
        f"означення {word!r} зникло з query_en: {parsed.query_en!r}"
    )
    values = [
        a.value for e in parsed.must for a in e.attributes
        if a.name.value == "wearing"
    ]
    assert any(word in v.lower() for v in values), (
        f"означення {word!r} зникло з ознаки: {values}"
    )


@then("система має сказати, що нічого не знайдено")
def _honest_absence(probe):
    from vsearch.search.retrieve import MIN_PROBABILITY

    shown = [r for r in probe["response"].results if r.probability >= MIN_PROBABILITY]
    assert not shown, (
        "система віддала впевнені результати на запит про відсутнє: "
        + ", ".join(f"{r.probability:.1%}" for r in shown[:3])
    )


@then("система не має стверджувати, що такого немає")
def _present_not_denied(probe):
    from vsearch.search.retrieve import MIN_PROBABILITY

    response = probe["response"]
    shown = [r for r in response.results if r.probability >= MIN_PROBABILITY]
    relative = [r for r in response.results if r.probability < MIN_PROBABILITY]
    # Достатньо, щоб система НЕ заявила про відсутність: або щось показано, або
    # приховане чесно перелічене, а не оголошене неіснуючим.
    assert shown or not response.notice or "немає" not in response.notice, (
        f"кадр у наборі Є, а система заявила про відсутність: {response.notice!r}; "
        f"кандидатів нижче межі: {len(relative)}"
    )


@when(parsers.parse('я показую видачу для "{query}" в інтерфейсі'),
      target_fixture="probe")
def _through_ui(real_set, query):
    from vsearch.api import ui

    last = None
    for out in ui.do_search(query, profile="balanced", scope="все", limit=6,
                            use_parser=True, category="", show_weak=False,
                            best_only=False, refine=False):
        last = out
    shots, _parsed, _body, _prov, frames = last
    return {"shots": shots, "frames": frames, "query": query}


@then("у галереї має бути стільки ж кадрів, скільки у видачі")
def _nothing_dropped(probe):
    assert len(probe["shots"]) == len(probe["frames"]), (
        f"у видачі {len(probe['frames'])} кадрів, а в галереї {len(probe['shots'])} — "
        "різниця означає, що результат зник мовчки й нумерація зсунулася"
    )
