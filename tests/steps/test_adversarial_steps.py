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


@when(parsers.parse('я порівнюю відсутнє "{absent}" із присутнім "{present}"'),
      target_fixture="probe")
def _search_pair(real_set, absent, present):
    searcher = real_set["searcher"]
    return {
        "absent": (absent, searcher.search(absent, limit=5)),
        "present": (present, searcher.search(present, limit=5)),
    }


#: У скільки разів відсутнє має бути менш упевненим за присутнє. Порядок
#: величини, а не підібране число: між «цього тут немає» і «це тут є» модель
#: розводить на два-три порядки (0.26% проти 55% на реальному наборі), тож
#: десятикратна вимога лишає широкий запас і не залежить від корпусу.
ABSENCE_MARGIN = 10.0


@then("відсутнє має бути на порядок менш упевненим за присутнє")
def _absence_is_relative(probe):
    absent_q, absent = probe["absent"]
    present_q, present = probe["present"]
    top_absent = max((r.probability for r in absent.results), default=0.0)
    top_present = max((r.probability for r in present.results), default=0.0)
    assert top_present > 0, (
        f"запит {present_q!r} нічого не дав — перевірка нічого не міряє"
    )
    assert top_absent * ABSENCE_MARGIN <= top_present, (
        f"{absent_q!r} дає {top_absent:.1%}, а {present_q!r} — {top_present:.1%}: "
        "система однаково впевнена в тому, чого немає, і в тому, що є"
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
    for out in ui.do_search(query, limit=6, use_parser=True, category="",
                            show_weak=False, best_only=False, refine=False):
        last = out
    shots, rows, _status, _parse, _notices, frames, _dump = last
    assert len(rows) == len(shots), (
        f"галерея й таблиця розійшлися: {len(shots)} кадрів проти {len(rows)} рядків — "
        "це означає, що якийсь результат є в одному поданні й відсутній в іншому"
    )
    return {"shots": shots, "frames": frames, "query": query}


@then("у галереї має бути стільки ж кадрів, скільки у видачі")
def _nothing_dropped(probe):
    assert len(probe["shots"]) == len(probe["frames"]), (
        f"у видачі {len(probe['frames'])} кадрів, а в галереї {len(probe['shots'])} — "
        "різниця означає, що результат зник мовчки й нумерація зсунулася"
    )
