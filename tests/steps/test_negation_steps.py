"""Крокові визначення для negation.feature — вимога п.11."""

from __future__ import annotations

from pathlib import Path

import pytest
from pytest_bdd import parsers, scenarios, then, when

from vsearch.search.query_model import (
    Attribute,
    AttributeName,
    Entity,
    ObjectClass,
    StructuredQuery,
)

scenarios("negation.feature")


@pytest.fixture(scope="session")
def parser():
    """Парсер запиту. Пропускаємо сценарії, якщо ваг немає."""
    from vsearch.search.parse import QueryParser

    instance = QueryParser()
    if not instance.is_available():
        pytest.skip(
            "немає ваг парсера: python scripts/fetch_models.py --only query_parser_gguf"
        )
    return instance


def _find(entities, object_name: str, attribute: str | None, value: str | None):
    for entity in entities:
        if entity.object.value != object_name:
            continue
        if attribute is None:
            return entity
        found = entity.attribute(AttributeName(attribute))
        if found and found.value.lower() == (value or "").lower():
            return entity
    return None


# ── дії ─────────────────────────────────────────────────────────────────────


@when(parsers.parse('я розбираю запит "{query}"'))
def _parse_query(context, parser, query):
    context["query"] = query
    context["bypassed"] = parser.should_bypass(query)
    try:
        context["parsed"] = parser.parse(query)
        context["error"] = None
    except Exception as exc:  # noqa: BLE001 — сам факт помилки і є предметом перевірки
        context["parsed"] = None
        context["error"] = exc


@when("я шукаю з умовою що обʼєкт не має ознаки")
def _search_excluding(context, golden, embedder):
    """Заперечення на синтетиці: виключаємо категорію, яка справді спрацювала.

    Людей тут не намальовано, тож перевіряємо сам МЕХАНІЗМ — чи зникає
    заборонена ознака з видачі — на ознаці, яку цей індекс знає. Категорію
    додаємо самі, бо наперед задані на кольорових фігурах не вмикаються, а
    сценарій, який мовчки пропускається, нічого не захищає.
    """
    from vsearch.represent.categories import CategoryEngine

    engine = CategoryEngine.with_defaults(embedder, golden.store)
    engine.add_user_category("excluded_probe", "a yellow shape")
    report = engine.apply_to_collection("regions", "excluded_probe")
    if not report.positive:
        pytest.skip("категорія не спрацювала на жодному регіоні")

    context["excluded_key"] = "cat_excluded_probe"
    query = StructuredQuery(
        query_en="a shape",
        must=[],
        must_not=[Entity(object=ObjectClass.OTHER, attributes=[])],
    )
    # Умову підмінюємо напряму: сценарій перевіряє фільтрацію, а не здатність
    # LLM вигадати саме цю категорію.
    query = query.model_copy()
    context["structured"] = query
    context["response"] = golden.searcher.search(
        "a shape", limit=10, structured=query, parse=False,
        extra_excluded=[[("cat_excluded_probe", True)]],
    )


@when("я шукаю за умовами, яких індекс не задовольняє")
def _search_impossible(context, golden):
    # Деградація перевіряється через ЇЇ СПРАВЖНІЙ тригер — замало результатів,
    # а не через нездійсненну умову.
    #
    # Раніше тут стояла умова «зброя кольору magenta», і вона працювала, бо
    # невідома ознака мовчки відкидалася. Тепер невідома ознака йде на льоту й
    # лише піднімає кандидатів у видачі, тож унеможливити запит нею більше не
    # можна — і це правильна поведінка, а не поломка. Крім того, на цьому
    # наборі жодна категорія не спрацювала, тож фасетів в індексі немає взагалі.
    impossible = StructuredQuery(
        query_en="a shape",
        must=[Entity(object=ObjectClass.WEAPON, attributes=[])],
    )
    context["response"] = golden.searcher.search(
        "a shape", limit=10, structured=impossible, parse=False,
        min_strict=10_000,
    )


# ── перевірки розбору ───────────────────────────────────────────────────────


@then(parsers.parse('умова "{slot}" має містити обʼєкт "{obj}" з атрибутом "{attr}={value}"'))
def _slot_contains(context, slot, obj, attr, value):
    parsed = context["parsed"]
    assert parsed is not None, f"розбір не вдався: {context['error']}"
    entities = parsed.must if slot == "must" else parsed.must_not
    assert _find(entities, obj, attr, value), (
        f"у {slot} немає {obj} з {attr}={value}; є: "
        + ", ".join(
            f"{e.object.value}{{{','.join(f'{a.name.value}={a.value}' for a in e.attributes)}}}"
            for e in entities
        )
    )


@then(parsers.parse('умова "{slot}" має містити {count:d} обʼєкти'))
def _slot_size(context, slot, count):
    parsed = context["parsed"]
    entities = parsed.must if slot == "must" else parsed.must_not
    assert len(entities) >= count, f"у {slot} лише {len(entities)}, очікувалося {count}"


@then(parsers.parse('умови "{slot}" мають бути порожніми'))
def _slot_empty(context, slot):
    parsed = context["parsed"]
    entities = parsed.must if slot == "must" else parsed.must_not
    assert not entities, f"{slot} не порожній: {entities}"


@then("запит має бути позначений як композитний")
def _is_compositional(context):
    assert context["parsed"].is_compositional, (
        "композитний запит має шукатися по регіонах — інакше ознаки не звʼязані"
    )


@then("англійський опис не має містити слова заперечення")
def _no_negation_in_text(context):
    text = context["parsed"].query_en.lower()
    leaked = [w for w in ("without", "no ", "not ", "немає", "без") if w in text]
    assert not leaked, (
        f"опис {text!r} містить заперечення {leaked}: вектор ловитиме саме те, "
        f"що ми виключаємо"
    )


@then(parsers.parse('мова оригіналу має бути визначена як "{lang}"'))
def _language_detected(context, lang):
    assert context["parsed"].language == lang, (
        f"визначено {context['parsed'].language!r}, очікувалося {lang!r}"
    )


@then(parsers.parse('умова "{slot}" має містити рівно {count:d} обʼєкт'))
def _slot_exact(context, slot, count):
    parsed = context["parsed"]
    entities = parsed.must if slot == "must" else parsed.must_not
    objs = [e.object.value for e in entities]
    assert len(entities) == count, (
        f"у {slot} {len(entities)} обʼєктів замість {count}: {objs}. "
        f"Зайва сутність звужує пошук до нуля."
    )


@then(parsers.parse('умова "{slot}" має містити обʼєкт "{obj}"'))
def _slot_has_object(context, slot, obj):
    parsed = context["parsed"]
    entities = parsed.must if slot == "must" else parsed.must_not
    objs = [e.object.value for e in entities]
    assert obj in objs, f"у {slot} немає {obj}; є {objs}"


@then("розбір має завершитися без помилки")
def _no_error(context):
    assert context["error"] is None, f"розбір впав: {context['error']}"


# ── перевірки пошуку ────────────────────────────────────────────────────────


@then("у видачі не має бути кадрів із цією ознакою")
def _excluded_absent(context, golden):
    key = context["excluded_key"]
    by_frame = {
        payload.get("frame_id"): payload
        for payload in golden.store.scroll_payloads("regions", limit=400)
        if payload.get(key) is True
    }
    returned = {result.frame_id for result in context["response"].results}
    leaked = returned & set(by_frame)
    assert not leaked, f"заборонена ознака {key} пролізла у видачу: {sorted(leaked)[:3]}"


@then("пошук має деградувати до мʼякого")
def _degraded(context):
    assert context["response"].degraded, (
        "жорсткі умови не виконуються жодним кадром, але деградації не сталося"
    )


@then("результат має містити пояснення про зняті обмеження")
def _has_notice(context):
    notice = context["response"].notice
    assert notice and "мʼякий" in notice, (
        f"користувач має знати, що обмеження знято; отримано {notice!r}"
    )


@then("видача не має бути порожньою")
def _not_empty(context):
    assert context["response"].results, (
        "порожня видача гірша за приблизну — саме заради цього існує мʼякий прохід"
    )
