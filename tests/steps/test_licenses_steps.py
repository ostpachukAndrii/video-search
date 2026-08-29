"""Крокові визначення для features/licenses.feature — доказ вимоги п.6."""

from __future__ import annotations

import sys
from pathlib import Path

from pytest_bdd import parsers, scenarios, then, when

from vsearch.backends.registry import ModelRegistry
from vsearch.licensing import UNVERIFIED, Verdict, audit, classify

scenarios("licenses.feature")


@when("я перевіряю ліцензії всіх моделей у маніфесті")
def _audit_all(context):
    context["report"] = audit(ModelRegistry())


@then("жодна ліцензія не має бути в списку відхилених")
def _no_rejected(context):
    blocking = [f"{f.model} ({f.license}): {f.reason}" for f in context["report"].blocking]
    assert not blocking, "непридатні для комерції ліцензії: " + "; ".join(blocking)


@then(parsers.parse('кожна модель зі станом "{state}" має бути позначена як опційна'))
def _unverified_is_optional(context, state):
    assert state == UNVERIFIED
    not_optional = [f.model for f in context["report"].unverified if not f.optional]
    assert not not_optional, (
        f"модель із непідтвердженою ліцензією не за фіче-флагом: {not_optional}"
    )


@then(parsers.parse('кожна модель зі станом "{state}" має мати примітку про блокер'))
def _unverified_has_note(context, state):
    assert state == UNVERIFIED
    silent = [f.model for f in context["report"].unverified if "БЛОКЕР" not in f.notes]
    assert not silent, f"непідтверджена ліцензія без примітки про блокер: {silent}"


@when(parsers.parse('модель має ліцензію "{license_id}"'))
def _given_license(context, license_id):
    context["verdict"], context["reason"] = classify(license_id)


@then("перевірка ліцензій має її відхилити")
def _must_reject(context):
    assert context["verdict"] is Verdict.REJECTED, (
        f"вердикт {context['verdict']}, а мав бути REJECTED"
    )
    assert context["reason"], "відхилення без пояснення причини непридатне для рішень"


@when("я перевіряю ліцензії встановлених пакетів")
def _scan_installed(context):
    # Скрипт читає `importlib.metadata`, тож перевіряється те саме оточення,
    # у якому підуть тести, а не окремий звіт, що міг застаріти.
    import re
    from importlib.metadata import distributions

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from check_deps import FORBIDDEN, license_of

    found: list[tuple[str, str]] = []
    for dist in distributions():
        name = dist.metadata.get("Name") or "?"
        found.append((name, license_of(dist)))
    context["packages"] = found
    context["forbidden_patterns"] = FORBIDDEN
    context["re"] = re


@then("жоден пакет не має сильного копілефту")
def _no_copyleft(context):
    re = context["re"]
    bad = [
        f"{name} ({text})"
        for name, text in context["packages"]
        if any(re.search(p, text, re.I) for p in context["forbidden_patterns"])
    ]
    assert not bad, f"копілефт у залежностях порушує п.6: {bad}"


@then("жоден пакет не лишається з невизначеною ліцензією")
def _no_unknown(context):
    # «Не вдалося перевірити» — не «можна». Мовчазний дозвіл тут коштував би
    # рівно стільки ж, скільки прямо заборонена ліцензія.
    unknown = [name for name, text in context["packages"] if text == "UNKNOWN"]
    assert not unknown, f"ліцензію не визначено, потрібна ручна перевірка: {unknown}"
