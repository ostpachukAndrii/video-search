"""Кроки для `notes.feature` — розмітка, зроблена руками.

Набір тимчасовий і самодостатній: коментарі стосуються вмісту, тож для
перевірки досить двох різних файлів і одного каталогу.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

scenarios("notes.feature")

MEDIA = Path(__file__).resolve().parents[2] / "data" / "personal"


@given("коментар до кадру", target_fixture="note")
def _note(tmp_path):
    from vsearch.index.catalog import Catalog
    from vsearch.ingest.images import sha256_of

    sources = sorted(MEDIA.glob("*.png"))[:2]
    if len(sources) < 2:
        pytest.skip("потрібні два зображення в data/personal")
    photo = tmp_path / "same_name.png"
    shutil.copy(sources[0], photo)
    catalog = Catalog(tmp_path / "catalog.db")
    asset_id = sha256_of(photo)
    catalog.save_note(asset_id, "полуниця в ящику внизу", str(photo))
    return {"catalog": catalog, "photo": photo, "asset_id": asset_id,
            "other": sources[1], "text": "полуниця в ящику внизу"}


@when("індекс перебудовується з нуля")
def _rebuild(note):
    note["catalog"].clear_assets()


@then("коментар має лишитися")
def _survives(note):
    assert note["catalog"].notes().get(note["asset_id"]) == note["text"], (
        "перебудова індексу знищила розмітку, зроблену руками — а вона "
        "коштує людського часу, на відміну від векторів"
    )


@when("за тим самим шляхом зʼявляється інше зображення")
def _replace(note):
    shutil.copy(note["other"], note["photo"])


@then("коментар не має стосуватися нового зображення")
def _not_inherited(note):
    from vsearch.ingest.images import sha256_of

    new_id = sha256_of(note["photo"])
    assert new_id != note["asset_id"], "файли виявилися однаковими — сценарій нічого не міряє"
    assert note["catalog"].notes().get(new_id) is None, (
        "новий вміст успадкував чужий коментар: ключем став шлях, а не sha256"
    )


@then(parsers.parse('стан коментаря має бути названий як "{state}"'))
def _state_named(note, state):
    states = {r["asset_id"]: r["state"] for r in note["catalog"].notes_with_state()}
    assert states.get(note["asset_id"]) == state, (
        f"стан {states.get(note['asset_id'])!r}, а мав бути {state!r}: "
        "застарілий коментар лишається невидимим і читається як актуальний"
    )


@when("я стираю текст коментаря")
def _clear(note):
    note["catalog"].save_note(note["asset_id"], "", str(note["photo"]))


@then("коментаря не має лишитися взагалі")
def _gone(note):
    assert not note["catalog"].notes(), (
        "порожній коментар збережено як порожній рядок: «стер» і «не було» "
        "стали різними станами з однаковим виглядом"
    )
