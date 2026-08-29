"""Тести золотого набору.

Розмітка — джерело істини для всіх метрик, тому помилка в ній тихо псує кожен
наступний вимір. Перевіряємо саму розмітку так само суворо, як і код.
"""

from __future__ import annotations

import json

import pytest

from vsearch import goldenset


def _write_set(tmp_path, name, assets, queries):
    root = tmp_path / name
    root.mkdir(parents=True)
    (root / goldenset.ASSETS_FILE).write_text(
        "\n".join(json.dumps(a, ensure_ascii=False) for a in assets), encoding="utf-8"
    )
    (root / goldenset.QUERIES_FILE).write_text(
        "\n".join(json.dumps(q, ensure_ascii=False) for q in queries), encoding="utf-8"
    )
    return tmp_path


class TestРеальніНабори:
    """Набори з репозиторію мають бути цілісними завжди."""

    @pytest.mark.parametrize("name", goldenset.available())
    def test_набір_завантажується_і_валідний(self, name):
        gs = goldenset.load(name)
        assert gs.assets, f"{name}: порожня розмітка активів"
        assert gs.queries, f"{name}: немає запитів — набір нічого не перевіряє"
        assert gs.validate() == []

    def test_набір_заперечень_має_заборонені_активи(self):
        # Без множини forbidden вимога п.11 нічим не вимірюється.
        gs = goldenset.load("people_glasses")
        negations = [q for q in gs.queries if q.is_negation]
        assert negations, "набір people_glasses без заперечних запитів безглуздий"
        assert all(q.forbidden for q in negations)

    def test_дзеркальні_запити_не_перетинаються(self):
        # «в окулярах» і «без окулярів» мають ділити набір, а не перекриватися.
        gs = goldenset.load("people_glasses")
        by_id = {q.query_id: q for q in gs.queries}
        pos, neg = by_id["pg_pos_001"], by_id["pg_neg_001"]
        assert not (pos.relevant & neg.relevant)
        assert pos.relevant == neg.forbidden

    def test_дрібні_обʼєкти_справді_дрібні(self):
        # Інакше сценарій п.12 «проходив» би на великих обʼєктах.
        gs = goldenset.load("small_objects")
        small = [o for a in gs.assets.values() for o in a.objects if o.is_small]
        assert small, "у наборі small_objects немає жодного обʼєкта менше 2% кадру"


class TestВалідація:
    def test_невідомий_актив_у_запиті_ловиться(self, tmp_path):
        root = _write_set(
            tmp_path,
            "broken",
            [{"asset_id": "a1", "path": "a1.jpg"}],
            [{"query_id": "q1", "text": "тест", "relevant": ["привид"]}],
        )
        problems = goldenset.load("broken", root).validate()
        assert any("невідомі активи" in p for p in problems)

    def test_актив_одночасно_очікуваний_і_заборонений(self, tmp_path):
        root = _write_set(
            tmp_path,
            "broken",
            [{"asset_id": "a1", "path": "a1.jpg"}],
            [{"query_id": "q1", "text": "тест", "relevant": ["a1"], "forbidden": ["a1"]}],
        )
        problems = goldenset.load("broken", root).validate()
        assert any("одночасно" in p for p in problems)

    def test_запит_без_очікувань_ловиться(self, tmp_path):
        root = _write_set(
            tmp_path,
            "broken",
            [{"asset_id": "a1", "path": "a1.jpg"}],
            [{"query_id": "q1", "text": "тест"}],
        )
        problems = goldenset.load("broken", root).validate()
        assert any("нічого не перевіряє" in p for p in problems)

    def test_дублікат_ідентифікатора_запиту(self, tmp_path):
        root = _write_set(
            tmp_path,
            "broken",
            [{"asset_id": "a1", "path": "a1.jpg"}],
            [
                {"query_id": "q1", "text": "a", "relevant": ["a1"]},
                {"query_id": "q1", "text": "b", "relevant": ["a1"]},
            ],
        )
        problems = goldenset.load("broken", root).validate()
        assert any("дублікат" in p for p in problems)

    def test_дублікат_активу_валить_завантаження(self, tmp_path):
        root = _write_set(
            tmp_path,
            "broken",
            [{"asset_id": "a1", "path": "a.jpg"}, {"asset_id": "a1", "path": "b.jpg"}],
            [{"query_id": "q1", "text": "тест", "relevant": ["a1"]}],
        )
        with pytest.raises(goldenset.GoldenSetError, match="дублікат"):
            goldenset.load("broken", root)

    def test_зіпсований_json_вказує_рядок(self, tmp_path):
        root = tmp_path / "broken"
        root.mkdir()
        (root / goldenset.ASSETS_FILE).write_text(
            '{"asset_id": "a1", "path": "a.jpg"}\n{ це не json }\n', encoding="utf-8"
        )
        (root / goldenset.QUERIES_FILE).write_text("", encoding="utf-8")
        with pytest.raises(goldenset.GoldenSetError, match=r":2:"):
            goldenset.load("broken", tmp_path)

    def test_відсутній_набір_підказує_доступні(self, tmp_path):
        with pytest.raises(goldenset.GoldenSetError, match="Доступні"):
            goldenset.load("немає_такого", tmp_path)
