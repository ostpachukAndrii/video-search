"""Юніт-тести метрик.

Метрики — це вимірювальний прилад для решти системи, тому вони мають бути
перевіреними до того, як щось почнуть вимірювати.
"""

from __future__ import annotations

import pytest

from eval.metrics import (
    average_precision,
    forbidden_hits,
    ndcg_at_k,
    negation_purity,
    percentile,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


class TestRecall:
    def test_усі_релевантні_у_топі(self):
        assert recall_at_k(["a", "b", "c"], {"a", "b"}, 3) == 1.0

    def test_половина_релевантних(self):
        assert recall_at_k(["a", "x", "y"], {"a", "b"}, 3) == 0.5

    def test_за_межами_вікна_не_рахується(self):
        assert recall_at_k(["x", "y", "a"], {"a"}, 2) == 0.0

    def test_порожня_множина_релевантних_дає_одиницю(self):
        # «Нічого не мало знайтися — нічого й не пропустили».
        # Інакше заперечні сценарії ділили б на нуль.
        assert recall_at_k(["x"], set(), 1) == 1.0

    def test_недодатний_k_це_помилка(self):
        with pytest.raises(ValueError):
            recall_at_k(["a"], {"a"}, 0)


class TestPrecision:
    def test_точність_у_вікні(self):
        assert precision_at_k(["a", "x", "b", "y"], {"a", "b"}, 4) == 0.5

    def test_видача_коротша_за_вікно(self):
        assert precision_at_k(["a"], {"a"}, 10) == 1.0


class TestAveragePrecision:
    def test_ідеальний_порядок(self):
        assert average_precision(["a", "b", "x"], {"a", "b"}) == 1.0

    def test_ранній_релевантний_цінніший(self):
        early = average_precision(["a", "x", "y"], {"a"})
        late = average_precision(["x", "y", "a"], {"a"})
        assert early > late


class TestReciprocalRank:
    def test_позиція_першого_влучання(self):
        assert reciprocal_rank(["x", "a"], {"a"}) == 0.5

    def test_немає_влучань(self):
        assert reciprocal_rank(["x", "y"], {"a"}) == 0.0


class TestNDCG:
    def test_ідеальний_порядок_дає_одиницю(self):
        gains = {"a": 3.0, "b": 2.0, "c": 1.0}
        assert ndcg_at_k(["a", "b", "c"], gains, 3) == pytest.approx(1.0)

    def test_зворотний_порядок_гірший(self):
        gains = {"a": 3.0, "b": 2.0, "c": 1.0}
        assert ndcg_at_k(["c", "b", "a"], gains, 3) < ndcg_at_k(["a", "b", "c"], gains, 3)

    def test_градації_розрізняються(self):
        # Кадр із чітко видимими окулярами і кадр, де вони ледь помітні,
        # обидва релевантні — але не однаково.
        gains = {"clear": 3.0, "faint": 1.0}
        assert ndcg_at_k(["clear", "faint"], gains, 2) > ndcg_at_k(["faint", "clear"], gains, 2)


class TestNegation:
    """Головна метрика п.11: важливо не що знайшлося, а що НЕ мало знайтися."""

    def test_заборонені_у_топі_перелічуються(self):
        assert forbidden_hits(["a", "bad", "c"], {"bad"}, 3) == ["bad"]

    def test_заборонені_поза_вікном_не_рахуються(self):
        assert forbidden_hits(["a", "b", "bad"], {"bad"}, 2) == []

    def test_чистота_без_порушень(self):
        assert negation_purity(["a", "b"], {"bad"}, 2) == 1.0

    def test_чистота_з_одним_порушенням(self):
        assert negation_purity(["a", "bad"], {"bad"}, 2) == 0.5


class TestPercentile:
    def test_медіана(self):
        assert percentile([1.0, 2.0, 3.0], 0.5) == 2.0

    def test_інтерполяція(self):
        assert percentile([0.0, 10.0], 0.95) == pytest.approx(9.5)

    def test_один_елемент(self):
        assert percentile([42.0], 0.95) == 42.0

    def test_порожня_вибірка_це_помилка(self):
        with pytest.raises(ValueError):
            percentile([], 0.5)
