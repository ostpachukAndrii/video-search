"""Метрики якості пошуку.

Свідомо на чистому Python без numpy: ці функції викликаються з крокових
визначень BDD, і вони мають працювати навіть у мінімальному оточенні M0, куди
важкі залежності ще не приїхали.

Домовленість про типи:
  * `ranked`    — список id результатів у порядку спадання релевантності;
  * `relevant`  — множина id, які вважаються правильними;
  * `gains`     — {id: градуйована релевантність} для nDCG (0 = нерелевантний).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from math import log2

__all__ = [
    "recall_at_k",
    "precision_at_k",
    "average_precision",
    "mean_average_precision",
    "reciprocal_rank",
    "ndcg_at_k",
    "forbidden_hits",
    "negation_purity",
    "binding_inverted",
    "binding_clean_rate",
    "percentile",
]


def _top(ranked: Sequence[str], k: int) -> Sequence[str]:
    if k <= 0:
        raise ValueError("k має бути додатним")
    return ranked[:k]


def recall_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Яка частка правильних відповідей потрапила в топ-k.

    Порожня множина релевантних повертає 1.0: «нічого не мало знайтися, нічого й
    не пропустили». Інакше сценарії із заперечним очікуванням давали б 0/0.
    """
    relevant_set = set(relevant)
    if not relevant_set:
        return 1.0
    found = sum(1 for item in _top(ranked, k) if item in relevant_set)
    return found / len(relevant_set)


def precision_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    relevant_set = set(relevant)
    top = _top(ranked, k)
    if not top:
        return 0.0
    return sum(1 for item in top if item in relevant_set) / len(top)


def average_precision(ranked: Sequence[str], relevant: Iterable[str]) -> float:
    """AP по всій видачі: середня точність у точках влучання."""
    relevant_set = set(relevant)
    if not relevant_set:
        return 1.0
    hits = 0
    total = 0.0
    for position, item in enumerate(ranked, start=1):
        if item in relevant_set:
            hits += 1
            total += hits / position
    return total / len(relevant_set)


def mean_average_precision(
    runs: Iterable[tuple[Sequence[str], Iterable[str]]],
) -> float:
    scores = [average_precision(ranked, relevant) for ranked, relevant in runs]
    return sum(scores) / len(scores) if scores else 0.0


def reciprocal_rank(ranked: Sequence[str], relevant: Iterable[str]) -> float:
    """1/позиція першого влучання. Метрика «чи достатньо глянути на перший екран»."""
    relevant_set = set(relevant)
    for position, item in enumerate(ranked, start=1):
        if item in relevant_set:
            return 1.0 / position
    return 0.0


def ndcg_at_k(ranked: Sequence[str], gains: Mapping[str, float], k: int) -> float:
    """nDCG з градуйованою релевантністю.

    Потрібен там, де «правильно/неправильно» замало: наприклад, кадр із чітко
    видимими окулярами і кадр, де вони ледь помітні, — обидва релевантні, але не
    однаково.
    """
    top = _top(ranked, k)
    dcg = sum(gains.get(item, 0.0) / log2(rank + 1) for rank, item in enumerate(top, start=1))
    ideal_gains = sorted((g for g in gains.values() if g > 0), reverse=True)[:k]
    idcg = sum(gain / log2(rank + 1) for rank, gain in enumerate(ideal_gains, start=1))
    return dcg / idcg if idcg else 0.0


def forbidden_hits(ranked: Sequence[str], forbidden: Iterable[str], k: int) -> list[str]:
    """Які заборонені елементи пролізли в топ-k.

    Ключова метрика для п.11: заперечення перевіряється не тим, що знайшлося,
    а тим, що НЕ мало знайтися й усе одно знайшлося.
    """
    forbidden_set = set(forbidden)
    return [item for item in _top(ranked, k) if item in forbidden_set]


def negation_purity(ranked: Sequence[str], forbidden: Iterable[str], k: int) -> float:
    """Частка топ-k, вільна від заборонених елементів. 1.0 = заперечення спрацювало."""
    top = _top(ranked, k)
    if not top:
        return 1.0
    return 1.0 - len(forbidden_hits(ranked, forbidden, k)) / len(top)


def binding_inverted(
    ranked: Sequence[str],
    relevant: Iterable[str],
    forbidden: Iterable[str],
) -> bool:
    """Чи стоїть хоч одна «пастка звʼязування» вище за релевантний результат.

    Чому не `negation_purity`: у пастці звʼязування присутні ВСІ ознаки запиту,
    просто на різних обʼєктах. Кадр із синім колом і червоним квадратом для
    запиту «червоне коло» справді ближчий за кадр із зеленим трикутником, тож
    його поява в хвості видачі коректна. Помилкою є інше — коли він
    ОБГАНЯЄ справжнє червоне коло.

    Тому метрика бінарна й порядкова: перевіряється не склад видачі, а те,
    чи не переставлено пастку поперед правильної відповіді.
    """
    position = {item: index for index, item in enumerate(ranked)}
    missing = len(ranked)
    worst_relevant = max((position.get(item, missing) for item in relevant), default=missing)
    best_trap = min((position.get(item, missing) for item in forbidden), default=missing)
    return best_trap <= worst_relevant


def binding_clean_rate(
    runs: Iterable[tuple[Sequence[str], Iterable[str], Iterable[str]]],
) -> float:
    """Частка запитів, де жодна пастка не обігнала релевантний результат."""
    outcomes = [
        not binding_inverted(ranked, relevant, forbidden)
        for ranked, relevant, forbidden in runs
    ]
    return sum(outcomes) / len(outcomes) if outcomes else 1.0


def percentile(values: Sequence[float], q: float) -> float:
    """Перцентиль для замірів затримки (p50/p95) з лінійною інтерполяцією."""
    if not values:
        raise ValueError("порожня вибірка")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q має бути в межах [0, 1]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight
