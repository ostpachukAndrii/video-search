"""Крокові визначення для real_photos.feature.

Набір посилається на особисті знімки поза репозиторієм. Якщо їх немає,
сценарії ПРОПУСКАЮТЬСЯ з поясненням: набір із чужими фото не має ламати
збірку, але й мовчки зникати він не повинен.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from eval.metrics import recall_at_k

scenarios("real_photos.feature")

#: Показуємо десять, але шукаємо глибше — і перевіряємо саме глибину.
FETCH_LIMIT = 10
#: Запити, чия відповідь тримається на точному рядку, а не на схожості сцени.
LEXICAL_QUERIES = ("rp_ticket_exact", "rp_augustiner_exact", "rp_encrypted_exact")


@given("проіндексовано реальні фото з розміткою", target_fixture="real_set")
def _real_set():
    from vsearch import goldenset
    from vsearch.search.retrieve import Searcher

    try:
        golden = goldenset.load("real_photos")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"немає набору real_photos: {exc}")
    missing = golden.missing_files()
    if missing:
        pytest.skip(
            f"немає {len(missing)} медіафайлів набору real_photos "
            f"(особисті знімки лежать поза репозиторієм)"
        )
    return {
        "golden": golden,
        "searcher": Searcher(),
        "mapping": {Path(a.path).name: a.asset_id for a in golden.assets.values()},
    }


def _run(real_set, queries):
    """Позиція рахується серед УСІХ результатів, а не лише серед розмічених.

    Це не дрібниця. Розмічено 12 знімків зі 116, тож фільтрація видачі до
    відомих активів викидала 104 фото з рахунку — і ціль, що стоїть насправді
    сьомою, виглядала першою. Метрика показувала Recall@5 = 1.00 там, де в
    інтерфейсі результат був помітно нижчий, і саме таку розбіжність найлегше
    не помітити: числа гарні, а користувач бачить інше.

    Нерозмічені кадри лишаються в переліку як `None` — вони займають місця,
    як і в справжній видачі.
    """
    runs = []
    for query in queries:
        response = real_set["searcher"].search(query.text, limit=FETCH_LIMIT)
        ranked: list[str | None] = []
        for result in response.results:
            ranked.append(real_set["mapping"].get(Path(result.path).name))
        runs.append({"query": query, "ranked": ranked, "response": response})
    return runs


@when("я виконую всі запити набору реальних фото")
def _run_all(real_set, context):
    context["runs"] = _run(real_set, real_set["golden"].queries)


@when("я виконую текстові запити набору реальних фото")
def _run_lexical(real_set, context):
    queries = [q for q in real_set["golden"].queries if q.query_id in LEXICAL_QUERIES]
    assert queries, "у наборі немає текстових запитів — сценарій нічого не міряє"
    context["runs"] = _run(real_set, queries)


@then(parsers.parse("середній Recall@{k:d} має бути не менше {threshold:f}"))
def _mean_recall(context, k, threshold):
    scores = [
        recall_at_k(r["ranked"], r["query"].relevant, k) for r in context["runs"]
    ]
    mean = sum(scores) / len(scores)
    worst = sorted(
        context["runs"],
        key=lambda r: recall_at_k(r["ranked"], r["query"].relevant, k),
    )[:3]
    assert mean >= threshold, (
        f"середній Recall@{k} {mean:.2f} нижче за {threshold}; найгірші: "
        + ", ".join(
            f"{r['query'].query_id}={recall_at_k(r['ranked'], r['query'].relevant, k):.2f}"
            for r in worst
        )
    )


@then("кожен запит має знайти свою ціль у межах глибини пошуку")
def _all_found(real_set, context):
    """Чи система взагалі дістала потрібний кадр.

    Міряється глибина ПОШУКУ, а не кількість показаних результатів: скільки
    показувати — рішення інтерфейсу. Вимога «у топ-10» на цьому наборі стала б
    межею в одну позицію: «чорна жіноча сумочка» дає 11, а «a handbag» — 4.
    Це мовний розрив у заземленні слова, а не поломка ранжування, і міряти
    його цією перевіркою неправильно — для якості є агрегат Recall@5.
    """
    from vsearch.search.retrieve import RESCORE_DEPTH

    deep = []
    for run in context["runs"]:
        response = real_set["searcher"].search(run["query"].text, limit=RESCORE_DEPTH)
        found = {
            real_set["mapping"].get(Path(r.path).name) for r in response.results
        }
        if not (found & set(run["query"].relevant)):
            deep.append(run["query"].query_id)
    assert not deep, (
        f"ціль не знайдена навіть на глибині {RESCORE_DEPTH}: {deep}"
    )


@then("впевненість має спадати від першого результату до останнього")
def _monotonic(context):
    """Домінування за обома вимірами (див. `invariants.feature`).

    Порядок зливає схожість із текстом і підтвердження сутностей, тож жодне
    одне число не спадає рівно. Перевіряється сильніше: кадр не стоїть вище
    за той, що кращий за ОБОМА.
    """
    eps = 1e-9
    failures = []
    for run in context["runs"]:
        rows = [
            (r.probability, r.entity_confidence if r.entity_confidence is not None else 1.0)
            for r in run["response"].results
        ]
        bad = [
            (i + 1, i + 2)
            for i, (hi, lo) in enumerate(zip(rows, rows[1:]))
            if lo[0] > hi[0] + eps and lo[1] > hi[1] + eps
        ]
        if bad:
            failures.append(f"{run['query'].query_id}: позиції {bad[:3]}")
    assert not failures, (
        "результат стоїть вище за той, що перевершує його за обома вимірами:"
        "\n  " + "\n  ".join(failures)
    )


@then("кожен має знайти свій кадр на першому місці")
def _lexical_first(context):
    failures = [
        f"{r['query'].query_id}: перший {r['ranked'][0] if r['ranked'] else '—'}, "
        f"треба {sorted(r['query'].relevant)}"
        for r in context["runs"]
        if not r["ranked"] or r["ranked"][0] not in r["query"].relevant
    ]
    assert not failures, "точний рядок не вивів кадр на перше місце:\n  " + "\n  ".join(
        failures
    )


@then("результат має пояснювати, які саме слова збіглися")
def _lexical_explained(context):
    for run in context["runs"]:
        top = run["response"].results[0]
        matched = top.matched_attrs.get("ocr_match")
        assert matched, (
            f"{run['query'].query_id}: збіг стався, але не видно, ЯКІ слова "
            f"знайдено. Для матеріалів справи такий результат непридатний."
        )


@then(parsers.parse("пасток у топ-5 має бути не більше {limit:d} на запит"))
def _traps_bounded(context, limit):
    failures = []
    for run in context["runs"]:
        traps = [a for a in run["ranked"][:5] if a in run["query"].forbidden]
        if len(traps) > limit:
            failures.append(f"{run['query'].query_id}: {traps}")
    assert not failures, (
        f"пасток у топ-5 більше за {limit}:\n  " + "\n  ".join(failures)
    )


@when(parsers.parse('я шукаю "{query}"'))
def _search_one(real_set, context, query):
    context["single"] = _run(real_set, [
        q for q in real_set["golden"].queries if q.text == query
    ])
    assert context["single"], f"у наборі немає запиту {query!r}"


@then("перші два результати мають бути обидва кадри з полуницею")
def _both_strawberries_first(context):
    run = context["single"][0]
    expected = set(run["query"].relevant)
    assert len(expected) == 2, (
        f"сценарій написано на два релевантні кадри, а їх {len(expected)}"
    )
    top = run["ranked"][:2]
    missing = expected - set(top)
    assert not missing, (
        f"у топ-2 бракує {sorted(missing)}; там стоїть {top}. "
        f"Повний порядок: {run['ranked'][:6]}"
    )
