"""Крокові визначення для semantic.feature і multilingual.feature.

Сценарії керуються даними: перелік запитів та очікування беруться із золотого
набору. Тому поповнення набору реальними матеріалами не потребує правок ані
тут, ані у feature-файлах.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pytest_bdd import parsers, scenarios, then, when

from eval.metrics import negation_purity, recall_at_k, reciprocal_rank

scenarios("semantic.feature")
scenarios("multilingual.feature")

#: Скільки результатів тягнемо: вистачає і на Recall@10, і на чистоту топ-5.
FETCH_LIMIT = 10


def _file_to_asset(golden) -> dict[str, str]:
    """Імʼя файлу → asset_id із розмітки.

    Конвеєр ідентифікує активи за sha256 вмісту, а розмітка — своїми
    ідентифікаторами. Спільне в них — імʼя файлу.
    """
    return {Path(a.path).name: a.asset_id for a in golden.assets.values()}


def _run(searcher, golden, queries) -> list[dict]:
    mapping = _file_to_asset(golden)
    runs = []
    for query in queries:
        # parse=False: ці сценарії міряють ЩІЛЬНИЙ пошук і його метрики.
        # Розбір перевіряється окремо в negation.feature, і змішувати їх
        # означало б не знати, що саме зламалося, коли метрика просяде.
        response = searcher.search(query.text, limit=FETCH_LIMIT, parse=False)
        ranked = []
        for result in response.results:
            asset = mapping.get(Path(result.path).name)
            if asset and asset not in ranked:
                ranked.append(asset)
        runs.append({"query": query, "ranked": ranked, "response": response})
    return runs


# ── дії ─────────────────────────────────────────────────────────────────────


@when("я виконую всі запити набору")
def _run_all(context, searcher, golden):
    context["runs"] = _run(searcher, golden, golden.queries)


@when(parsers.parse('я виконую запити мовою "{lang}"'))
def _run_lang(context, searcher, golden, lang):
    queries = [q for q in golden.queries if q.lang == lang]
    if not queries:
        pytest.skip(f"у наборі немає запитів мовою {lang!r}")
    context["runs"] = _run(searcher, golden, queries)


@when("я виконую перший запит набору")
def _run_first(context, searcher, golden):
    context["runs"] = _run(searcher, golden, golden.queries[:1])


@when("я намагаюся працювати з індексом на інших параметрах")
def _incompatible_signature(context, golden):
    from vsearch.config import get_profile
    from vsearch.index.catalog import SignatureMismatch

    # Каталог саме цього набору вже містить підпис, з яким його індексували,
    # тому звіряння з іншою роздільністю мусить впасти.
    other = dict(get_profile("balanced").index_signature())
    other["max_num_patches"] = other["max_num_patches"] * 2
    with pytest.raises(SignatureMismatch) as excinfo:
        golden.catalog.assert_compatible(other)
    context["error"] = str(excinfo.value)


# ── перевірки ───────────────────────────────────────────────────────────────


def _answered(run) -> bool:
    """Чи система взагалі стверджує, що щось знайшла.

    Межа не вигадана під тест: `is_confident` — це та сама калібрована межа,
    за якою UI вирішує, показувати результат чи ні. Якщо жоден кандидат її не
    перетнув, відповідь системи — «нічого немає», і питати про порядок такої
    видачі немає сенсу.
    """
    return any(result.is_confident for result in run["response"].results)


def _unanswered(runs) -> list[str]:
    return [
        f"{r['query'].query_id} ({r['query'].lang}) «{r['query'].text}»: "
        f"кращий {max((x.probability for x in r['response'].results), default=0.0):.2%}"
        for r in runs
        if r["query"].relevant and not _answered(r)
    ]


@then(parsers.parse("Recall@{k:d} має бути не менше {threshold:f} для кожного запиту"))
def _recall_each(context, k, threshold):
    failures = []
    for run in context["runs"]:
        query = run["query"]
        if not query.relevant:
            continue
        value = recall_at_k(run["ranked"], query.relevant, k)
        if value < threshold:
            failures.append(f"{query.query_id} ({query.lang}): Recall@{k}={value:.2f}")
    assert not failures, "нижче порога:\n  " + "\n  ".join(failures)


@then(parsers.parse("середній Recall@{k:d} має бути не менше {threshold:f}"))
def _mean_recall(context, k, threshold):
    """Пара до поодинокої межі: та ловить провал одного запиту, ця — усіх разом.

    Без агрегату набір міг би тихо просісти з 1.00 до 0.71 на кожному запиті
    й лишитися зеленим.
    """
    scores = [
        recall_at_k(run["ranked"], run["query"].relevant, k)
        for run in context["runs"]
        if run["query"].relevant
    ]
    mean = sum(scores) / len(scores) if scores else 0.0
    assert mean >= threshold, (
        f"середній Recall@{k} {mean:.3f} нижче за {threshold}; "
        f"найгірші: " + ", ".join(
            f"{r['query'].query_id}={recall_at_k(r['ranked'], r['query'].relevant, k):.2f}"
            for r in sorted(
                (r for r in context["runs"] if r["query"].relevant),
                key=lambda r: recall_at_k(r["ranked"], r["query"].relevant, k),
            )[:3]
        )
    )


@then(parsers.parse("без впевненої відповіді має лишитися не більше {limit:d} запиту"))
@then(parsers.parse("без впевненої відповіді має лишитися не більше {limit:d} запитів"))
def _unanswered_bounded(context, limit):
    """Друга половина послаблення вище, без якої воно було б дірою.

    Пропускати метрики для безрезультатних запитів безпечно рівно доти, доки
    таких запитів одиниці. Якщо їх стане багато, попередня перевірка мовчки
    перестала б міряти будь-що — саме той тихий провал, що коштував найдорожче.
    """
    missing = _unanswered(context["runs"])
    assert len(missing) <= limit, (
        f"система не дала впевненої відповіді на {len(missing)} запитів "
        f"(дозволено {limit}):\n  " + "\n  ".join(missing)
    )


@then(parsers.parse("середній MRR має бути не менше {threshold:f}"))
def _mean_mrr(context, threshold):
    scores = [
        reciprocal_rank(run["ranked"], run["query"].relevant)
        for run in context["runs"]
        if run["query"].relevant
    ]
    mean = sum(scores) / len(scores) if scores else 0.0
    assert mean >= threshold, f"середній MRR={mean:.3f} < {threshold}"


@then(parsers.parse("середня чистота топ-{k:d} має бути не менше {threshold:f}"))
def _mean_purity(context, k, threshold):
    scores = [
        negation_purity(run["ranked"], run["query"].forbidden, k)
        for run in context["runs"]
        if run["query"].forbidden
    ]
    if not scores:
        pytest.skip("у наборі немає запитів із забороненими активами")
    mean = sum(scores) / len(scores)
    assert mean >= threshold, f"середня чистота топ-{k}={mean:.3f} < {threshold}"


@then(parsers.parse("Recall@{k:d} для кожної мови має бути не менше {threshold:f}"))
def _recall_per_language(context, k, threshold):
    by_lang = _recall_by_language(context, k)
    low = {lang: value for lang, value in by_lang.items() if value < threshold}
    assert not low, f"мови нижче порога: {low} (усі: {by_lang})"


@then(parsers.parse("розрив між найкращою і найгіршою мовою не має перевищувати {gap:f}"))
def _language_gap(context, gap):
    by_lang = _recall_by_language(context, FETCH_LIMIT)
    if len(by_lang) < 2:
        pytest.skip("для порівняння мов потрібно щонайменше дві")
    spread = max(by_lang.values()) - min(by_lang.values())
    assert spread <= gap, f"розрив {spread:.3f} > {gap}; за мовами: {by_lang}"


def _recall_by_language(context, k) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for run in context["runs"]:
        query = run["query"]
        if not query.relevant:
            continue
        buckets.setdefault(query.lang, []).append(recall_at_k(run["ranked"], query.relevant, k))
    return {lang: round(sum(v) / len(v), 3) for lang, v in buckets.items()}


@then("кожен результат має містити оцінку схожості")
def _has_score(context):
    for run in context["runs"]:
        assert run["response"].results, "порожня видача нічого не доводить"
        for result in run["response"].results:
            assert isinstance(result.score, float)
            assert "впевненість" in result.explain()


@then("кожен результат має містити ідентифікатор кадру та джерела")
def _has_identifiers(context):
    for run in context["runs"]:
        for result in run["response"].results:
            assert result.asset_id, "немає asset_id — результат не привʼязати до джерела"
            assert result.frame_id, "немає frame_id"
            assert result.path, "немає шляху до файлу"


@then("кожен результат має містити версії моделей, якими його отримано")
def _has_provenance(context):
    for run in context["runs"]:
        for result in run["response"].results:
            provenance = result.provenance
            assert provenance.get("embed_model"), "немає назви моделі"
            assert len(provenance.get("embed_revision", "")) == 40, (
                "ревізія має бути 40-символьним commit SHA — інакше результат "
                "неможливо відтворити"
            )
            assert provenance.get("max_num_patches"), "немає роздільності"


@then("має бути помилка про несумісність із вказівкою переіндексувати")
def _mismatch_explained(context):
    error = context["error"]
    assert "іншому просторі" in error
    assert "--recreate" in error, "помилка має підказувати конкретну команду"
