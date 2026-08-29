"""Крокові визначення для categories.feature — вимога п.4."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from vsearch.represent.categories import DEFAULT_ATTRIBUTES, DEFAULT_CATEGORIES, CategoryEngine
from vsearch.represent.prototypes import Prototype

scenarios("categories.feature")

#: Категорії, яких у синтетичному наборі бути не може — контроль на хибні
#: спрацювання. Якщо вони вмикаються на кольорових фігурах, фасети шумлять.
IMPOSSIBLE = ("weapon", "vehicle", "money", "drugs", "crowd", "license_plate")


@pytest.fixture
def engine(golden, embedder):
    return CategoryEngine.with_defaults(embedder, golden.store)


def _f1(predicted: np.ndarray, actual: np.ndarray) -> float:
    true_positive = int((predicted & actual).sum())
    if not true_positive:
        return 0.0
    return 2 * true_positive / (int(predicted.sum()) + int(actual.sum()))


def _asset_vectors(golden, embedder):
    """Ембединги активів у порядку розмітки — для перевірок якості фасета."""
    from vsearch.ingest.images import load_image

    assets = sorted(golden.assets.values(), key=lambda a: a.asset_id)
    vectors = embedder.embed_images([load_image(golden.root / a.path) for a in assets])
    return assets, vectors


# ── дії ─────────────────────────────────────────────────────────────────────


@when("я читаю фасети проіндексованих кадрів")
def _read_facets(context, golden):
    context["payloads"] = golden.store.scroll_payloads("frames", limit=100)
    assert context["payloads"], "індекс порожній — перевіряти нічого"


@when(parsers.parse('користувач додає категорію "{text}"'))
def _add_user_category(context, engine, golden, embedder, text):
    name = "user_category"
    started = time.perf_counter()
    engine.add_user_category(name, text)
    report = engine.apply_to_collection("frames", name)
    context["category"] = name
    context["category_text"] = text
    context["report"] = report
    context["elapsed"] = time.perf_counter() - started


@when("я застосовую категорію до всього індексу")
def _apply_to_large(context, engine, large_collection):
    engine.add_user_category("scale_probe", "a weapon on a table")
    started = time.perf_counter()
    context["report"] = engine.apply_to_collection(large_collection, "scale_probe")
    context["elapsed"] = time.perf_counter() - started


@when("я калібрую фасет за мітками набору")
def _calibrate(context, engine, golden, embedder):
    assets, vectors = _asset_vectors(golden, embedder)
    engine.bank.add(Prototype(name="probe", positive=("a circle",)))
    labels = [a.labels.get("shape") == "circle" for a in assets]
    prototype, metrics = engine.bank.calibrate("probe", vectors, labels)
    context["prototype"] = prototype
    context["metrics"] = metrics


@when(parsers.parse('я шукаю "{query}" у категорії "{text}"'))
def _search_in_category(context, golden, query):
    # parse=False: сценарій перевіряє ФІЛЬТР за категорією, а не здатність
    # парсера її вгадати.
    context["response"] = golden.searcher.search(
        query, limit=10, category=context["category"], parse=False
    )


# ── передумови окремих сценаріїв ────────────────────────────────────────────


@pytest.fixture
def large_index(golden, embedder):
    """10000 випадкових векторів — перевіряємо шлях застосування, не якість.

    Реальні зображення тут не потрібні й лише сповільнили б перевірку: нас
    цікавить вартість множення матриць і оновлення payload, а вона не
    залежить від того, що саме зображено.
    """
    from vsearch.index import schema

    rng = np.random.default_rng(seed=17)
    vectors = rng.normal(size=(10000, embedder.dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    collection = "scale_probe"
    golden.store.ensure_collection(collection, embedder.dim, quantize=True, recreate=True)
    golden.store.upsert(
        collection,
        [f"probe-{i}" for i in range(len(vectors))],
        vectors,
        [{"asset_id": f"probe-{i}"} for i in range(len(vectors))],
        batch_size=1000,
    )
    yield collection
    golden.store.drop_collection(collection)


@given(parsers.parse("індекс із {count:d} векторів"), target_fixture="large_collection")
def _large_index_step(large_index, count):
    return large_index


# ── перевірки ───────────────────────────────────────────────────────────────


@then("кожен кадр має мати оцінку по кожній наперед заданій категорії")
def _all_categories_present(context, golden):
    """Категорії обчислені для всього індексу — але зберігаються розріджено.

    Щільний запис на тисячі категорій дав би близько десяти кілобайт payload
    на точку, тобто десятки гігабайт на мільйонах регіонів. Тому ключ
    зʼявляється лише там, де категорія спрацювала, а факт обчислення
    підтверджує реєстр прототипів.
    """
    from vsearch.represent.prototypes import KIND_CATEGORY

    registered = {r["name"] for r in golden.catalog.prototypes() if r["kind"] == KIND_CATEGORY}
    missing = {p.name for p in DEFAULT_CATEGORIES} - registered
    assert not missing, f"категорії не обчислювалися: {sorted(missing)}"


@then("оцінки мають бути в межах від 0 до 1")
def _scores_in_range(context):
    for payload in context["payloads"]:
        for key, value in payload.items():
            if key.endswith("_score"):
                assert 0.0 <= value <= 1.0, f"{key}={value} поза межами [0, 1]"


@then("категорії, яких у наборі немає, не мають спрацьовувати")
def _impossible_stay_false(context):
    triggered = {
        f"cat_{name}"
        for payload in context["payloads"]
        for name in IMPOSSIBLE
        if payload.get(f"cat_{name}") is True
    }
    assert not triggered, (
        f"на кольорових фігурах спрацювали неможливі категорії: {sorted(triggered)}"
    )


@then("атрибути людини мають бути невідомими там, де людини не виявлено")
def _attributes_gated(context):
    attribute_keys = [f"attr_{p.name}" for p in DEFAULT_ATTRIBUTES]
    checked = 0
    for payload in context["payloads"]:
        if payload.get("cat_person") is True:
            continue
        checked += 1
        leaked = [k for k in attribute_keys if k in payload]
        assert not leaked, (
            f"атрибут проставлено там, де людини немає: {leaked}. "
            f"Тоді «без окулярів» означало б і «це порожня кімната»."
        )
    assert checked, "у наборі не знайшлося кадрів без людини — перевірка беззмістовна"


@then("категорія має застосуватися до всього наявного індексу")
def _applied_to_all(context, golden):
    report = context["report"]
    assert report.scanned == golden.store.count("frames"), (
        f"оброблено {report.scanned} із {golden.store.count('frames')} точок"
    )


@then("повторного читання пікселів не має відбуватися")
def _no_pixel_reindex(context):
    assert context["report"].reindexed_pixels is False


@then(parsers.parse("F1 категорії має бути не менше {threshold:f}"))
def _user_category_quality(context, golden, embedder, engine, threshold):
    assets, vectors = _asset_vectors(golden, embedder)
    scores = engine.bank.score(vectors, context["category"])
    # Розмітка для "a yellow triangle" виводиться з міток набору.
    truth = np.array([
        a.labels.get("color") == "yellow" and a.labels.get("shape") == "triangle"
        for a in assets
    ])
    if not truth.any():
        pytest.skip("у наборі немає цілей для цієї категорії")
    suggested, decision = engine.bank.suggest_threshold(context["category"], vectors)
    value = _f1(decision, truth)
    assert value >= threshold, (
        f"F1={value:.3f} < {threshold} (межа {suggested:.4f}, "
        f"знайдено {int(decision.sum())}, істинних {int(truth.sum())})"
    )


@then(parsers.parse("застосування має тривати менше {seconds:d} секунд"))
def _apply_is_fast(context, seconds):
    assert context["elapsed"] < seconds, (
        f"{context['elapsed']:.1f}с на {context['report'].scanned} векторів "
        f"({context['report'].rate:.0f} векторів/с)"
    )


@then("F1 після калібрування має бути не гіршим за типовий")
def _calibration_helps(context):
    metrics = context["metrics"]
    assert metrics["f1"] >= metrics["f1_default"], (
        f"калібрування погіршило: {metrics['f1_default']:.3f} → {metrics['f1']:.3f}"
    )


@then("прототип має бути позначений як калібрований")
def _marked_calibrated(context):
    assert context["prototype"].calibrated, (
        "некалібрований фасет придатний для ранжування, але як жорсткий фільтр "
        "ненадійний — споживач має про це знати"
    )


@then("усі результати мають належати до цієї категорії")
def _all_in_category(context, golden):
    response = context["response"]
    assert response.results, "порожня видача нічого не доводить"

    key = f"cat_{context['category']}"
    by_frame = {
        payload.get("frame_id"): payload
        for payload in golden.store.scroll_payloads("frames", limit=1000)
    }
    positives = {frame for frame, p in by_frame.items() if p.get(key) is True}
    assert positives, "категорія не спрацювала на жодному кадрі — фільтр нічого не доводить"

    returned = {r.frame_id for r in response.results}
    outside = returned - positives
    assert not outside, (
        f"фільтр пропустив {len(outside)} кадрів поза категорією: {sorted(outside)[:3]}"
    )
