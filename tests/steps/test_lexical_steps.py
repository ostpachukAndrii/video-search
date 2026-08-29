"""Крокові визначення для lexical.feature — лексичний шар (M4f)."""

from __future__ import annotations

import numpy as np
import pytest
from pytest_bdd import given, scenarios, then, when

from vsearch.index import schema
from vsearch.index.store import VectorStore
from vsearch.represent import lexical

scenarios("lexical.feature")

#: Написи навмисно такі, які щільний ембединг НЕ розрізняє: два номери
#: відрізняються лише символами, і візуально це та сама сцена.
CAPTIONS = {
    "plate_a": "автомобіль з номером AA1234BB біля будинку",
    "plate_b": "автомобіль з номером BC7788KK на стоянці",
    "sign": "Augustiner-Brau Munchen пивна вивіска",
    "plain": "краєвид без жодного напису",
}
TARGET = "plate_a"
EXACT = "AA1234BB"
#: Та сама послідовність із однією поплутаною літерою: 2→Z, класична помилка OCR.
MISREAD = "AA1Z34BB"
DIM = 8


@pytest.fixture
def lexical_store():
    store = VectorStore(prefix="lex_")
    yield store
    store.drop_collection(schema.FRAMES)


@given("проіндексовано кадри з написами", target_fixture="context")
def _index_captions(lexical_store):
    """Щільні вектори навмисно ведуть НЕ туди.

    Кадр без напису отримує вектор, найближчий до запиту, а потрібний —
    найдальший. Так перевіряється саме лексичний шар: якби щільний і без
    нього давав правильну відповідь, сценарій нічого не доводив би.
    """
    lexical_store.ensure_collection(
        schema.FRAMES, DIM, quantize=False, recreate=True, sparse=True
    )
    keys = list(CAPTIONS)
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(len(keys), DIM)).astype("float32")
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    lexical_store.upsert(
        schema.FRAMES,
        keys,
        vectors,
        [
            {
                "frame_id": key,
                "path": f"{key}.jpg",
                "ocr_text": CAPTIONS[key],
                "ocr_engine": "florence2",
            }
            for key in keys
        ],
        sparse=[lexical.build(CAPTIONS[key]) for key in keys],
    )
    # Запит-вектор навмисно вказує на кадр БЕЗ напису.
    return {"store": lexical_store, "dense": vectors[keys.index("plain")]}


def _run(context, text: str):
    store, dense = context["store"], context["dense"]
    sparse = lexical.build_query(text)
    context["dense_only"] = store.search(schema.FRAMES, dense, limit=4)
    context["hybrid"] = store.search_hybrid(schema.FRAMES, dense, sparse, limit=4)


@given("проіндексовано кадри з описами", target_fixture="context")
def _index_with_captions(lexical_store):
    """Те саме сховище, але слово живе лише в ОПИСІ, не в написі.

    Так перевіряється саме внесок підпису: якби слово було і в написі, тест
    зеленів би й без нього.
    """
    lexical_store.ensure_collection(
        schema.FRAMES, DIM, quantize=False, recreate=True, sparse=True
    )
    keys = list(CAPTIONS)
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(len(keys), DIM)).astype("float32")
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    payloads, sparse = [], []
    for key in keys:
        # Напис порожній — усе слово тільки в описі.
        caption = CAPTIONS[key]
        payloads.append({
            "frame_id": key, "path": f"{key}.jpg",
            "ocr_text": "", "caption": caption, "ocr_engine": "florence2",
        })
        sparse.append(lexical.build(caption))
    lexical_store.upsert(schema.FRAMES, keys, vectors, payloads, sparse=sparse)
    return {"store": lexical_store, "dense": vectors[keys.index("plain")]}


#: Слово живе лише в описі однієї сцени й у жодному написі.
CAPTION_ONLY_TERM = "Munchen"
CAPTION_ONLY_TARGET = "sign"


@when("я шукаю слово, якого немає в написі, але є в описі")
def _search_caption_only(context):
    context["expected"] = CAPTION_ONLY_TARGET
    _run(context, CAPTION_ONLY_TERM)


@when("я шукаю точний рядок з напису")
def _search_exact(context):
    _run(context, EXACT)


@when("я шукаю рядок з однією помилково прочитаною літерою")
def _search_misread(context):
    _run(context, MISREAD)


@when("я шукаю запит, якого немає в жодному написі")
def _search_absent(context):
    _run(context, "гелікоптер над морем")


@given("рушій OCR не покриває іврит", target_fixture="context")
def _engine_without_hebrew():
    from vsearch.represent.ocr import Florence2Ocr, OcrResult

    return {
        "result": OcrResult("", "florence2", unsupported=Florence2Ocr.UNSUPPORTED)
    }


@then("потрібний кадр має бути першим")
def _target_first(context):
    top = context["hybrid"][0].payload["frame_id"]
    expected = context.get("expected", TARGET)
    assert top == expected, (
        f"перший результат {top!r}, а не {expected!r}: лексичний шар не спрацював. "
        f"Порядок: {[h.payload['frame_id'] for h in context['hybrid']]}"
    )


@then("без лексичного шару він першим не був")
def _dense_alone_fails(context):
    top = context["dense_only"][0].payload["frame_id"]
    assert top != TARGET, (
        "щільний пошук і без лексики ставить потрібний кадр першим — "
        "тоді сценарій нічого не доводить, і дані треба переробити"
    )


@then("видача має збігтися з видачею без лексичного шару")
def _same_as_dense(context):
    hybrid = [h.payload["frame_id"] for h in context["hybrid"]]
    dense = [h.payload["frame_id"] for h in context["dense_only"]]
    assert hybrid == dense, (
        f"лексичний шар змінив порядок там, де жоден терм не збігся: "
        f"{dense} → {hybrid}"
    )


@then("результат має містити прочитаний текст і назву рушія")
def _provenance_present(context):
    payload = context["hybrid"][0].payload
    assert payload.get("ocr_text"), "у результаті немає прочитаного тексту"
    assert payload.get("ocr_engine"), "у результаті немає назви рушія OCR"
    assert EXACT.lower() in payload["ocr_text"].lower()


@then("система має повідомити про непокриту писемність")
def _unsupported_reported(context):
    result = context["result"]
    assert not result.has_text
    assert result.unsupported, (
        "порожній текст без переліку непокритих писемностей неможливо "
        "відрізнити від справжньої відсутності тексту"
    )
    assert "hebrew" in result.unsupported
