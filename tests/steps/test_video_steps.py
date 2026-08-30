"""Кроки для `video.feature` — відбір ключових кадрів і згортання сцен (M5a).

Набір — реальні ролики з галереї користувача (`data/video/`), бо синтетика тут
не доводить нічого: у проєкті вже тричі вимір на намальованому давав
ПРОТИЛЕЖНУ відповідь (ризик 10). Медіа лежить поза репозиторієм, тож без нього
сценарії пропускаються, а не червоніють.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pytest_bdd import given, scenarios, then, when

scenarios("video.feature")

VIDEO_DIR = Path(__file__).resolve().parents[2] / "data" / "video"

#: Скільки роликів брати: більше не додає властивостей, лише час.
SAMPLE = 3

#: Наскільки мають різнитися dHash сусідніх ВІДІБРАНИХ кадрів. Дорівнює
#: `DUPLICATE_DISTANCE` не випадково: сценарій перевіряє саме те, що
#: дедуплікація справді працює, а не те, що вона налаштована на якесь число.
MIN_APART = 6

#: Стеля швидкості відбору. `video_fps_floor` у профілі 0.5, плюс запас на
#: зміни сцени, які беруться понад інтервал: помітна межа, а не підігнане
#: число.
MAX_RATE = 1.5


@given("проіндексовано відео з галереї", target_fixture="videos")
def _videos():
    from vsearch.search.retrieve import Searcher

    files = sorted(VIDEO_DIR.glob("*.mov")) + sorted(VIDEO_DIR.glob("*.mp4"))
    if not files:
        pytest.skip("немає відео в data/video (особисті матеріали поза репозиторієм)")
    return {"files": files[:SAMPLE], "searcher": Searcher()}


@when("я відбираю ключові кадри з відео", target_fixture="sampled")
def _sample(videos):
    from vsearch.ingest.video import probe, sample_keyframes

    out = []
    for path in videos["files"]:
        info = probe(path)
        keys = list(sample_keyframes(path, fps_floor=0.5))
        assert keys, f"{path.name}: жодного ключового кадру — відбір нічого не дав"
        out.append({"path": path, "info": info, "keys": keys})
    return out


@then("швидкість відбору має лишатися в межах профілю")
def _rate(sampled):
    # Міряється ШВИДКІСТЬ відбору, а не частка кадрів. Частка залежить від
    # тривалості: 7.7-секундний ролик при відборі раз на дві секунди дає
    # чотири кадри, тобто 1.7% від 232, — і це правильна робота, а не поломка.
    # Перша редакція сценарію саме так і помилилася.
    for item in sampled:
        seconds = item["info"]["duration_s"]
        if seconds < 1:
            continue
        rate = len(item["keys"]) / seconds
        assert rate <= MAX_RATE, (
            f"{item['path'].name}: {rate:.2f} ключових кадрів на секунду при "
            f"межі {MAX_RATE} — відбір не звужує, і індекс роздується на відео"
        )


@then("кожен ключовий кадр має мати позицію в часі в межах тривалості")
def _timestamps(sampled):
    for item in sampled:
        limit_ms = item["info"]["duration_s"] * 1000
        for key in item["keys"]:
            assert key.ts_ms >= 0, f"{item['path'].name}: відʼємний час {key.ts_ms}"
            # Допуск на останній кадр: тривалість контейнера округлена.
            assert key.ts_ms <= limit_ms + 1000, (
                f"{item['path'].name}: кадр на {key.ts_ms} мс при тривалості "
                f"{limit_ms:.0f} мс — мітка часу не веде нікуди"
            )


@then("сусідні ключові кадри мають помітно різнитися")
def _distinct(sampled):
    from vsearch.ingest.video import _distance

    for item in sampled:
        # Хеші беруться З САМИХ КАДРІВ, а не перераховуються: інакше сценарій
        # перевіряв би іншу реалізацію, ніж та, якою робиться дедуплікація.
        hashes = [k.digest for k in item["keys"]]
        pairs = [
            (i, _distance(hashes[i - 1], hashes[i])) for i in range(1, len(hashes))
        ]
        too_close = [(i, d) for i, d in pairs if d < MIN_APART]
        assert not too_close, (
            f"{item['path'].name}: відібрані кадри {too_close} майже однакові — "
            "дедуплікація не спрацювала, і статичний запис роздує індекс"
        )


@when("я шукаю у відео те, що там є", target_fixture="found")
def _search_video(videos):
    from vsearch.index import schema
    from vsearch.index.store import VectorStore

    store = VectorStore()
    points, _ = store.client.scroll(
        collection_name=store.name(schema.FRAMES), limit=4096, with_payload=True
    )
    known = {
        str(p.payload.get("frame_id"))
        for p in points
        if p.payload.get("media_type") == "video"
    }
    if not known:
        pytest.skip("відео ще не проіндексоване: `vsearch index data/video`")
    # Запит навмисно НЕ підібраний під конкретний ролик: беремо кілька
    # загальних і лишаємо ті, де відео взагалі знайшлося. Інакше сценарій
    # перевіряв би вдалий добір слова, а не роботу конвеєра.
    for query in ("dog", "street", "a car", "people outdoors"):
        results = videos["searcher"].search(query, limit=40).results
        hits = [r for r in results if r.frame_id in known]
        if hits:
            return {"query": query, "results": results, "hits": hits}
    pytest.skip("жоден пробний запит не підняв відео — потрібен інший матеріал")


@then('кожен відеорезультат має нести тип носія "video" і позицію в часі')
def _video_provenance(found):
    for hit in found["hits"]:
        assert hit.media_type == "video", (
            f"кадр із відео позначений як {hit.media_type!r}: провенанс губить "
            "тип носія, і слідчий не бачить, що це запис"
        )
        assert hit.ts_ms is not None, (
            "відеорезультат без мітки часу веде до файлу, а не до моменту"
        )


@then("одна сцена має бути представлена не більше ніж одним результатом")
def _one_per_shot(found):
    seen: dict[str, int] = {}
    for result in found["results"]:
        shot = result.provenance.get("shot_id") or result.matched_attrs.get("shot_id")
        if shot:
            seen[shot] = seen.get(shot, 0) + 1
    repeated = {k: v for k, v in seen.items() if v > 1}
    assert not repeated, (
        f"сцена займає кілька позицій ({repeated}) — один ролик витісняє з "
        "видачі решту матеріалу"
    )


@then("провенанс має казати, скільки кадрів сцени відповіли")
def _shot_size(found):
    counted = [
        r for r in found["hits"] if r.matched_attrs.get("shot_frames")
    ]
    assert counted, (
        "жоден відеорезультат не каже, скільки кадрів сцени відповіли — "
        "тривалість події втрачена"
    )
