"""Схема індексу Qdrant.

Три колекції з різними ролями (див. docs/architecture.md):

  frames   — один вектор на кадр або ключовий кадр відео; груба семантика;
  regions  — вектор на плитку чи кроп обʼєкта; саме він знаходить дрібне (п.12);
  faces    — окремий простір ембедингів облич (п.13), за фіче-флагом.

Вектори зберігаються з бінарною квантизацією: 20M векторів по 1152 виміри у
fp32 — це 92 ГБ, у 1 біт на вимір — 2.9 ГБ. Оригінали лежать на диску і
використовуються для rescore топ-кандидатів, тому падіння якості мале, а
виграш у памʼяті визначає, поміститься система в машину чи ні.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

FRAMES = "frames"
REGIONS = "regions"
FACES = "faces"

#: Розмірність ембедингів облич (SFace). Не збігається з візуальною вежею,
#: тому це окрема колекція, а не ще одне поле в regions.
FACE_DIM = 128

#: Імʼя розрідженого вектора з лексичним шаром (OCR, згодом ASR).
#:
#: Іменований, на відміну від щільного: Qdrant вимагає імені для розріджених,
#: а щільний лишається безіменним, щоб не переписувати наявні виклики.
SPARSE_TEXT = "text"

#: Скільки кандидатів піднімати з бінарного індексу перед rescore.
#: Множник до limit; менше — швидше, але падає recall.
DEFAULT_OVERSAMPLING = 4.0


@dataclass(frozen=True)
class PayloadField:
    """Поле payload, яке треба індексувати для фільтрації.

    У Qdrant payload-індекс не створюється сам: без нього фільтр працює, але
    вироджується в повний перебір. А фільтри — це те, чим ми реалізуємо
    заперечення, тож саме тут вони мають бути швидкими.
    """

    name: str
    schema: str  # keyword | integer | float | bool | text
    purpose: str


FRAME_PAYLOAD: tuple[PayloadField, ...] = (
    PayloadField("asset_id", "keyword", "згортання результатів до одного активу"),
    PayloadField("media_type", "keyword", "фільтр «лише відео» / «лише фото»"),
    PayloadField("shot_id", "keyword", "згортання кадрів у сцену (п.8)"),
    PayloadField("ts_ms", "integer", "позиція в часі для відео"),
    PayloadField("caption", "text", "лексичний шар BM25"),
    PayloadField("ocr_text", "text", "написи, номери, документи"),
    PayloadField("asr_text", "text", "мовлення у відео"),
    PayloadField("indexed_at", "integer", "відбір за часом індексації"),
)

REGION_PAYLOAD: tuple[PayloadField, ...] = (
    PayloadField("asset_id", "keyword", "згортання до активу"),
    PayloadField("frame_id", "keyword", "звʼязок регіону з кадром"),
    PayloadField("region_type", "keyword", "tile | object | person | frame"),
    PayloadField("label", "keyword", "клас від детектора"),
    PayloadField("area_ratio", "float", "відсів надто дрібних або надто великих"),
    # Фасети сюди НЕ виписуються руками: їх дає `facet_fields()` із того
    # самого реєстру прототипів, який їх і обчислює. Два переліки вже
    # розходилися одного разу, і це коштувало повного перебору на кожному
    # структурному фільтрі.
)

FACE_PAYLOAD: tuple[PayloadField, ...] = (
    PayloadField("asset_id", "keyword", "згортання до активу"),
    PayloadField("frame_id", "keyword", "звʼязок із кадром"),
    PayloadField("identity_cluster_id", "keyword", "кластер особи"),
    PayloadField("gender_hint", "keyword", "допоміжна ознака, не факт"),
    PayloadField("age_band_hint", "keyword", "допоміжна ознака, не факт"),
)

def facet_fields() -> tuple[PayloadField, ...]:
    """Payload-індекси для ФАСЕТІВ, виведені з реєстру прототипів.

    Раніше цей перелік писався руками окремо від того, що насправді
    записується, і два переліки розійшлися непомітно: індекси існували для
    `attr_gender` та `attr_age_band`, яких код НІКОЛИ не писав, а ключі, за
    якими справді фільтрують запити — `cat_person`, `attr_gender_male`,
    `attr_adult` і 36 категорій — індексу не мали. Наслідок був тихий:
    кожен структурний фільтр вироджувався в повний перебір на боці Qdrant.
    На кількох тисячах точок це непомітно, на мільйонах — непридатно.

    Тому джерело одне. Імпорт лінивий: `schema` має лишатися придатним до
    імпорту без важких залежностей `represent`.
    """
    from vsearch.represent.categories import DEFAULT_CATEGORIES, PERSON_ATTRIBUTES

    return tuple(
        PayloadField(
            proto.payload_key,
            "bool",
            f"фасет {proto.name}: фільтр і передумова",
        )
        for proto in (*DEFAULT_CATEGORIES, *PERSON_ATTRIBUTES)
    )


PAYLOAD_BY_COLLECTION: dict[str, tuple[PayloadField, ...]] = {
    FRAMES: FRAME_PAYLOAD,
    REGIONS: REGION_PAYLOAD,
    FACES: FACE_PAYLOAD,
}


def vector_params(dim: int, *, quantize: bool) -> dict[str, Any]:
    """Параметри вектора для створення колекції.

    Повертається як звичайний словник, щоб модуль лишався імпортовним без
    qdrant_client — його читають і тести схеми, і документація.
    """
    params: dict[str, Any] = {
        "size": dim,
        "distance": "Cosine",
        # Оригінали на диску: у памʼяті лишається лише бінарний індекс.
        "on_disk": quantize,
    }
    if quantize:
        params["quantization_config"] = {
            "binary": {"always_ram": True},
        }
    return params


def index_signature_key() -> str:
    """Ключ, під яким у каталозі лежить підпис індексу.

    Вектори, побудовані іншим ембедером або з іншою роздільністю NaFlex,
    лежать в іншому просторі. Мовчазне змішування дало б не помилку, а просто
    погані результати — найгірший вид поломки, тому підпис звіряється явно.
    """
    return "index_signature"
