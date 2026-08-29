#!/usr/bin/env python3
"""Генератор синтетичного набору "smoke".

Це НЕ заміна золотому набору з реальних матеріалів. Призначення одне: довести,
що конвеєр працює наскрізь — індексація, векторний пошук, метрики, — не чекаючи
на реальні дані. Розмітка тут відома точно, бо зображення ми самі й малюємо.

Набір навмисно перевіряє те, що синтетика перевірити МОЖЕ: колір, форму,
мультимовність запиту й дрібні обʼєкти. Сцен на кшталт «автомобіль на нічній
вулиці» тут немає й бути не може.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "tests" / "golden" / "smoke"

SIZE = (768, 432)  # 16:9 — перевіряє, що NaFlex не спотворює пропорції
BG = (245, 245, 245)

COLORS = {
    "red": (200, 30, 30),
    "blue": (30, 60, 200),
    "green": (30, 160, 60),
    "yellow": (230, 200, 40),
}
SHAPES = ("circle", "square", "triangle")

#: Запити трьома мовами на ту саму сутність — мультимовність (п.5)
#: перевіряється тим самим набором, без окремої розмітки.
COLOR_WORDS = {
    "red": {"en": "a red shape", "uk": "червона фігура", "he": "צורה אדומה"},
    "blue": {"en": "a blue shape", "uk": "синя фігура", "he": "צורה כחולה"},
    "green": {"en": "a green shape", "uk": "зелена фігура", "he": "צורה ירוקה"},
    "yellow": {"en": "a yellow shape", "uk": "жовта фігура", "he": "צורה צהובה"},
}
SHAPE_WORDS = {
    "circle": {"en": "a circle", "uk": "коло", "he": "עיגול"},
    "square": {"en": "a square", "uk": "квадрат", "he": "ריבוע"},
    "triangle": {"en": "a triangle", "uk": "трикутник", "he": "משולש"},
}


def draw(shape: str, color: tuple[int, int, int], small: bool = False) -> Image.Image:
    img = Image.new("RGB", SIZE, BG)
    d = ImageDraw.Draw(img)
    if small:
        # ~0.6% площі кадру: на цілому кадрі майже зникає, кропом — видно.
        cx, cy, r = 640, 350, 26
    else:
        cx, cy, r = SIZE[0] // 2, SIZE[1] // 2, 130
    box = [cx - r, cy - r, cx + r, cy + r]
    if shape == "circle":
        d.ellipse(box, fill=color)
    elif shape == "square":
        d.rectangle(box, fill=color)
    else:
        d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=color)
    return img


# ── набір "clutter": дрібний обʼєкт серед захаращення ───────────────────────
#
# На білому тлі SigLIP знаходить обʼєкт на 0.6% площі не гірше за великий, тож
# такий набір нічого не доводить про плиткування. Тут тло навмисно заповнене
# сірими та коричневими прямокутниками, а ціль — насичений колір. Так колірна
# розмітка лишається однозначною, а повнокадровий ембединг втрачає ціль серед
# деталей — саме той випадок, заради якого існує п.12.

CLUTTER_OUT = REPO_ROOT / "tests" / "golden" / "clutter"
CLUTTER_TONES = [
    (110, 110, 115), (95, 88, 80), (135, 130, 125), (70, 68, 72),
    (150, 145, 138), (88, 92, 96), (120, 108, 95), (60, 60, 64),
]


def draw_clutter(shape: str, color: tuple[int, int, int], seed: int, tiny: bool):
    import random

    rng = random.Random(seed)
    img = Image.new("RGB", SIZE, (100, 100, 104))
    d = ImageDraw.Draw(img)
    for _ in range(70):
        x, y = rng.randint(0, SIZE[0]), rng.randint(0, SIZE[1])
        w, h = rng.randint(20, 110), rng.randint(20, 90)
        d.rectangle([x, y, x + w, y + h], fill=rng.choice(CLUTTER_TONES))

    r = 16 if tiny else 60
    cx = rng.randint(r + 10, SIZE[0] - r - 10)
    cy = rng.randint(r + 10, SIZE[1] - r - 10)
    box = [cx - r, cy - r, cx + r, cy + r]
    if shape == "circle":
        d.ellipse(box, fill=color)
    elif shape == "square":
        d.rectangle(box, fill=color)
    else:
        d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=color)

    area = (2 * r) ** 2 / (SIZE[0] * SIZE[1])
    bbox = [(cx - r) / SIZE[0], (cy - r) / SIZE[1], 2 * r / SIZE[0], 2 * r / SIZE[1]]
    return img, area, bbox


def build_clutter() -> int:
    media = CLUTTER_OUT / "media"
    media.mkdir(parents=True, exist_ok=True)

    assets, index = [], 0
    for color_name, rgb in COLORS.items():
        for shape in SHAPES:
            for tiny in (True, False):
                index += 1
                asset_id = f"cl_{index:03d}"
                img, area, bbox = draw_clutter(shape, rgb, seed=index * 7, tiny=tiny)
                img.save(media / f"{asset_id}.png")
                assets.append({
                    "asset_id": asset_id,
                    "path": f"media/{asset_id}.png",
                    "media_type": "image",
                    "labels": {"color": color_name, "shape": shape, "small": tiny},
                    "objects": [{"label": shape, "bbox": bbox, "area_ratio": round(area, 5)}],
                    "notes": "дрібна ціль серед захаращення" if tiny else "велика ціль серед захаращення",
                })

    by_color: dict[str, list[str]] = {c: [] for c in COLORS}
    tiny_ids: set[str] = set()
    for asset in assets:
        by_color[asset["labels"]["color"]].append(asset["asset_id"])
        if asset["labels"]["small"]:
            tiny_ids.add(asset["asset_id"])

    queries = []
    for color_name, words in COLOR_WORDS.items():
        for lang in ("en", "uk"):
            relevant = by_color[color_name]
            queries.append({
                "query_id": f"clutter_{color_name}_{lang}",
                "text": words[lang],
                "lang": lang,
                "relevant": relevant,
                "forbidden": [a for c, ids in by_color.items() if c != color_name for a in ids],
                "gains": {a: (1.0 if a in tiny_ids else 3.0) for a in relevant},
            })
    # Окремі запити тільки на дрібні цілі — саме вони вимірюють п.12.
    # gains обовʼязкові: без них nDCG не рахується, і порівняння конфігурацій
    # втрачає єдину метрику, чутливу до ПОРЯДКУ, а не лише до складу видачі.
    for color_name, words in COLOR_WORDS.items():
        tiny_of_color = [a for a in by_color[color_name] if a in tiny_ids]
        queries.append({
            "query_id": f"clutter_tiny_{color_name}",
            "text": words["uk"],
            "lang": "uk",
            "relevant": tiny_of_color,
            "gains": {a: 3.0 for a in tiny_of_color},
            "notes": "лише дрібні цілі: без плиткування повнокадровий ембединг їх втрачає",
        })

    (CLUTTER_OUT / "assets.jsonl").write_text(
        "\n".join(json.dumps(a, ensure_ascii=False) for a in assets) + "\n", encoding="utf-8")
    (CLUTTER_OUT / "queries.jsonl").write_text(
        "\n".join(json.dumps(q, ensure_ascii=False) for q in queries) + "\n", encoding="utf-8")
    (CLUTTER_OUT / ".gitignore").write_text("media/\n", encoding="utf-8")

    tiny_area = min(a["objects"][0]["area_ratio"] for a in assets if a["labels"]["small"])
    print(f"Створено набір clutter: {len(assets)} зображень, {len(queries)} запитів")
    print(f"  найдрібніша ціль: {tiny_area*100:.2f}% площі кадру")
    return len(assets)


# ── набір "binding": два обʼєкти з перехресними ознаками ────────────────────
#
# Перевіряє проблему звʼязування — найтонше місце всієї архітектури.
#
# Кадр із червоним колом і синім квадратом та кадр із синім колом і червоним
# квадратом на РІВНІ КАДРУ виглядають однаково: обидва містять «червоне» і
# обидва містять «коло». Розрізнити їх можна лише там, де ознаки лежать на
# одному обʼєкті, тобто на рівні регіону.
#
# Саме тому фрейм-рівневі атрибути шкідливі за побудовою, а не просто неточні:
# їхня конʼюнкція стверджує те, чого в кадрі немає.

BINDING_OUT = REPO_ROOT / "tests" / "golden" / "binding"

#: Пари (лівий обʼєкт, правий обʼєкт). Перші дві — дзеркальні: однаковий набір
#: кольорів і форм, протилежне звʼязування.
BINDING_PAIRS = [
    (("red", "circle"), ("blue", "square")),
    (("blue", "circle"), ("red", "square")),
    (("green", "triangle"), ("yellow", "circle")),
    (("yellow", "triangle"), ("green", "circle")),
    (("red", "square"), ("green", "circle")),
    (("green", "square"), ("red", "circle")),
]


def draw_pair(left: tuple[str, str], right: tuple[str, str]) -> Image.Image:
    img = Image.new("RGB", SIZE, BG)
    d = ImageDraw.Draw(img)
    for (color_name, shape), cx in ((left, 200), (right, 568)):
        rgb = COLORS[color_name]
        r, cy = 105, SIZE[1] // 2
        box = [cx - r, cy - r, cx + r, cy + r]
        if shape == "circle":
            d.ellipse(box, fill=rgb)
        elif shape == "square":
            d.rectangle(box, fill=rgb)
        else:
            d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=rgb)
    return img


def build_binding() -> int:
    media = BINDING_OUT / "media"
    media.mkdir(parents=True, exist_ok=True)

    assets = []
    for index, (left, right) in enumerate(BINDING_PAIRS, start=1):
        asset_id = f"bd_{index:03d}"
        draw_pair(left, right).save(media / f"{asset_id}.png")
        assets.append({
            "asset_id": asset_id,
            "path": f"media/{asset_id}.png",
            "media_type": "image",
            "labels": {
                "left_color": left[0], "left_shape": left[1],
                "right_color": right[0], "right_shape": right[1],
                # Набір ознак КАДРУ — навмисно однаковий у дзеркальних парах.
                "frame_colors": sorted({left[0], right[0]}),
                "frame_shapes": sorted({left[1], right[1]}),
            },
            "objects": [
                {"label": f"{left[0]} {left[1]}", "bbox": [0.12, 0.26, 0.27, 0.49],
                 "area_ratio": 0.132},
                {"label": f"{right[0]} {right[1]}", "bbox": [0.60, 0.26, 0.27, 0.49],
                 "area_ratio": 0.132},
            ],
            "notes": f"{left[0]} {left[1]} ліворуч, {right[0]} {right[1]} праворуч",
        })

    def has_bound(asset, color, shape):
        labels = asset["labels"]
        return (
            (labels["left_color"] == color and labels["left_shape"] == shape)
            or (labels["right_color"] == color and labels["right_shape"] == shape)
        )

    def has_unbound(asset, color, shape):
        """Кадр містить і колір, і форму — але на РІЗНИХ обʼєктах."""
        labels = asset["labels"]
        return (
            color in labels["frame_colors"]
            and shape in labels["frame_shapes"]
            and not has_bound(asset, color, shape)
        )

    queries = []
    for color in ("red", "blue", "green", "yellow"):
        for shape in ("circle", "square", "triangle"):
            relevant = [a["asset_id"] for a in assets if has_bound(a, color, shape)]
            # Заборонені — не «нерелевантні взагалі», а саме пастка звʼязування:
            # кадри, де є і колір, і форма, але порізно.
            forbidden = [a["asset_id"] for a in assets if has_unbound(a, color, shape)]
            if not relevant or not forbidden:
                continue
            queries.append({
                "query_id": f"bind_{color}_{shape}",
                "text": f"a {color} {shape}",
                "lang": "en",
                "relevant": relevant,
                "forbidden": forbidden,
                "gains": {a: 3.0 for a in relevant},
                "notes": "заборонені кадри містять ті самі ознаки, але на різних обʼєктах",
            })

    (BINDING_OUT / "assets.jsonl").write_text(
        "\n".join(json.dumps(a, ensure_ascii=False) for a in assets) + "\n", encoding="utf-8")
    (BINDING_OUT / "queries.jsonl").write_text(
        "\n".join(json.dumps(q, ensure_ascii=False) for q in queries) + "\n", encoding="utf-8")
    (BINDING_OUT / ".gitignore").write_text("media/\n", encoding="utf-8")

    traps = sum(len(q["forbidden"]) for q in queries)
    print(f"Створено набір binding: {len(assets)} зображень, {len(queries)} запитів")
    print(f"  пасток звʼязування (ті самі ознаки на різних обʼєктах): {traps}")
    return len(assets)


def main() -> int:
    media = OUT / "media"
    media.mkdir(parents=True, exist_ok=True)

    assets, index = [], 0
    for color_name, rgb in COLORS.items():
        for shape in SHAPES:
            for small in (False, True):
                index += 1
                asset_id = f"sm_{index:03d}"
                name = f"{asset_id}.png"
                draw(shape, rgb, small).save(media / name)
                area = 0.006 if small else 0.163
                assets.append(
                    {
                        "asset_id": asset_id,
                        "path": f"media/{name}",
                        "media_type": "image",
                        "labels": {"color": color_name, "shape": shape, "small": small},
                        "objects": [
                            {
                                "label": shape,
                                "bbox": [0.80, 0.75, 0.07, 0.12] if small
                                else [0.33, 0.10, 0.34, 0.60],
                                "area_ratio": area,
                            }
                        ],
                        "notes": "дрібний обʼєкт у куті кадру" if small else "",
                    }
                )

    by_color: dict[str, list[str]] = {c: [] for c in COLORS}
    by_shape: dict[str, list[str]] = {s: [] for s in SHAPES}
    is_small: dict[str, bool] = {}
    for asset in assets:
        by_color[asset["labels"]["color"]].append(asset["asset_id"])
        by_shape[asset["labels"]["shape"]].append(asset["asset_id"])
        is_small[asset["asset_id"]] = asset["labels"]["small"]

    # Релевантність включає обидва розміри: дрібна червона фігура — теж
    # червона фігура. Різницю в помітності виражають gains для nDCG, а не
    # виключення з relevant. Перша версія набору робила саме цю помилку, і
    # сценарій упав не через модель, а через розмітку.
    def graded(ids: list[str]) -> dict[str, float]:
        return {a: (1.0 if is_small[a] else 3.0) for a in ids}

    queries = []
    for color_name, words in COLOR_WORDS.items():
        for lang, text in words.items():
            relevant = by_color[color_name]
            forbidden = [
                a for c, ids in by_color.items() if c != color_name for a in ids
            ]
            queries.append(
                {
                    "query_id": f"color_{color_name}_{lang}",
                    "text": text,
                    "lang": lang,
                    "relevant": relevant,
                    "forbidden": forbidden,
                    "gains": graded(relevant),
                }
            )
    for shape, words in SHAPE_WORDS.items():
        for lang in ("en", "uk"):
            relevant = by_shape[shape]
            queries.append(
                {
                    "query_id": f"shape_{shape}_{lang}",
                    "text": words[lang],
                    "lang": lang,
                    "relevant": relevant,
                    "gains": graded(relevant),
                }
            )

    (OUT / "assets.jsonl").write_text(
        "\n".join(json.dumps(a, ensure_ascii=False) for a in assets) + "\n", encoding="utf-8"
    )
    (OUT / "queries.jsonl").write_text(
        "\n".join(json.dumps(q, ensure_ascii=False) for q in queries) + "\n", encoding="utf-8"
    )
    (OUT / ".gitignore").write_text("media/\n", encoding="utf-8")

    print(f"Створено набір smoke: {len(assets)} зображень, {len(queries)} запитів")
    build_clutter()
    build_binding()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
