"""Малювання результатів пошуку.

Без рамки на зображенні перевірити знахідку неможливо: слідчий бачить кадр і
мусить вірити на слово, що система знайшла саме той дрібний обʼєкт. Тому рамка
й підпис малюються завжди, коли результат прийшов із регіону.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image

    from vsearch.search.retrieve import SearchResult

logger = logging.getLogger(__name__)

#: Кольори за походженням результату — видно, що саме спрацювало.
COLORS = {
    "object": (255, 64, 64),    # пропозиція детектора при індексації
    "tile": (64, 160, 255),     # плитка
    "frame": (120, 200, 120),   # кадр цілком
    "detected": (255, 176, 0),  # обумовлена детекція під слово запиту
}
FALLBACK_COLOR = (255, 200, 0)

#: Кольори для рамок РІЗНИХ сутностей запиту. Коли запит називає кількох
#: обʼєктів («дівчина з дитиною»), колір за походженням рамки вже не головне:
#: важливіше бачити, ЩО саме позначено кожним прямокутником. Тому сутності
#: отримують власну палітру, а походження лишається в підписі під фото.
ENTITY_COLORS = (
    (255, 96, 96),    # перша сутність
    (96, 176, 255),   # друга
    (128, 216, 128),  # третя
    (240, 176, 64),   # четверта
    (200, 128, 240),  # пʼята
)

#: Позначки джерел. Знати, ЩО саме дало результат, важливо не для краси:
#: якщо знахідки стабільно приходять із плиток, детектор не окуповує свою
#: вартість; якщо з кадрів — плиткування зайве. Без цієї розбивки обидва
#: висновки довелося б угадувати.
SOURCE_MARK = {
    "object": "🔴 обʼєкт",
    "tile": "🔵 плитка",
    "frame": "🟢 кадр",
    "detected": "🟠 детекція",
    "region": "🟡 регіон",
}


def source_of(result: "SearchResult") -> str:
    """Звідки прийшов результат — за найкращою його ділянкою."""
    if result.regions:
        return result.regions[0].kind or "region"
    kind = str(result.matched_attrs.get("region", "")) or "frame"
    return kind


def source_label(result: "SearchResult") -> str:
    return SOURCE_MARK.get(source_of(result), "🟡 регіон")


def sources_summary(results: list) -> str:
    """Зведення по всій видачі: що саме спрацювало і скільки разів."""
    from collections import Counter

    counts = Counter(source_of(r) for r in results)
    if not counts:
        return ""
    parts = [
        f"{SOURCE_MARK.get(kind, kind)} — {n}"
        for kind, n in counts.most_common()
    ]
    return " · ".join(parts)


#: Пояснення до рамок. Живе поруч із малюванням, бо мусить мінятися разом
#: із ним: легенда, що розійшлася з картинкою, гірша за її відсутність.
#: Пояснення до ДВОХ чисел у рядку під фото.
LEGEND_TWO_SCORES = (
    "**Два числа під фото — два різні виміри, і порядок зливає обидва.**  \n"
    "Перше — наскільки кадр схожий на текст запиту. Друге («сутності») — "
    "наскільки підтверджено те, що запит НАЗВАВ: жінку, дитину, сумку. "
    "Кадр може бути слабко схожим на фразу, але впевнено містити названий "
    "обʼєкт — і навпаки.  \n"
    "Тому жодне з них не спадає рівно згори вниз: спадає їхнє злиття. Але "
    "результат ніколи не стоїть нижче за той, що програє йому за ОБОМА."
)

LEGEND_ENTITIES = (
    "**Рамки названі за сутностями запиту.** Кожен обʼєкт, який назвав запит, "
    "має власний колір і підпис: на «дівчина з дитиною» видно окремо рамку "
    "«жінка» й рамку «дитина». Дві безіменні рамки не давали б перевірити, "
    "чи знайдено обох, чи двічі одну людину.  \n"
    "**Відсоток на рамці** — впевненість у тому, що це саме ця сутність "
    "(фасет), а не схожість кадру з текстом запиту. Числа різні навмисно: "
    "рамка відповідає на питання «хто це», рядок над галереєю — «наскільки "
    "кадр відповідає запиту»."
)

LEGEND = (
    "**Рамки:** товста з великим відсотком — ділянка, за якою кадр потрапив "
    "у видачу; тонкі приглушені з номером — інші ділянки, що теж відповіли.  \n"
    "**Колір:** 🔴 пропозиція детектора · 🔵 плитка 288 px · 🟢 кадр цілком · "
    "🟠 детекція під слово запиту.  \n"
    "🟠 зʼявляється лише там, де ввімкнено «Уточнити детекцією» І кадр "
    "потрапив у верхівку уточнення — тому вона є не на кожному фото.  \n"
    "**Відсоток на рамці** — калібрована впевненість саме цієї ділянки, "
    "а не сирий косинус."
)

LEGEND_BEST_ONLY = (
    "**Рамки:** показано лише ту ділянку, за якою кадр потрапив у видачу. "
    "Зніміть «Лише найкраща ділянка», щоб побачити решту знахідок у кадрі.  \n"
    "**Колір:** 🔴 пропозиція детектора · 🔵 плитка 288 px · 🟢 кадр цілком."
)


def placeholder(result: "SearchResult", *, size: int = 320) -> "Image":
    """Плитка на місце кадру, який не вдалося показати.

    Раніше такий результат просто ЗНИКАВ із галереї: `annotate` повертав None,
    і виклик мовчки його пропускав. Наслідок бачив користувач — на запит
    «полуниця» верхній кадр не показувався, а нумерація починалася з #2, тож
    виглядало це як «система знайшла лише одне фото».

    У розслідуванні прихована знахідка коштує дорожче за зайву: місце має
    лишитися видимим і названим, а причина — написаною на ньому.
    """
    from PIL import Image, ImageDraw

    tile = Image.new("RGB", (size, size), (58, 58, 62))
    draw = ImageDraw.Draw(tile)
    name = Path(result.path).name or "без шляху"
    lines = ["ФАЙЛ НЕДОСТУПНИЙ", name[:34], f"{result.probability:.0%}"]
    font = _font(16)
    for index, text in enumerate(lines):
        draw.text((14, 20 + index * 26), text, fill=(240, 200, 90), font=font)
    return tile


def annotate(
    result: "SearchResult", *, max_side: int = 640, best_only: bool = False
) -> "Image | None":
    """Кадр із намальованою рамкою знахідки.

    `best_only` лишає одну рамку — ту, за якою кадр ранжується. Потрібне не
    для краси: після переходу на піксельні плитки кадр несе до шести рамок,
    і питання «а котра з синіх саме та» стає буквальним.
    """
    from PIL import ImageDraw

    from vsearch.ingest.images import load_checked

    path = Path(result.path)
    if not path.exists():
        return None
    try:
        # Той самий завантажувач, що й при індексації. Інакше пошкоджений
        # файл, який ми свідомо врятували на індексації, валив би показ —
        # і слідчий бачив би помилку замість доказу, який система знайшла.
        image, _ = load_checked(path)
    except Exception:  # noqa: BLE001 — один нечитабельний файл не ламає видачу
        logger.warning("не вдалося намалювати %s", path.name)
        return None

    # Зменшуємо ДО малювання, а не після. Інакше рамки й підписи малювалися в
    # роздільності оригіналу (960×1280) і стискалися разом із ним удвічі —
    # підпис у 10 px ставав 5 px і не читався. Рамки нормалізовані, тож
    # порядок кроків на їхню геометрію не впливає.
    image.thumbnail((max_side, max_side))

    # Малюємо ВСІ ділянки, що відповіли на запит, а не лише найкращу.
    # Раніше показувалася одна, і на фото з двома людьми в окулярах виглядало
    # так, ніби другу система не побачила, — хоча в індексі вона була.
    regions = result.regions or (
        [type("R", (), {"bbox": result.bbox, "score": result.score,
                        "kind": str(result.matched_attrs.get("region", "")) or "frame",
                        "label": ""})()]
        if result.bbox else []
    )
    if best_only:
        regions = regions[:1]
    if regions:
        draw = ImageDraw.Draw(image)
        base = max(2, int(min(image.width, image.height) * 0.006))

        # Кожна сутність запиту — свій колір. Порядок стабільний: перша
        # названа в запиті завжди перша в палітрі, тож між кадрами видачі
        # «жінка» лишається того самого кольору.
        entity_order = {
            name: index
            for index, name in enumerate(
                dict.fromkeys(
                    getattr(r, "entity", "") for r in regions if getattr(r, "entity", "")
                )
            )
        }

        # Порядок малювання зворотний: найкраща ділянка кладеться ОСТАННЬОЮ,
        # тож її рамка й підпис лягають поверх решти. Після переходу на
        # піксельні плитки кадр несе до шести рамок, і найважливіша з них
        # інакше опинялася б під сусідніми.
        for order, region in reversed(list(enumerate(regions))):
            x, y, w, h = region.bbox
            left, top = x * image.width, y * image.height
            right, bottom = (x + w) * image.width, (y + h) * image.height
            best = order == 0
            entity = getattr(region, "entity", "")
            color = (
                ENTITY_COLORS[entity_order[entity] % len(ENTITY_COLORS)]
                if entity in entity_order
                else COLORS.get(region.kind, FALLBACK_COLOR)
            )
            named = entity in entity_order
            width = base * 2 if (best or named) else max(1, base // 2)
            if not best and not named:
                # Решта ділянок приглушені, а не просто тонші: різниця в
                # товщині 4 px проти 2 на зменшеній до 640 px картинці не
                # читається, і всі шість плиток виглядали однаково вагомими.
                color = _dim(color)
            draw.rectangle([left, top, right, bottom], outline=color, width=width)
            _label(
                draw, _region_label(region, order), left, top, color, width,
                size=_font_size(image, best=best),
            )

    return image


#: Підпис має бути читабельним на мініатюрі галереї, а не лише у відкритому
#: вигляді.
def _font_size(image: "Image", *, best: bool) -> int:
    """Кегль від БІЛЬШОЇ сторони вже зменшеного кадру.

    Від меншої вести не можна: вертикальне фото 295×640 давало б кегль 14, а
    горизонтальне того ж розміру — 32, і підписи стрибали б між сусідніми
    картками галереї. Більша сторона після `thumbnail` дорівнює `max_side` для
    будь-якої орієнтації, тож розмір виходить сталим.
    """
    base = max(16, int(max(image.width, image.height) * 0.035))
    return base if best else max(12, int(base * 0.65))


def _font(size: int):
    """Шрифт потрібного розміру, вбудований у Pillow.

    Свідомо БЕЗ системних шрифтів: середовище виконання не має мережі й може
    не мати нічого, крім самого образу, тож шрифт, знайдений на ноутбуці,
    просто зник би в продакшені. Наслідок — у підписах лише ASCII: вбудований
    шрифт не має ані кирилиці, ані «★», і обидва малювалися б порожнім
    прямокутником. Тому найкраща ділянка позначена не символом, а більшим
    кеглем, товщою рамкою й повним кольором.
    """
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1 — кегль не налаштовується
        return ImageFont.load_default()


def _dim(color: tuple[int, int, int], amount: float = 0.55) -> tuple[int, int, int]:
    """Змішати колір із сірим — приглушити, не втративши видимості.

    Освітлення до білого зникало б на світлих кадрах, затемнення до чорного —
    на темних. Сірий читається на обох.
    """
    return tuple(int(c + (128 - c) * amount) for c in color)  # type: ignore[return-value]


def _region_label(region, order: int) -> str:
    """Підпис рамки: чому саме ця ділянка тут.

    Показується ЙМОВІРНІСТЬ, а не косинус. Косинус 0.19 проти 0.10 виглядає
    майже однаково, хоча це 100% проти 1% — тобто «ось воно» проти «нічого
    немає». Саме ця різниця й вирішує, куди дивитися.
    """
    probability = getattr(region, "probability", 0.0)
    entity = getattr(region, "entity", "")
    if entity:
        # Назва сутності попереду числа: на «дівчина з дитиною» головне
        # питання не «наскільки впевнено», а «де тут хто».
        #
        # І число — ФАСЕТНЕ, а не косинус із текстом запиту. Це різні виміри:
        # фасет контрастний («людина» проти тла), запит — абсолютна схожість
        # із фразою. На дзеркальному селфі фасет дає 0.85, а «a woman» —
        # 0.0%, бо модель тренована на підписах із видимою постаттю. Підпис
        # «woman · 4.7%» показував число не про ту рамку.
        facet = getattr(region, "entity_score", None)
        value = facet if facet is not None else probability
        return f"{entity} · {format_confidence(value)}"
    text = format_confidence(probability) if order == 0 else f"{order + 1}: {format_confidence(probability)}"
    label = getattr(region, "label", "")
    # Підпис плитки («r3c0») — це її координата в сітці, людині вона нічого не
    # дає. Підпис детектора («handbag») дає, і його варто показати.
    if label and not _is_tile_label(label):
        text += f" · {label}"
    return text


def _is_tile_label(label: str) -> bool:
    return len(label) > 1 and label[0] == "r" and "c" in label and label[1:].replace("c", "").isdigit()


def _label(draw, text: str, left: float, top: float, color, width: int, size: int = 14) -> None:
    """Підпис над рамкою, з підкладкою — інакше не читається на світлому."""
    pad = max(3, size // 5)
    font = _font(size)
    try:
        box = draw.textbbox((0, 0), text, font=font)
    except Exception:  # noqa: BLE001 — без шрифту просто не малюємо підпис
        return
    text_w, text_h = box[2] - box[0], box[3] - box[1]
    # Підпис над рамкою, а якщо зверху немає місця — під верхньою межею,
    # усередині. Інакше на знахідці біля верхнього краю кадру він обрізався б.
    y = top - text_h - 2 * pad - width
    if y < 0:
        y = top + width
    draw.rectangle([left, y, left + text_w + 2 * pad, y + text_h + 2 * pad], fill=color)
    draw.text((left + pad, y + pad - box[1]), text, fill=(0, 0, 0), font=font)


def format_confidence(value: float) -> str:
    """Впевненість у вигляді, де видно різницю там, де вона важлива.

    Округлення до цілих ховає саме той діапазон, у якому ухвалюється
    рішення: 0.3% (обʼєкта немає) і 3.5% (дрібний обʼєкт на фоні) обидва
    показувалися б як «0%» і «4%», а різниця між ними вирішальна.
    """
    if value >= 0.10:
        return f"{value:.0%}"
    if value >= 0.01:
        return f"{value:.1%}"
    return f"{value:.2%}"


def caption(result: "SearchResult", rank: int) -> str:
    """Короткий підпис під зображенням у галереї."""
    parts = [
        f"#{rank}",
        source_label(result),
        format_confidence(result.probability),
        Path(result.path).name,
    ]
    # Друге свідчення показується ОКРЕМИМ числом, а не змішується з першим.
    #
    # Порядок видачі зливає два виміри за рангами (ADR-019), і жодне ОДНЕ
    # число не може бути монотонним із таким злиттям. Спроба показати одне —
    # це або приховати друге свідчення, або збрехати про порядок. Тому їх два,
    # і кожне підписане тим, на що воно відповідає.
    if result.entity_confidence is not None:
        parts.insert(3, f"сутності {format_confidence(result.entity_confidence)}")
    # Пониження має бути ВИДИМИМ. Кадр, у якому слово запиту не заземлилося,
    # лишається у видачі — але слідчий мусить знати, що саме система не
    # підтвердила, інакше нижча позиція виглядає як довільна.
    if result.matched_attrs.get("not_grounded"):
        parts.append(f"⚠ не підтверджено: «{result.matched_attrs['not_grounded']}»")
    if result.matched_attrs.get("instances_found"):
        # Запит просив кілька екземплярів, а підтвердити вдалося менше.
        # Кадр лишається у видачі: система не вміє рахувати надійно (ADR-022),
        # тож це підказка слідчому, а не вирок кадру.
        parts.append(f"⚠ екземплярів {result.matched_attrs['instances_found']}")

    named = [r for r in result.regions if getattr(r, "entity", "")]
    if named:
        # Запит назвав обʼєкти — показуємо впевненість ПО КОЖНОМУ. «Решта ≤»
        # тут була б безглуздою: рамки відповідають на різні питання, і
        # спільної стелі в них немає. Саме через це в підписі «19%» сусідило
        # з «решта ≤22%», що читалося як помилка.
        parts.append(" · ".join(
            f"{r.entity} {format_confidence(r.entity_score if r.entity_score is not None else r.probability)}"
            for r in named
        ))
    elif len(result.regions) > 1:
        kinds = {r.kind for r in result.regions}
        extra = "+".join(SOURCE_MARK.get(k, k).split()[0] for k in sorted(kinds))
        # Разом із кількістю — стеля решти ділянок. Без неї «ділянок 6» читалося
        # як «шість знахідок», хоча зазвичай це одна знахідка й п'ять сусідніх
        # плиток на 1–3%. Число одразу каже, чи варто дивитися на інші рамки.
        rest = max((r.probability for r in result.regions[1:]), default=0.0)
        parts.append(
            f"ділянок {len(result.regions)} {extra} (решта ≤{format_confidence(rest)})"
        )
    if result.bbox:
        area = result.bbox[2] * result.bbox[3]
        parts.append(f"{area * 100:.1f}% кадру")
    if result.ts_ms is not None:
        parts.append(f"{result.ts_ms / 1000:.1f}с")
    return " · ".join(parts)


def describe_parse(parsed) -> str:
    """Розбір запиту в читабельному вигляді — головна панель для перевірки п.11."""
    if parsed is None or (not parsed.must and not parsed.must_not):
        return "_Розбір не застосовувався: короткий запит або вимкнено._"

    lines = [
        f"**Мова оригіналу:** `{parsed.language}`",
        f"**Опис для ембедера:** `{parsed.query_en}`",
        f"**Композитний:** {'так' if parsed.is_compositional else 'ні'} · "
        f"**Заперечення:** {'так' if parsed.has_negation else 'ні'}",
        "",
    ]
    for title, entities in (("Має бути", parsed.must), ("Не має бути", parsed.must_not)):
        lines.append(f"**{title}:**")
        if not entities:
            lines.append("- —")
        for index, entity in enumerate(entities):
            attrs = ", ".join(f"`{a.name.value}={a.value}`" for a in entity.attributes)
            count = f" ×{entity.count}" if entity.count > 1 else ""
            lines.append(
                f"- `[{index}]` **{entity.object.value}**{count} {attrs or ''}".rstrip()
            )
        lines.append("")
    if parsed.relations:
        lines.append("**Просторові відношення:**")
        for relation in parsed.relations:
            kind = "геометрія" if relation.is_geometric else "потребує VLM"
            lines.append(
                f"- `[{relation.subject}]` —*{relation.predicate.value}*→ "
                f"`[{relation.target}]` ({kind})"
            )
        lines.append("")
        lines.append("> ⚠️ Відношення розбираються, але ще не застосовуються в пошуку (M4c).")
    return "\n".join(lines)
