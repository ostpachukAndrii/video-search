"""Пошук.

На M1 це щільний пошук по кадрах. Структура результату вже містить усе, що
знадобиться далі — регіон, рамку, збіги фасетів, пояснення — щоб додавання
регіонів (M2), фасетів (M3) і заперечень (M4) не ламало споживачів API.

Кожен результат несе провенанс: чим саме він знайдений і якою версією моделі.
Для розслідування результат без пояснення непридатний, а версіонування робить
його відтворюваним.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from collections.abc import Sequence

from vsearch.config import Profile, get_profile
from vsearch.index import schema
from vsearch.index.catalog import Catalog
from vsearch.index.store import Hit, VectorStore, build_bound_filter, build_filter
from vsearch.represent import lexical
from vsearch.represent.embed import Siglip2Embedder
from vsearch.search.query_model import EMPTY, ObjectClass, StructuredQuery

#: Скільки ділянок показувати на одному кадрі. Більше — каша: Florence-2
#: пропонує вкладені рамки (людина, потім її обличчя), і всі вони відповідають
#: на запит про людину.
MAX_SHOWN_REGIONS = 6

#: Поріг перекриття для показу. Нижчий за індексаційний (0.85): у видачі
#: важливіше не засмітити кадр, ніж зберегти кожен відтінок знахідки.
DISPLAY_IOU = 0.55

#: Частка вкладеності, за якої дрібна рамка вважається тією самою знахідкою,
#: що й більша. Florence-2 пропонує обличчя всередині людини, і обидві чесно
#: відповідають на запит про людину — але показувати їх окремо лише збиває.
DISPLAY_CONTAINMENT = 0.8


def _collect_regions(hit: Hit) -> list["RegionHit"]:
    """Усі ділянки кадру, що відповіли на запит, — очищені для показу.

    Ранжування лишає один хіт на кадр, але знахідок у ньому може бути кілька:
    на фото з двома людьми в окулярах система знаходить обох. Показувати одну
    рамку означало б стверджувати, що другої людини не помічено.
    """
    from vsearch.represent.tiling import Region, deduplicate

    candidates: list[Region] = []
    entity_of: dict[int, str] = {}
    score_of: dict[int, float | None] = {}
    for candidate in [hit, *hit.siblings]:
        bbox = candidate.payload.get("bbox")
        if not bbox:
            continue
        try:
            candidates.append(
                Region(
                    *bbox,
                    # Кадри не мають `region_type` — він є лише в регіонах.
                    # Без цієї гілки будь-яка знахідка рівня кадру падала в
                    # «невідомий тип» і малювалася жовтим, хоча легенда обіцяє
                    # зелений. Жовтий лишається тільки для справді невідомого.
                    kind=(
                        candidate.payload.get("region_type")
                        or ("frame" if candidate.collection == schema.FRAMES else "region")
                    ),
                    # Підпис сутності має пріоритет над міткою детектора: для
                    # людини «жінка» змістовніше за «person».
                    label=candidate.entity
                    or str(candidate.payload.get("label") or ""),
                    score=candidate.score,
                )
            )
            entity_of[id(candidates[-1])] = candidate.entity
            score_of[id(candidates[-1])] = candidate.entity_score
        except ValueError:
            continue

    # Рамки РІЗНИХ сутностей не зливаються між собою, хоч би як вони
    # перекривалися: дитина на руках у жінки лежить усередині її рамки, і
    # відсів за вкладеністю прибрав би саме те, що треба показати.
    named = [c for c in candidates if entity_of.get(id(c))]
    rest = [c for c in candidates if not entity_of.get(id(c))]
    kept_named: list[Region] = []
    for region in named:
        if not any(
            entity_of.get(id(region)) == entity_of.get(id(other))
            and region.iou(other) >= DISPLAY_IOU
            for other in kept_named
        ):
            kept_named.append(region)
    kept_rest = [
        region for region in deduplicate(
            rest, iou_threshold=DISPLAY_IOU, containment_threshold=DISPLAY_CONTAINMENT,
        )
        if not any(region.iou(other) >= DISPLAY_IOU for other in kept_named)
    ]
    # Коли запит назвав обʼєкти, показуємо САМЕ їх. Безіменні рамки міряють
    # інше питання — схожість із усім текстом запиту, а не з конкретною
    # сутністю, — тому їхні відсотки не порівнянні з іменованими: у кадрі
    # «жінка 19%» сусідило з безіменною плиткою на 92%, і це читалося як
    # помилка ранжування. Та сама помилка змішування шкал, лише у показі.
    #
    # Відома межа: на кадрі з двома жінками рамка буде одна — жорсткий прохід
    # лишає по одному найкращому регіону на сутність.
    kept = (kept_named if kept_named else kept_rest)[:MAX_SHOWN_REGIONS]
    return [
        RegionHit(
            bbox=region.bbox, score=region.score, kind=region.kind,
            label=region.label, entity=entity_of.get(id(region), ""),
            entity_score=score_of.get(id(region)),
        )
        for region in kept
    ]


#: У скільки разів глибше тягнути регіони порівняно з бажаною кількістю
#: результатів. Регіонів на кадр кілька, тож без запасу один деталізований
#: кадр зайняв би всю вибірку.
logger = logging.getLogger(__name__)

REGION_FETCH_FACTOR = 6

#: Мінімальна глибина вибірки кандидатів, незалежно від того, скільки
#: результатів попросили показати.
#:
#: Без неї глибина дорівнювала `limit × фактор`, і запит на 3 результати
#: піднімав 18 регіонів проти 60 при запиті на 10 — тобто просто не бачив
#: кращих кандидатів. Видача на 3 і перші 3 з видачі на 10 розходилися, а
#: виглядало це як нестабільність ранжування.
#:
#: Кількість показаних результатів — рішення інтерфейсу. Глибина пошуку —
#: рішення про повноту. Змішувати їх не можна.
MIN_CANDIDATE_DEPTH = 120

#: Наскільки слабшим за найкращий може бути фасет ділянки, щоб її все одно
#: можна було показати як представника сутності.
#:
#: Рамка відповідає на питання «де тут хто», ранжування — на «наскільки кадр
#: підходить запиту». Це різні питання, а відповідала на обидва одна ділянка —
#: найсхожіша на ТЕКСТ запиту. На знімку з шезлонга це давало рамку на
#: плавчині вдалині: її голова видима, тож вона схожа на «a woman» більше за
#: ноги на пів кадру. При цьому детектор запропонував рамку навколо ніг (51%
#: кадру), і фасет визнав її людиною з оцінкою 0.81 проти 0.91 у плавчині —
#: різниця саме в межах цього допуску.
ENTITY_FACET_MARGIN = 0.15

#: Скільки кандидатів доводити до переоцінки канонічним текстом.
#:
#: Проходи віддають верхівку за РАНГАМИ, і лише потім вона переоцінюється
#: одним текстом. Якщо віддати рівно `limit`, кандидати нижче межі не
#: отримають шансу — а переоцінка часто піднімає саме їх. Через це видача на
#: 3 і перші 3 з видачі на 10 розходилися.
RESCORE_DEPTH = 30

#: Скільки кандидатів тягнути на КОЖНУ сутність у жорсткому проході.
#:
#: Тут діє інша логіка, ніж у щільному пошуку. Відбирає ФІЛЬТР, а схожість
#: лише впорядковує вже відібране — тож обмежувати вибірку кількома десятками
#: «найсхожіших» означає викидати придатні кадри ні за що.
#:
#: Виміряно на реальному прикладі: за запитом «чоловік та жінка» потрібне фото
#: стояло на 18 позиції серед жіночих регіонів і на 80 серед чоловічих. При
#: глибині 50 воно випадало з перетину — тобто кадр, що ідеально відповідає
#: запиту, не показувався взагалі.
#:
#: Фільтр і так тримає вибірку малою (331 чоловічий регіон, 709 жіночих на
#: 1654), тож глибина обмежується згори лише заради панорамних випадків.
STRICT_FETCH_LIMIT = 2000

#: Стеля точного обходу заборонених кадрів. Потрібна не для швидкості, а щоб
#: заперечення на half-корпусі («не людина» — 46% точок) не пробігало пів
#: індексу. Обрив НАЗИВАЄТЬСЯ вголос: мовчазне перетворення точного
#: виключення на приблизне — це рівно той клас помилки, що коштував тут
#: найдорожче.
NEGATION_SCAN_CAP = 200_000

#: Наскільки ознака з відкритого словника впливає на порядок видачі.
#:
#: Різниця косинусів прототипу лежить у тому ж масштабі, що й сама схожість,
#: тож одиниця означає «ознака важить стільки ж, скільки текстовий збіг».
#: Більше зробило б ознаку важливішою за сам запит, менше — непомітною.
ON_THE_FLY_WEIGHT = 1.0

#: Де шукати. Вибір між цими режимами вирішує проблему звʼязування.
#:
#: Кадр із червоним колом і синім квадратом та кадр із синім колом і червоним
#: квадратом мають майже однакові вектори КАДРУ: обидва містять «червоне» і
#: обидва містять «коло». Розрізнити їх можна лише там, де ознаки лежать на
#: одному обʼєкті — тобто на рівні регіону.
#:
#: Виміряно на наборі binding (9 запитів-пасток, метрика «жодна пастка не
#: стоїть вище за релевантне»):
#:   тільки кадри   — 3/9   ← кадрові вектори звʼязування не тримають узагалі
#:   кадри+регіони  — 8/9
#:   supplement     — 9/9   ← ранжують регіони, кадри лише доповнюють повноту
#:
#: Ціна supplement — близько 1% Recall@10 на наборі smoke. Вона виправдана:
#: помилка звʼязування дає впевнено хибний результат, а в розслідуванні це
#: найгірший різновид помилки.
SCOPE_AUTO = "auto"
SCOPE_FRAMES = "frames"
SCOPE_REGIONS = "regions"
SCOPE_BOTH = "both"


#: Скільки результатів має дати жорсткий прохід, щоб не вмикати мʼякий.
MIN_STRICT_RESULTS = 5

#: Скільки верхніх кадрів уточнювати обумовленою детекцією.
#:
#: Обмеження чисто вартісне: ~1.6 с на кадр. Глибше має сенс лише тоді, коли
#: уточнення піде в фон і не триматиме користувача.
REFINE_TOP_K = 10

#: У скільки разів розширювати знайдену рамку перед оцінюванням.
#:
#: Детектор окреслює обʼєкт ВПРИТУЛ, а модель упізнає його, коли той займає
#: близько половини кропа (ADR-010) — рамка впритул позбавляє її контексту,
#: за яким вона й розпізнає. Виміряно на чотирьох випадках:
#:
#:     кадр / запит              ×1.0   ×1.5   ×2.0
#:     дерев'яний сарай          1.4%  22.6%   9.8%
#:     жіноча сумка             61.6%  74.3%   6.0%
#:     дівчина в синій спідниці 56.5%  24.2%  28.3%
#:     дівчина в купальнику      2.0%   5.8%  24.0%
#:
#: ×1.5 найкращий у середньому і не провалюється в жодному. Для сараю це
#: різниця між «невидимий» і «знайдений».
DETECTION_CONTEXT = 1.5

#: Скільки кадрів піднімати лексичним проходом.
#:
#: Щедро, бо відбирає тут ФІЛЬТР за термами, а не схожість: кадр або містить
#: рядок, або ні. Той самий урок, що з глибиною жорсткого проходу.
LEXICAL_FETCH_LIMIT = 200

#: Яку частку слів запиту треба знайти в тексті, щоб рахувати це збігом.
#:
#: Половина, а не будь-який збіг: одне випадкове спільне слово («на», «the»)
#: є майже в кожному довгому знімку інтерфейсу, і без межі такі кадри
#: спливали б на кожен багатослівний запит.
MIN_LEXICAL_COVERAGE = 0.5

#: Межа, нижче якої збіг вважається слабким. Це ПОЗНАЧКА, а не фільтр:
#: за замовчуванням нічого не відсівається.
#:
#: Косинус сам по собі оманливий — 0.076 і 0.149 виглядають як «трохи гірше»
#: і «трохи краще», хоча різниця між ними це різниця між «цього тут немає» і
#: «це тут є». Сигмоїда моделі розводить їх однозначно: на реальному наборі
#: «гелікоптер» дав 0.0026, «білий велосипед» 0.0349, а «людина» 0.5539.
#:
#: Значення 1% обрано за вимірами на реальному наборі, а не на око:
#:
#:   «гелікоптер», якого в наборі немає   — 0.26%
#:   білий велосипед: є, але дрібний і на фоні — 3.5%
#:   «людина»                              — 55%
#:
#: Тобто 1% відсікає завідомо відсутнє й не чіпає слабких, але справжніх
#: знахідок. Ставити вище було б небезпечно: на 5% зник би той велосипед,
#: а прихований доказ у розслідуванні коштує дорожче за зайвого кандидата.
#:
#: Тому приховане ЗАВЖДИ перелічується в notice, і його можна показати.
#:
#: І застосовується ця межа лише в ШАРІ ПОДАННЯ, а не тут. Пошук віддає
#: знайдене з оцінкою впевненості; що з цим робити — рішення інтерфейсу.
#: Коли межа стояла типовою в самому пошуку, вона мовчки відсікала справжні
#: збіги й завалила метрики на золотих наборах.
MIN_PROBABILITY = 0.01

#: Скільки стандартних відхилень над медіаною пулу кандидатів робить збіг
#: «не шумом». Конвенційна статистична межа, НЕ підібрана під цей корпус:
#: підбирати поріг на кількох вимірах — окремо описана помилка (ризик 10).
RELATIVE_SIGMA = 2.0


def _stands_out(pool: list[float]) -> float:
    """Косинус, вище якого кандидат вирізняється на тлі ВЛАСНОГО пулу запиту.

    Абсолютна впевненість залежить від формулювання канонічної фрази: «a can
    of pepsi» і «a pepsi can» дають 0.087 і 0.121 на тому самому кадрі. Пул
    міряється тим самим текстом, що й кандидат, тож ця межа від формулювання
    не залежить.

    Пул менший за три кандидати нічого не каже про розподіл — тоді відносної
    межі немає, і лишається сама абсолютна.
    """
    if len(pool) < 3:
        return float("inf")
    arr = np.asarray(pool, dtype=np.float32)
    spread = float(arr.std())
    if spread <= 0:
        return float("inf")
    return float(np.median(arr)) + RELATIVE_SIGMA * spread


def _category_key(name: str) -> str:
    """Імʼя категорії → ключ payload. Приймає і коротку назву, і повний ключ."""
    return name if name.startswith(("cat_", "attr_")) else f"cat_{name}"


#: Згладжувальна стала RRF. Класичне значення 60 із роботи Cormack та ін.:
#: воно робить різницю між 1-ю та 2-ю позиціями помітною, але не переважною,
#: тож один канал не може виграти самим лише впевненим першим місцем.
RRF_K = 60

#: Вага канонічної англійської відносно оригіналу запиту.
#:
#: Рівні ваги виявилися компромісом на користь гіршого каналу: коли модель не
#: заземлює поняття мовою оригіналу, вона впевнено відповідає на СПРОЩЕНИЙ
#: запит («дівчина в синьому» замість «дівчина в синій спідниці»), і половина
#: голосу віддається цій підміні.
#:
#: Виміряно на семи запитах із перевіреними очима відповідями — середня
#: позиція правильного кадру:
#:
#:   лише оригінал   8.0
#:   лише англійська 5.1
#:   максимум        6.4   ← було; найгірше з усіх злиттів
#:   RRF рівний      5.7
#:   RRF 1:3         5.1   ← не гірше за англійську, але зберігає повноту
#:
#: Оригінал лишається саме заради повноти: на «чорна жіноча сумочка» він дає
#: 9 позицію проти 11 в англійської, бо переклад втрачає відтінок.
CANONICAL_WEIGHT = 3.0

#: Наскільки канонічна фраза має узгоджуватися з оригіналом, щоб їй давали
#: потрійну вагу.
#:
#: Потреба знайдена на запиті «полуниця»: парсер переклав його як **«a bench»**
#: («полуниці» — як «a fence»). Далі вигаданий переклад важив утричі більше за
#: оригінал, і пошук чесно шукав лавки. Причому SigLIP українське слово
#: заземлює ДОБРЕ — сирий текст без розбору дає 94.5% на потрібному кадрі.
#: Тобто шкодив саме переклад, а вага його ще й підсилювала.
#:
#: Згода міряється тією самою моделлю, що вже завантажена: SigLIP
#: мультимовний, тож «полуниця» і «strawberry» лежать поруч, а «полуниця» і
#: «a bench» — ні. Нової залежності це не додає.
#:
#: Виміряно на 10 парах:
#:
#:   полуниця ↔ a bench      0.456   ← вигадка
#:   полуниці ↔ a fence      0.656   ← вигадка
#:   полуниця ↔ a shelf      0.631   ← контроль
#:   ─────────────────────── розрив
#:   ... з дитиною ↔ ...     0.736   ← правильний, найгірший із правильних
#:   полуниця ↔ strawberry   0.808
#:   чоловік в окулярах ↔ …  0.921
#:
#: ⚠️ Межа виведена з ДЕСЯТИ вимірів і лежить у розриві 0.66–0.74. За власним
#: правилом проєкту (ризик 10) такий поріг міряє удачу, доки його не
#: перевірено ширше. Тому наслідок навмисно мʼякий: канонічний канал не
#: викидається, а лише втрачає ПЕРЕВАГУ й важить нарівні з оригіналом. Хибне
#: спрацювання тоді коштує мало, а виміряний випадок виправляється.
TRANSLATION_AGREEMENT_MIN = 0.70

#: Ваги зворотного звʼязку Rocchio: `Q' = Q_text + 0.75·Q_pos − 0.25·Q_neg`.
#:
#: Найчастіший слідчий сценарій — «знайди ще таких самих»: людина бачить один
#: правильний кадр і хоче решту. Текстом це часто не формулюється взагалі
#: («оцей чоловік», «така сама сумка»), а показати пальцем можна завжди.
#:
#: Позитив важить утричі більше за негатив, і це не симетрія навпаки, а
#: властивість задачі: «схоже на ЦЕ» — твердження про ціль, а «не таке, як
#: ТЕ» — лише про один із багатьох способів помилитися. Рівна вага дала б
#: одному відкинутому кадру стільки ж голосу, скільки самому зразку.
ROCCHIO_POSITIVE = 0.75
ROCCHIO_NEGATIVE = 0.25


def _rocchio(
    vector: "np.ndarray",
    positive: "list[np.ndarray]",
    negative: "list[np.ndarray]",
) -> "np.ndarray":
    """Зсунути запит до відмічених зразків і від відкинутих.

    Вектори зображень мовно нейтральні, тож зсув однаково правильний для
    будь-якого текстового каналу — і застосовується до всіх.

    Результат нормується: далі він іде в косинусну міру, де довжина не має
    значення, зате впливає на калібровану впевненість. Без нормування
    «знайти схоже» тихо зсувало б усі показані відсотки.
    """
    shifted = np.asarray(vector, dtype=np.float32).copy()
    if positive:
        shifted += ROCCHIO_POSITIVE * np.mean(positive, axis=0).astype(np.float32)
    if negative:
        shifted -= ROCCHIO_NEGATIVE * np.mean(negative, axis=0).astype(np.float32)
    norm = float(np.linalg.norm(shifted))
    return shifted / norm if norm > 0 else np.asarray(vector, dtype=np.float32)


def _with_context(region, factor: float = DETECTION_CONTEXT):
    """Розширити рамку навколо її центру, лишаючись у межах кадру.

    Показується при цьому ОРИГІНАЛЬНА рамка детектора — розширена потрібна
    лише для оцінювання. Слідчому має бути видно, де саме обʼєкт, а не де
    закінчується контекст, який ми додали моделі.
    """
    from vsearch.represent.tiling import Region

    cx, cy = region.x + region.w / 2, region.y + region.h / 2
    w, h = min(1.0, region.w * factor), min(1.0, region.h * factor)
    x = min(max(0.0, cx - w / 2), 1.0 - w)
    y = min(max(0.0, cy - h / 2), 1.0 - h)
    return Region(x, y, w, h, kind=region.kind, label=region.label)


def _distinct_instances(hits: list[Hit], conditions, limit: int) -> list[Hit]:
    """До `limit` РІЗНИХ підтверджених екземплярів сутності в кадрі.

    «Різних» — за перекриттям рамок: одна людина, знайдена вісьмома плитками,
    це один екземпляр, а не вісім. Без цього «дві дівчини» задовольнялося б
    однією дівчиною, порахованою двічі, і кількість у запиті нічого не
    означала б.

    Порядок відбору той самий, що й для однієї рамки: серед упевнених —
    найпомітніші.
    """
    from vsearch.represent.tiling import Region

    # Плитка НЕ є свідченням окремого екземпляра. Сітка кладеться за
    # геометрією кадру, а не за його вмістом: дві сусідні плитки різні
    # ЗАВЖДИ, хоч би що на них було. Виміряно на PHOTO-2026-06-21-17-46-28:
    # дві плитки зі спільним центром (IoU 0.00, Δцентрів 0.00) — це одна
    # дівчина, розрізана сіткою на голову й тіло, а рахувалися вони як дві.
    # Отже рахувати екземпляри можна лише регіонами, які запропонував
    # детектор, бо тільки вони походять із вмісту.
    objects = [h for h in hits if h.payload.get("region_type") == "object"]
    hits = objects or hits

    scored = [(h, _facet_confidence(h.payload, conditions)) for h in hits]
    known = [(h, c) for h, c in scored if c is not None]
    if known:
        top = max(c for _, c in known)
        confident = [h for h, c in known if c >= top - ENTITY_FACET_MARGIN]
        # «Найпомітніша серед підтверджених» — саме в такому порядку. Площа
        # обирає ділянку лише ПІСЛЯ того, як фасет підтвердив, що там є те,
        # що запит назвав.
        confident.sort(key=lambda h: -float(h.payload.get("area_ratio") or 0.0))
    else:
        # Підтверджувати нічим: сутність не дала жодної фасетної умови
        # («соняшник» розбирається в сутність без класу й ознак). Тоді
        # найбільша ділянка не стверджує НІЧОГО — на запит «соняшник» нею
        # виявився мангал, найбільший обʼєкт кадру. Лишається єдине наявне
        # свідчення — схожість самої ділянки із запитом.
        confident = sorted(hits, key=lambda h: -h.score)

    picked: list[Hit] = []
    boxes: list[Region] = []
    for hit in confident:
        bbox = hit.payload.get("bbox")
        if not bbox:
            continue
        try:
            region = Region(*bbox, kind="region")
        except ValueError:
            continue
        if any(region.iou(other) >= DISPLAY_IOU for other in boxes):
            continue
        picked.append(hit)
        boxes.append(region)
        if len(picked) >= limit:
            break
    return picked or confident[:limit]


def _prominent(hits: list[Hit], conditions) -> Hit:
    """Найпомітніша ділянка серед тих, що впевнено підтверджують сутність.

    «Впевнено» — фасет не гірший за найкращий на `ENTITY_FACET_MARGIN`.
    «Найпомітніша» — найбільша за площею. Саме її людина назве відповіддю на
    питання «де тут жінка», навіть якщо дрібніша ділянка трохи схожіша на сам
    текст запиту.
    """
    scored = [(h, _facet_confidence(h.payload, conditions)) for h in hits]
    known = [(h, c) for h, c in scored if c is not None]
    if not known:
        return hits[0]
    top = max(c for _, c in known)
    confident = [h for h, c in known if c >= top - ENTITY_FACET_MARGIN]
    return max(confident, key=lambda h: float(h.payload.get("area_ratio") or 0.0))


def _collapse_shots(results: list["SearchResult"]) -> list["SearchResult"]:
    """Лишити по одному найкращому кадру на сцену, зберігши порядок.

    Сцена — `shot_id`, проставлений при відборі ключових кадрів: сусідні кадри
    без зміни плану належать одній. Для знімків `shot_id` немає, і вони
    проходять недоторканими.

    Скільки кадрів сцени відповіли — не викидається, а йде в провенанс: це
    відповідь на питання «як довго це тривало», яке в матеріалах справи важить
    більше за сам факт збігу.
    """
    kept: list[SearchResult] = []
    seen: dict[str, SearchResult] = {}
    for result in results:
        shot = result.provenance.get("shot_id") or result.matched_attrs.get("shot_id")
        if not shot:
            kept.append(result)
            continue
        first = seen.get(shot)
        if first is None:
            seen[shot] = result
            kept.append(result)
            first = result
        count = int(first.matched_attrs.get("shot_frames", 1))
        first.matched_attrs = {**first.matched_attrs, "shot_frames": count + (first is not result)}
    return kept


def _fuse_evidence(results: list["SearchResult"]) -> list["SearchResult"]:
    """Порядок за злиттям ДВОХ свідчень: схожості з текстом і впевненості в
    названих сутностях.

    Шкали несумісні, і це видно з даних: текстова впевненість гуляє від 0.5%
    до 67%, фасетна — від 13% до 98%, і між ними немає спільної одиниці.
    Добуток тому не годиться: ширший діапазон просто глушить вужчий, і
    порядок лишається таким, ніби фасета немає. Ранги від діапазону не
    залежать — це той самий інструмент, яким уже зливаються два мовні канали
    (ADR-013), і з тієї самої причини.

    Виміряно на золотому наборі (середня позиція цілі / Recall@5 / пасток у
    топ-5): лише текст 2.6 / 0.79 / 9, лише фасет 3.1 / 0.69 / 5, добуток
    2.6 / 0.80 / 10, **ранги 2.4 / 0.80 / 8** — краще за всіма трьома.

    Практично це знімає випадок, з якого все почалося: дівчина в купальнику,
    підтверджена фасетом на 95%, більше не стоїть нижче за кадр із фасетом
    23% лише тому, що текстовий косинус там на волосину вищий.
    """
    channels: list[list[int]] = []
    if any(r.entity_confidence is not None for r in results):
        channels.append(sorted(
            range(len(results)),
            key=lambda i: -(results[i].entity_confidence or 0.0),
        ))
    if any(r.detection_confidence is not None for r in results):
        channels.append(sorted(
            range(len(results)),
            key=lambda i: -(results[i].detection_confidence or 0.0),
        ))
    if not results or not channels:
        return sorted(results, key=lambda r: -r.probability)

    # Перший канал — ДОБУТОК схожості й підтвердження, а не сама схожість.
    # Добуток уже кращий за чистий текст (2.6 проти 3.0 середньої позиції), а
    # ранговий канал додає до нього те, чого добуток не бачить: кадр, де
    # фасет високий, а текст слабкий, підіймається за рангом фасета, навіть
    # якщо добуток лишився малим.
    by_text = sorted(
        range(len(results)),
        key=lambda i: -(results[i].probability * (results[i].entity_confidence or 1.0)),
    )
    score: dict[int, float] = {}
    for order in (by_text, *channels):
        for rank, index in enumerate(order):
            score[index] = score.get(index, 0.0) + 1.0 / (RRF_K + rank + 1)
    return [results[i] for i in sorted(score, key=lambda i: -score[i])]


def _facet_confidence(payload: dict, conditions) -> float | None:
    """Наскільки впевнено точка задовольняє УМОВИ своєї сутності.

    Береться найслабша з умов: сутність підтверджена настільки, наскільки
    підтверджена її найгірше обґрунтована частина — та сама логіка, що й у
    злитті сутностей у кадр.

    Заперечна умова читається як доповнення: якщо фасет «чоловік» має 0.2, то
    впевненість у «не чоловік» дорівнює 0.8.
    """
    parts: list[float] = []
    for key, value in conditions:
        score = payload.get(f"{key}_score")
        if score is None:
            score = payload.get(f"{key}_confidence")
        if score is None:
            continue
        parts.append(1.0 - float(score) if value is False else float(score))
    return min(parts) if parts else None


def _channel_weights(
    count: int, canonical: int = -1, weight: float = CANONICAL_WEIGHT
) -> list[float]:
    """Ваги каналів: канонічний важить більше, решта — по одиниці.

    Канонічним НЕ завжди є переклад: коли запит уже англійською, канонічний —
    сам оригінал, а «переклад» у кращому разі його повторює.

    `weight` знижується до 1.0, коли переклад не узгоджується з оригіналом
    (див. `TRANSLATION_AGREEMENT_MIN`): вага має спиратися на довіру, а не
    видаватися каналу за самим його статусом.
    """
    index = canonical % count if count else 0
    return [weight if i == index else 1.0 for i in range(count)]


def _rrf_by_frame(
    runs: list[list[Hit]], canonical: int = -1, weight: float = CANONICAL_WEIGHT
) -> dict[str, float]:
    """RRF-оцінка кожного КАДРУ за кількома ранжуваннями регіонів.

    Кадр отримує ранг за своїм найкращим регіоном у кожному каналі. Інакше
    один деталізований кадр, чиї тридцять три плитки посіли перші тридцять
    три позиції, вніс би в суму стільки, скільки тридцять три різні кадри.
    """
    scores: dict[str, float] = {}
    for run, weight in zip(runs, _channel_weights(len(runs), canonical, weight)):
        seen: set[str] = set()
        rank = 0
        for hit in run:
            key = hit.payload.get("frame_id", "")
            if not key or key in seen:
                continue
            seen.add(key)
            scores[key] = scores.get(key, 0.0) + weight / (RRF_K + rank + 1)
            rank += 1
    return scores


def _fuse_rrf(
    runs: list[list[Hit]], limit: int, canonical: int = -1,
    weight: float = CANONICAL_WEIGHT,
) -> list[Hit]:
    """Злити кілька ранжувань за ПОЗИЦІЯМИ, а не за величиною оцінки.

    Раніше злиття брало максимум косинуса по каналах, і це виявилося хибним
    за побудовою: абсолютний рівень косинуса залежить від самого тексту
    запиту, а не лише від того, наскільки він підходить кадру. Запит, чий
    текстовий вектор лежить у щільнішій області простору, дає вищі бали
    ВСЬОМУ — і забирає максимум у кожного кандидата, повністю глушачи другий
    канал.

    Виміряно на «дівчина в синій спідниці» (три фото, максимум по регіонах):

        фото                          укр      англ
        дві сині спідниці            10.7%    38.1%   ← потрібне
        блакитна сорочка             45.6%    16.4%
        блакитна сорочка             53.1%     5.1%

    Англійський канал упорядковує правильно, український — точно навпаки, і
    максимум брав саме його. Ранги ж від абсолютного рівня не залежать: канал
    без розрізнювальної здатності дає майже випадковий порядок і тому не може
    систематично перебивати канал, який розрізняє.
    """
    scores: dict[str, float] = {}
    best: dict[str, Hit] = {}
    from_canonical: set[str] = set()
    weights = _channel_weights(len(runs), canonical, weight)
    canonical_run = canonical % len(runs) if runs else 0
    for index, (run, weight) in enumerate(zip(runs, weights)):
        for rank, hit in enumerate(run):
            key = hit.payload.get("frame_id") or hit.point_id
            scores[key] = scores.get(key, 0.0) + weight / (RRF_K + rank + 1)
            # Представник кадру — ділянка, яку обрав КАНОНІЧНИЙ канал. Раніше
            # бралася ділянка з найвищим косинусом по всіх каналах, і далі
            # вона оцінювалася канонічним текстом — тобто міряли одну
            # ділянку, а обирали іншу. На «чоловік в синій сорочці» це
            # опускало правильний кадр із 52% до 5%: український канал обрав
            # плитку з тлом, а англійський обрав би сам обʼєкт.
            is_canonical = index == canonical_run
            if key not in best or (is_canonical and key not in from_canonical):
                best[key] = hit
            elif is_canonical and hit.score > best[key].score:
                best[key] = hit
            elif not is_canonical and key not in from_canonical and hit.score > best[key].score:
                best[key] = hit
            if is_canonical:
                from_canonical.add(key)
    order = sorted(scores, key=lambda k: -scores[k])[:limit]
    return [best[k] for k in order]


def _aggregate(
    frame_hits: list[Hit],
    region_hits: list[Hit],
    limit: int,
    *,
    scope: str = SCOPE_AUTO,
) -> list[Hit]:
    """Звести кадри й регіони до одного ранжованого переліку кадрів.

    Оцінка кадру — максимум по його регіонах. Вимір на наборі clutter показав,
    що простий максимум перевершує зважені альтернативи (середнє з кадром,
    другий за величиною, softmax) на дрібних цілях.

    У режимі `auto` кадровий вектор додається ЛИШЕ тоді, коли жоден регіон
    цього кадру не спливнув. Тобто регіони вирішують ранжування, а кадри
    доповнюють повноту — і не можуть підняти кадр, чиї обʼєкти запиту не
    відповідають. Саме цим лікується звʼязування (див. SCOPE_* вище).

    Один кадр зустрічається у видачі рівно раз: без згортання кадр із чотирма
    плитками зайняв би чотири місця в топі, витіснивши інші активи.
    """
    best: dict[str, Hit] = {}
    #: Усі регіони кожного кадру — щоб показати їх усі, а не лише переможця.
    all_hits: dict[str, list[Hit]] = {}

    def offer(hit: Hit, *, only_if_absent: bool = False, collect: bool = True) -> None:
        key = hit.payload.get("frame_id") or hit.point_id
        if collect:
            all_hits.setdefault(key, []).append(hit)
        current = best.get(key)
        if current is None:
            best[key] = hit
        elif not only_if_absent and hit.score > current.score:
            best[key] = hit

    if scope != SCOPE_FRAMES:
        for hit in region_hits:
            offer(hit)

    if scope == SCOPE_FRAMES or scope == SCOPE_BOTH:
        for hit in frame_hits:
            offer(hit, collect=False)
    elif scope == SCOPE_AUTO:
        for hit in frame_hits:
            offer(hit, only_if_absent=True, collect=False)

    ordered = sorted(best.values(), key=lambda h: -h.score)[:limit]
    for hit in ordered:
        key = hit.payload.get("frame_id") or hit.point_id
        hit.siblings = all_hits.get(key, [])
    return ordered


@dataclass
class RegionHit:
    """Один регіон кадру, що відповів на запит."""

    bbox: tuple[float, float, float, float]
    score: float
    kind: str = "region"
    label: str = ""
    #: Сутність запиту, яку підтверджує ця рамка («жінка», «дитина»).
    entity: str = ""
    #: Впевненість саме в цій сутності — за фасетами, не за текстом запиту.
    entity_score: float | None = None
    #: Калібрована ймовірність саме ЦІЄЇ ділянки.
    #:
    #: Потрібна окремо від `SearchResult.probability`, бо після переходу на
    #: піксельні плитки кадр несе до шести рамок, і без числа на кожній
    #: неможливо сказати, котра з них власне знахідка. Косинус для цього не
    #: годиться: різниця 0.19 проти 0.10 виглядає дрібною, а це 100% проти 1%.
    probability: float = 0.0


@dataclass
class SearchResult:
    """Одна знахідка з поясненням, чому вона знайдена."""

    asset_id: str
    frame_id: str
    score: float
    #: Ймовірність відповідності за калібровкою моделі, у межах [0, 1].
    #: Саме її варто показувати людині: косинус без калібровки нічого не каже.
    probability: float = 0.0
    path: str = ""
    media_type: str = "image"
    ts_ms: int | None = None
    #: Рамка НАЙКРАЩОГО регіону — за ним кадр і ранжується.
    bbox: tuple[float, float, float, float] | None = None
    #: УСІ регіони цього кадру, що відповіли на запит, від кращого до гіршого.
    #:
    #: Ранжування лишає один хіт на кадр, інакше знімок із чотирма плитками
    #: зайняв би чотири місця в топі. Але показувати теж один — помилка: на
    #: фото з двома людьми в окулярах знайдено обох, а рамка малювалася одна,
    #: і виглядало це так, ніби другу людину система не побачила.
    regions: list["RegionHit"] = field(default_factory=list)
    #: Наскільки підтверджені НАЗВАНІ в запиті сутності — найслабша з них.
    #: Окремо від `probability`, бо міряє інше: `probability` каже, наскільки
    #: кадр схожий на текст запиту, ця — наскільки в ньому підтверджено те,
    #: що запит назвав. Порядок видачі зливає обидві за рангами.
    entity_confidence: float | None = None
    #: Впевненість обумовленої детекції — третє свідчення, коли уточнення
    #: увімкнене. Як і решта, бере участь у злитті РАНГАМИ, а не замінює
    #: порядок собою: заміна коштувала падіння Recall@5 з 0.79 до 0.54.
    detection_confidence: float | None = None
    #: Які фасети збіглися і з якою впевненістю (M3–M4).
    matched_attrs: dict[str, Any] = field(default_factory=dict)
    #: Версії моделей, якими отримано результат (п.7).
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def is_confident(self) -> bool:
        return self.probability >= MIN_PROBABILITY

    def explain(self) -> str:
        """Коротке пояснення для UI та матеріалів справи."""
        parts = [f"впевненість {self.probability:.0%} (cos {self.score:.3f})"]
        if len(self.regions) > 1:
            parts.append(f"знайдено ділянок: {len(self.regions)}")
        if self.bbox:
            parts.append(f"рамка {tuple(round(v, 3) for v in self.bbox)}")
        if self.ts_ms is not None:
            parts.append(f"момент {self.ts_ms / 1000:.1f}с")
        if self.matched_attrs:
            parts.append("фасети: " + ", ".join(f"{k}={v}" for k, v in self.matched_attrs.items()))
        return "; ".join(parts)


@dataclass
class SearchResponse:
    query: str
    results: list[SearchResult]
    latency_ms: float
    profile: str
    #: Розібраний запит: must / must_not / relations.
    parsed: StructuredQuery | None = None
    #: Чи довелося зняти жорсткі фільтри, щоб узагалі щось знайти.
    degraded: bool = False
    #: Пояснення для користувача, коли пошук деградував.
    notice: str = ""
    #: Скільки кадрів узагалі задовольнили жорсткі умови (не лише показаних).
    #: Без цього числа неможливо здогадатися, що потрібний кадр існує, просто
    #: стоїть за межею ліміту: «чоловік та жінка» дає 42 збіги, а показується 8.
    total_matches: int | None = None

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self):
        return iter(self.results)

    @property
    def is_strict(self) -> bool:
        return not self.degraded

    @property
    def asset_ids(self) -> list[str]:
        """Ранжований перелік активів — саме його очікують метрики."""
        seen: list[str] = []
        for result in self.results:
            if result.asset_id not in seen:
                seen.append(result.asset_id)
        return seen


class Searcher:
    """Тримає модель і зʼєднання відкритими між запитами."""

    def __init__(
        self,
        profile: Profile | None = None,
        store: VectorStore | None = None,
        catalog: Catalog | None = None,
        embedder: Siglip2Embedder | None = None,
    ) -> None:
        self.profile = profile or get_profile()
        self.store = store or VectorStore()
        self.catalog = catalog or Catalog()
        self.embedder = embedder or Siglip2Embedder(profile=self.profile)
        self._known_keys: set[str] | None = None
        self._last_unknown: list[str] = []
        #: Розріджений вектор поточного запиту. None означає «лексичний шар
        #: вимкнено профілем», порожній — «у запиті немає жодного терма».
        self._sparse_query: Any = None
        #: Який із текстових каналів канонічний. Від нього залежить не лише
        #: вага в злитті, а й ВИБІР представника кадру: оцінювати ділянку,
        #: обрану іншим каналом, — те саме, що міряти не той обʼєкт.
        self._canonical_index: int = 0
        self._canonical_weight: float = CANONICAL_WEIGHT
        self._last_total: int | None = None

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        query_filter: Any = None,
        category: str | None = None,
        scope: str = SCOPE_AUTO,
        structured: StructuredQuery | None = None,
        parse: bool = True,
        extra_excluded: list[list[tuple[str, Any]]] | None = None,
        min_strict: int = MIN_STRICT_RESULTS,
        min_probability: float = 0.0,
        #: Кадри, відмічені слідчим як «таке саме» і «не таке» (`frame_id` з
        #: попередньої видачі). Зсувають запит за Rocchio — див. `_rocchio`.
        similar_to: Sequence[str] = (),
        unlike: Sequence[str] = (),
        use_template: bool | None = None,
    ) -> SearchResponse:
        """Текстовий запит → ранжовані кадри.

        Два проходи, і другий існує через конкретний спосіб зламатися:
        LLM-парсер на розмитому запиті може згенерувати надто жорсткі умови,
        і система поверне порожньо. Порожня видача гірша за приблизну — тому
        якщо жорсткий прохід дав менше `min_strict` результатів, повторюємо
        суто щільним пошуком і чесно про це повідомляємо.
        """
        started = time.perf_counter()
        self._last_unknown = []
        self._last_total = None
        parsed = structured if structured is not None else self._parse(query, parse)

        text = parsed.query_en or query
        if use_template is None:
            # Шаблон «this is a photo of …» застосовується ЗАВЖДИ.
            #
            # Раніше тут стояла евристика «лише короткі й не композитні»,
            # і вона була не просто неточною, а шкідливою. Вимір на 16 запитах
            # по реальному індексу: шаблон кращий у 15 із них, середня
            # впевненість 47.6% проти 14.3%. На окремих запитах різниця
            # стократна — «велосипед» дає 24.3% із шаблоном і 0.04% без.
            #
            # Причина в тому, як SigLIP тренували: підписи мали саме таку
            # форму, тож голий іменник для моделі поза розподілом.
            use_template = True
        # Шукаємо ОБОМА формулюваннями — оригіналом і англійською нормалізацією.
        #
        # Вибирати одне неправильно, і це виміряно на реальному індексі.
        # Оригінал кращий у 9 запитах із 10 (45.5% проти 31.2%): SigLIP 2
        # тренований на 109 мовах, і переклад втрачає інформацію —
        # «текст на екрані» дає 98.6%, «text on screen» лише 58.2%.
        #
        # Але є винятки, де переклад рятує: івритське «פנים» двозначне
        # («обличчя» і «всередині»), тож англійська дає 82.8% проти 13.9%.
        #
        # Тому не вибираємо один, а зливаємо обидва — але за РАНГАМИ, а не
        # за величиною косинуса, і з більшою вагою канонічного каналу
        # (див. CANONICAL_WEIGHT). Ціна — один текстовий ембединг і один
        # додатковий пошук.
        texts = [query]
        if parsed.query_en and parsed.query_en.strip().lower() != query.strip().lower():
            texts.append(parsed.query_en)

        # Лексичний вектор будується з ОБОХ формулювань: номер чи прізвище
        # переклад не змінює, а от «München» проти «Мюнхен» дає різні терми,
        # і втратити одне з них означало б втратити половину шансів на збіг.
        self._sparse_query = (
            lexical.build_query(" ".join(texts))
            if (self.profile.use_ocr or self.profile.use_captions) else None
        )

        # Якщо запит уже англійською, переклад не додає нічого, а зіпсувати
        # може: на «Champagne» парсер видав «a champagne», і ця неграматична
        # форма дала 9% там, де саме слово дає 53%. Канонічним каналом тоді
        # лишається оригінал.
        canonical_index = 0 if parsed.language == "en" else len(texts) - 1
        self._canonical_index = canonical_index
        matrix = self.embedder.embed_texts(texts, use_template=use_template)
        # Довіра до перекладу вимірюється ДО того, як він отримає перевагу.
        # Парсер може перекласти «полуниця» як «a bench», і тоді потрійна вага
        # канонічного каналу не підсилює сигнал, а підсилює вигадку.
        self._canonical_weight = CANONICAL_WEIGHT
        translation_notice = ""
        if canonical_index != 0:
            agreement = float(matrix[0] @ matrix[canonical_index])
            if agreement < TRANSLATION_AGREEMENT_MIN:
                # Переклад ВІДКИДАЄТЬСЯ, а не зважується. Спершу тут стояло
                # зниження ваги до рівної — і на «полуниця» це нічого не
                # змінило, бо переклад бере участь не лише у злитті: ним
                # переоцінюється верхівка, з нього рахується показана
                # впевненість і з нього береться слово для детекції. Кадр
                # виходив підписаний схожістю з «a bench».
                #
                # Хибний переклад — не слабке свідчення, а свідчення ПРО ІНШЕ.
                # Тому канонічним лишається оригінал, як це вже робиться для
                # запитів, що й так англійською.
                bad = texts[canonical_index]
                texts = texts[:1]
                matrix = matrix[:1]
                canonical_index = 0
                self._canonical_index = 0
                # Разом із перекладом відкидається й СТРУКТУРА. Вона виведена
                # з тієї самої галюцинації: «полуниці» дало сутність
                # `building` (від «a fence»), і фільтр за будівлями лишався
                # навіть після заміни тексту. Якщо переклад не про запит, то й
                # розібрані з нього сутності не про нього.
                parsed = parsed.model_copy(update={
                    "query_en": texts[0], "must": [], "must_not": [],
                    "categories": [],
                })
                translation_notice = (
                    f"Переклад «{bad}» не узгоджений з оригіналом "
                    f"(згода {agreement:.2f}) — його відкинуто, пошук іде за "
                    f"оригінальним формулюванням."
                )
                logger.info(
                    "переклад %r не узгоджений з %r (%.2f) — відкинуто",
                    bad, texts[0], agreement,
                )
        vector = matrix[0]
        alt_vectors = list(matrix[1:])

        # Зворотний звʼязок застосовується до ВСІХ каналів: вектор зображення
        # мовно нейтральний, тож зсув однаково правильний і для оригіналу, і
        # для перекладу. Застосувати лише до одного означало б розвести канали
        # в різні боки й зробити їхні ранги неспівставними.
        feedback_missing: list[str] = []
        feedback_applied = False
        if similar_to or unlike:
            pos, neg = self._feedback_vectors(similar_to, unlike, feedback_missing)
            if pos or neg:
                vector = _rocchio(vector, pos, neg)
                alt_vectors = [_rocchio(v, pos, neg) for v in alt_vectors]
                feedback_applied = True

        category_unknown = ""
        if category is not None:
            if query_filter is not None:
                raise ValueError("вкажіть або category, або query_filter, не обидва")
            key = _category_key(category)
            # Фільтр на неіснуючий ключ не помиляється — він просто ніколи не
            # збігається. Порожня видача без пояснення неможливо відрізнити
            # від чесної відсутності матеріалу, а це найдорожчий різновид
            # помилки в цій системі.
            if key not in self.known_payload_keys():
                category_unknown = key
                logger.info("індекс не знає категорії %s — фільтр не застосовано", key)
            else:
                query_filter = build_filter(must=[(key, True)])

        if feedback_missing:
            translation_notice = (translation_notice + " " if translation_notice else "") + (
                f"Позначених кадрів немає в індексі: {len(feedback_missing)} — "
                f"їх не враховано у «знайти схоже»."
            )
        degraded, notice = False, translation_notice
        hits: list[Hit] = []
        # Глибина відбору навмисно більша за ліміт показу: далі верхівка
        # переоцінюється канонічним текстом, і кандидати нижче межі показу
        # часто піднімаються вище. Ліміт застосовується в самому кінці.
        # Глибина СТАЛА, а не похідна від ліміту показу. Раніше тут стояло
        # `max(limit, RESCORE_DEPTH)`, тобто запит на 60 результатів піднімав
        # 60 кандидатів, а на 10 — тридцять, і злиття за рангами давало РІЗНИЙ
        # порядок верхівки. Інваріант «менший ліміт не змінює порядок» це
        # ловить, але тримався він лише тому, що на 123 кадрах глибина
        # накривала майже весь корпус: на 384 кадрах порядок на limit=60 уже
        # інший, ніж на limit=30.
        #
        # Це та сама межа, що й у решті механізмів з ADR-015: вони не
        # ламаються, а тихо гіршають, коли вибірка перестає накривати індекс.
        #
        # Переоцінка коштує читання збережених векторів і множення, тож
        # глибина 120 замість 30 майже безкоштовна.
        depth = max(limit, MIN_CANDIDATE_DEPTH)
        if parsed.must or parsed.must_not or extra_excluded:
            hits = self._strict_pass(
                vector, parsed, depth, extra=query_filter,
                extra_excluded=extra_excluded, alt_vectors=alt_vectors,
            )
            # Поріг порівнюється з тим, скільки жорсткий прохід МІГ віддати,
            # а не зі скількома його попросили показати.
            #
            # Спершу тут стояло обмеження лімітом показу — щоб запит на 3
            # результати не падав у мʼякий прохід через те, що жорсткий чесно
            # віддав рівно 3. Але справжня причина була інша: глибина відбору
            # дорівнювала ліміту. Щойно глибина стала окремою (`depth`),
            # обмеження лімітом лишилося зайвим — і почало шкодити, бо
            # унеможливлювало явно заданий високий `min_strict`.
            if len(hits) < min(min_strict, depth):
                degraded = True
                notice = (
                    f"Жорсткий пошук дав {len(hits)} результатів. "
                    f"Застосовано мʼякий пошук без структурних обмежень."
                )
                if self._last_unknown:
                    notice += (
                        f" Індекс не знає фасетів: {', '.join(self._last_unknown)} — "
                        f"ці умови не застосовувалися."
                    )
        if not hits or degraded:
            hits = self._dense_pass(
                vector, depth, scope=scope, query_filter=query_filter,
                alt_vectors=alt_vectors,
            )

        info = self.embedder.info
        provenance = {
            "embed_model": info.model_name,
            "embed_repo": info.repo_id,
            "embed_revision": info.revision,
            "max_num_patches": info.max_num_patches,
            "device": info.device,
            "profile": self.profile.name,
            "scope": scope,
            "strict": not degraded,
        }
        # Канонічний канал — останній: `texts = [оригінал, query_en]`. Якщо
        # переклад збігся з оригіналом, канал один, і переоцінювати нічого.
        # Зі зворотним звʼязком переоцінка потрібна ЗАВЖДИ, навіть коли канал
        # один. Зсунутий за Rocchio вектор — це вже майже вектор ЗОБРАЖЕННЯ, а
        # калібровка сигмоїди виведена для косинусів «текст↔зображення». Без
        # переоцінки вся видача показувала 100%: шкала просто насичувалася.
        # Порядок при цьому лишається за зсунутим вектором — він і є те, що
        # слідчий попросив, — а число під фото знову означає відповідність
        # ТЕКСТУ запиту.
        hits = self._rescore_canonical(
            hits,
            matrix[canonical_index]
            if (len(matrix) > 1 or feedback_applied) else None,
        )
        if feedback_applied and not text.strip():
            notice = (notice + " " if notice else "") + (
                "Запит без тексту: порядок визначено лише схожістю на "
                "позначені кадри, а відсоток під фото не є відповідністю тексту."
            )
        if category_unknown:
            notice = (notice + " " if notice else "") + (
                f"Індекс не знає категорії {category_unknown!r} — фільтр не "
                f"застосовано. Перелік доступних: vsearch categories."
            )
        results = [self._to_result(hit, provenance) for hit in hits]
        for result in results:
            text_probability = float(self.embedder.probability(result.score))
            for region in result.regions:
                region.probability = float(self.embedder.probability(region.score))

            # Впевненість у кадрі — ДОБУТОК двох свідчень: наскільки кадр
            # схожий на текст запиту і наскільки підтверджені названі в ньому
            # обʼєкти. Обидва потрібні: схожість без підтвердженої сутності —
            # це «щось схоже десь у кадрі», а підтверджена сутність без
            # схожості — «жінка є, але запит був не про це».
            #
            # Виміряно на золотому наборі (середня позиція цілі): лише текст
            # 3.0, лише фасет 3.2, **добуток 2.8**. Recall@5 і Recall@10 при
            # цьому не змінюються, тобто виграш чистий.
            #
            # Це також знімає скаргу, яка й привела сюди: кадр із дівчиною в
            # купальнику, підтвердженою на 64%, більше не стоїть нижче за
            # кадр, де підтверджено на 5%, лише через те, що текстовий
            # косинус там на волосину вищий.
            entity_scores = [
                r.entity_score for r in result.regions
                if r.entity and r.entity_score is not None
            ]
            result.probability = text_probability
            result.entity_confidence = min(entity_scores) if entity_scores else None

        # Лексичний шар — ПІСЛЯ обчислення ймовірностей, і це не дрібниця:
        # спершу він стояв раніше, і наступний цикл перезаписував його
        # впевненість назад із косинуса. Точний збіг рядка мовчки зникав.
        results = _fuse_evidence(results)
        results = self._apply_lexical(results, query, parsed, query_filter)
        # Ліміт показу — ОСТАННІЙ крок. До нього всі етапи працюють на повній
        # глибині, інакше кількість запитаних результатів мовчки визначала б,
        # які кандидати взагалі дійдуть до переоцінки.
        # Сцена займає ОДНУ позицію. Без цього ролик із двадцяти ключових
        # кадрів забирає всю верхівку: чотири з шести перших місць на «dog»
        # були кадрами одного відео з інтервалом у дві секунди. Для слідчого
        # це не двадцять знахідок, а одна подія — і решта видачі при цьому
        # витісняється.
        #
        # Це та сама причина, з якої ранг рахується по КАДРАХ, а не по
        # регіонах (ADR-013): свідчення має важити один раз.
        results = _collapse_shots(results)
        pool = [r.score for r in results]
        results = results[:limit]

        # Слабкі збіги ПОКАЗУЮТЬСЯ, лише позначаються. Спокуса відсіяти їх
        # велика — на запит «білий велосипед» система підсвічувала людей у
        # білому з косинусом 0.076. Але той велосипед у кадрі БУВ, просто
        # дрібний і на фоні: поріг приховав би справжню знахідку.
        #
        # У розслідуванні прихований доказ коштує дорожче за зайвий кандидат,
        # тому вирішує людина, а система лише чесно каже, наскільки впевнена.
        # Абсолютна межа сама по собі непридатна, і це вимір, а не здогад.
        # «банка пепсі» дає 0.24%, «pepsi can» — 10.3%, ТОЙ САМИЙ кадр і та
        # сама плитка з банкою. Різниця лише в тому, як парсер сформулював
        # канонічну фразу: «a can of pepsi» проти «a pepsi can». Тобто число,
        # яким вирішується видимість, залежить від формулювання запиту, а не
        # від того, що в кадрі.
        #
        # Наслідок був найгіршого класу: система писала «найімовірніше, такого
        # в індексі немає» про кадр, що стояв другим.
        #
        # Тому до абсолютної межі додано ВІДНОСНУ: чи вирізняється кандидат на
        # тлі власного пулу цього ж запиту. Вона від формулювання не залежить,
        # бо і кандидат, і пул міряються одним текстом. Межа 2σ — конвенційна
        # межа «це не шум», а не підібране під корпус число; підбирати його на
        # шести вимірах було б рівно тією помилкою, що вже описана в ризику 10.
        #
        # Виміряно: присутнє дає 2.1–3.1σ, відсутнє («гелікоптер», «танк») —
        # 0.14σ.
        strong_relative = _stands_out(pool)
        weak = [
            r for r in results
            if r.probability < min_probability and r.score < strong_relative
        ]
        if min_probability > 0:
            hidden = {id(r) for r in weak}
            results = [r for r in results if id(r) not in hidden]
        if self._last_total and not degraded and self._last_total > len(results):
            notice = (notice + " " if notice else "") + (
                f"Умовам відповідає кадрів: {self._last_total}, показано {len(results)}. "
                f"Збільште ліміт, щоб побачити решту."
            )
        if weak:
            best_weak = max(r.probability for r in weak)
            if results:
                notice = (notice + " " if notice else "") + (
                    f"Приховано збігів із впевненістю нижче {min_probability:.0%}: "
                    f"{len(weak)} (найкращий {best_weak:.1%})."
                )
            else:
                notice = (
                    f"Нічого не знайдено. Усі {len(weak)} кандидатів мають впевненість "
                    f"нижче {min_probability:.0%} (найкращий {best_weak:.1%}) — "
                    f"найімовірніше, такого в індексі немає."
                )

        return SearchResponse(
            query=query,
            results=results,
            latency_ms=(time.perf_counter() - started) * 1000,
            profile=self.profile.name,
            parsed=parsed,
            degraded=degraded,
            notice=notice,
            total_matches=None if degraded else self._last_total,
        )

    def _parse(self, query: str, enabled: bool) -> StructuredQuery:
        if not enabled:
            return EMPTY.model_copy(update={"query_en": query})
        from vsearch.search.parse import get_parser

        return get_parser().parse_or_empty(query)

    def _dense_pass(
        self,
        vector,
        limit: int,
        *,
        scope: str,
        query_filter: Any = None,
        alt_vectors: list | None = None,
    ) -> list[Hit]:
        """Щільний пошук без структурних умов."""
        if alt_vectors:
            runs = [
                self._dense_pass(candidate, limit, scope=scope, query_filter=query_filter)
                for candidate in [vector, *alt_vectors]
            ]
            return _fuse_rrf(runs, limit, canonical=self._canonical_index,
                             weight=self._canonical_weight)

        oversampling = float(self.profile.ann_oversampling)
        # Лексичний шар живе на кадрі: текст належить знімку, а не плитці.
        # Якщо в запиті немає жодного терма або колекція без розрідженого
        # індексу, `search_hybrid` вироджується у звичайний щільний пошук.
        depth = max(limit * 2, MIN_CANDIDATE_DEPTH)
        frame_hits = self.store.search(
            schema.FRAMES, vector, limit=depth,
            query_filter=query_filter, oversampling=oversampling,
        )
        region_hits = (
            self.store.search(
                schema.REGIONS, vector,
                limit=max(limit * REGION_FETCH_FACTOR, MIN_CANDIDATE_DEPTH),
                query_filter=query_filter, oversampling=oversampling,
            )
            # Оцінка, а не точний підрахунок: питання тут одне — «чи є
            # взагалі регіони», і заради нього пробігати найбільшу
            # колекцію на кожному запиті марно (ADR-015).
            if scope != SCOPE_FRAMES and self.store.count(schema.REGIONS, exact=False)
            else []
        )
        return _aggregate(frame_hits, region_hits, limit, scope=scope)

    def known_payload_keys(self, *, refresh: bool = False) -> set[str]:
        """Фасети, які індекс справді знає.

        Потрібне як запобіжник від розсинхрону: парсер видає класи обʼєктів зі
        свого переліку, а індекс знає лише ті категорії, які були пораховані.
        Умова на невідомий ключ не помиляється — вона просто ніколи не
        збігається, тож жорсткий прохід дає нуль і мовчки падає в мʼякий.
        Мовчазна деградація гірша за відсутність фільтра.
        """
        if refresh:
            self._known_keys = None
        if self._known_keys is None:
            # Спершу — ті, що мають payload-ІНДЕКС. Це точний перелік і
            # одна дешева відповідь на колекцію, а не вибірка: питання «чи
            # можна за цим фільтрувати ШВИДКО» має рівно цю відповідь.
            indexed: set[str] = set()
            for collection in (schema.REGIONS, schema.FRAMES):
                try:
                    info = self.store.client.get_collection(self.store.name(collection))
                    indexed |= {
                        k for k in (info.payload_schema or {})
                        if k.startswith(("cat_", "attr_")) and not k.endswith("_score")
                    }
                except Exception:  # noqa: BLE001 — колекції може не бути
                    pass

            # Вибірка лишається ДОПОВНЕННЯМ, а не заміною: користувацька
            # категорія, застосована до готового індексу, ще не має свого
            # payload-індексу, але фільтрувати за нею вже можна — просто
            # повільніше. Відкинути її означало б мовчки зламати заперечення
            # одразу після того, як слідчий її додав.
            # Джерело істини — сам ІНДЕКС, а не реєстр каталогу. Реєстр може
            # відставати (категорію додали й застосували, але не зберегли), і
            # тоді запобіжник відкинув би цілком робочу умову. Питання, на яке
            # ми відповідаємо, буквально таке: чи може цей фільтр колись
            # збігтися — а це видно лише в payload.
            keys: set[str] = set()
            for collection in (schema.REGIONS, schema.FRAMES):
                for payload in self.store.scroll_payloads(collection, limit=64):
                    keys |= {
                        k for k in payload
                        if k.startswith(("cat_", "attr_")) and not k.endswith("_score")
                    }
            self._known_keys = keys | indexed
        return self._known_keys

    def _split_conditions(
        self, conditions: list[tuple[str, Any]]
    ) -> tuple[list[tuple[str, Any]], list[tuple[str, Any]]]:
        """Розділити умови на ті, що фільтруються, і ті, що рахуються на льоту.

        Раніше друга група просто відкидалася, і запит мовчки втрачав ознаку:
        «рожеве плаття» перетворювалося на «плаття», бо рожевого не було в
        передобчисленні. Перелічити всі можливі ознаки неможливо, тож замість
        переліку — механізм: ознака стає прототипом із власного тексту й
        оцінює вже піднятих кандидатів.

        Фільтр лишається для того, що передобчислене: він звужує вибірку ДО
        пошуку по ANN, і на мільйонах це незамінне. Прототип на льоту працює
        по вже відібраних, тож звузити мільйони не може — але й не мусить.
        """
        known = self.known_payload_keys()
        # Перелік кешується на весь час життя пошуковця, тож категорія, додана
        # ПІСЛЯ першого запиту, лишалася б невидимою назавжди. Наслідок не
        # косметичний: невідомий ключ опускається до мʼякого підняття, і
        # заперечення за ним перестає бути точним — заборонений кадр пролізає
        # у видачу. Тому на промах перелік перечитується один раз.
        #
        # Сценарій цілком реальний: слідчий додає свою категорію в інтерфейсі
        # й одразу нею шукає.
        if any(k not in known for k, _ in conditions):
            known = self.known_payload_keys(refresh=True)
        if not known:  # індекс без фасетів — усе рахуємо на льоту
            return [], list(conditions)
        filterable = [(k, v) for k, v in conditions if k in known]
        on_the_fly = [(k, v) for k, v in conditions if k not in known]
        return filterable, on_the_fly

    def _refine_on_the_fly(
        self, hits: list[Hit], conditions: list[tuple[str, Any]]
    ) -> list[Hit]:
        """Врахувати ознаку, якої немає в індексі, — ПІДНЯТТЯМ, а не відсівом.

        Спокуса зробити тут фільтр велика, але вимір показав, що вона хибна:
        розривне правило на двохстах кандидатах межі не знаходить, і ознака або
        не звужує нічого (чорний, рожевий, червоний — 200 із 200), або відсікає
        майже все (білий — 1 із 200). Обидва результати гірші за відсутність
        ознаки.

        Тому оцінка прототипу додається до оцінки схожості. Кандидат із
        ознакою піднімається, кандидат без неї лишається у видачі нижче. Для
        розслідування це правильний компроміс: прихований доказ коштує дорожче
        за зайвого кандидата, а рішення ухвалює людина.

        Вартість — один текстовий ембединг на ознаку (~80 мс), а не на
        кандидата. Множення матриць по кандидатах — частки мілісекунди.
        """
        if not conditions or not hits:
            return hits
        vectors = [h.vector for h in hits if h.vector is not None]
        if len(vectors) != len(hits):
            logger.warning("немає векторів кандидатів — ознаки на льоту пропущено")
            return hits

        from vsearch.represent.prototypes import Prototype, PrototypeBank
        from vsearch.search.query_model import (
    ObjectClass,
            attribute_negatives, attribute_phrases, object_phrases,
        )

        bank = PrototypeBank(self.embedder)
        for key, value in conditions:
            name = key.split("_", 1)[1] if "_" in key else key
            phrases = (
                object_phrases(name) if key.startswith("cat_")
                else attribute_phrases(name, value)
            )
            # Негатив у межах категорії там, де він відомий: інакше прототип
            # міряє «чи це людина», а не «що на ній надіто» (див.
            # `_NEGATIVE_TEMPLATES`).
            bank.add(Prototype(
                name=key, positive=phrases,
                negative=() if key.startswith("cat_") else attribute_negatives(name),
            ))

        matrix = np.asarray(vectors, dtype=np.float32)
        scores = bank.score_many(matrix, [k for k, _ in conditions])
        boost = np.zeros(len(hits), dtype=np.float32)
        for name in scores:
            boost += scores[name].raw
            # Оцінка прототипу кладеться в ТЕ САМЕ поле payload, що й
            # передобчислені фасети. Далі вибір рамки, її підпис і впевненість
            # працюють без жодної окремої гілки: ознака з відкритого словника
            # нічим не відрізняється від закритої.
            #
            # Payload тут локальний — це копія, підняте з Qdrant для цього
            # запиту. В індекс нічого не пишеться.
            for hit, value in zip(hits, scores[name].probability):
                hit.payload[f"{name}_score"] = float(value)

        # Вага підняття підібрана так, щоб ознака впливала на порядок, але не
        # перекривала саму схожість: різниця косинусів лежить у тому ж
        # масштабі, що й оцінка пошуку, тож коефіцієнт близький до одиниці.
        for hit, delta in zip(hits, boost):
            hit.score = float(hit.score + ON_THE_FLY_WEIGHT * delta)
        return sorted(hits, key=lambda h: -h.score)

    def _strict_pass(
        self,
        vector,
        parsed: StructuredQuery,
        limit: int,
        *,
        extra: Any = None,
        extra_excluded: list[list[tuple[str, Any]]] | None = None,
        alt_vectors: list | None = None,
    ) -> list[Hit]:
        """Пошук зі звʼязаними умовами: кожна сутність шукається окремо.

        Умови однієї сутності застосовуються до ОДНІЄЇ точки — так Qdrant дає
        звʼязування безкоштовно. Потім множини кадрів перетинаються: «чоловік
        біля червоної машини» вимагає, щоб у кадрі знайшлися ОБИДВА регіони,
        а не один регіон, який водночас людина й автомобіль.

        Це і є той другий прохід, що закриває обмеження, зафіксоване на M4a.
        """
        oversampling = float(self.profile.ann_oversampling)
        fetch = STRICT_FETCH_LIMIT

        per_entity: list[dict[str, Hit]] = []
        siblings: dict[str, list[Hit]] = {}
        #: Кадри, де сутність є, але знайдено МЕНШЕ екземплярів, ніж просить
        #: запит: `frame_id -> (знайдено, треба)`. Це помітка провенансу, а не
        #: фільтр, — див. ADR-022.
        shortfall: dict[str, tuple[int, int]] = {}
        canonical_seen: set[str] = set()
        unknown: list[str] = []
        entity_labels: list[str] = []
        # ВІДКЛЮЧЕНО за виміром. Механізм правильний і безпечний — категорія
        # береться з ДОСЛІВНОГО збігу цілого слова, тож вигадати її неможливо
        # (на відміну від спроби довірити вибір парсерові). Але вмикати його
        # нема сенсу, доки самі категорії такі, як зараз.
        #
        # `cat_nudity` спрацьовує на 1823 регіонах із 3861, `cat_underwear` —
        # на 2180, і їхні оцінки насичені: 93–99% у всієї верхівки. Канал, що
        # каже майже те саме про майже все, не є свідченням. Підключений, він
        # опустив цільовий кадр із 2 позиції на 5, бо розмив порядок рівним
        # шумом.
        #
        # Причина глибша за цей випадок: категорія, для якої не вдалося
        # вивести поріг із корпусу, використовує власне рішення прототипа
        # (ADR-016), а воно широке. Раніше такі категорії просто не
        # записувалися — тепер вони є, але надто нечіткі.
        #
        # Розблокує це поріг за КВАНТИЛЕМ: «помітно більше за типове» існує
        # завжди, на відміну від розриву в розподілі. Це окрема робота з
        # власним виміром.
        category_terms: list[list[tuple[str, object]]] = []
        # Категорії сцени йдуть тими самими умовами, що й сутності: кожна
        # вимагає СВОГО регіону в кадрі. Купальник і людина можуть бути
        # різними ділянками, тож зливати їх в одну умову не можна.
        #
        # Досі це поле розбору не використовувалося взагалі: 28 із 36
        # обчислених категорій не мали жодного способу потрапити в запит.
        # «Дівчина в купальнику» зводилася до «дівчина плюс щось».
        # ВІДКЛЮЧЕНО після виміру. Задум правильний — 28 із 36 категорій не
        # мають іншого способу потрапити в запит, — але реалізація через
        # конʼюнкцію шкодить: парсер їх ВИГАДУЄ. На «дівчина в синій спідниці»
        # він видав `underwear, swimwear, outdoor`, і три неіснуючі умови
        # прибрали потрібне фото з видачі повністю. Recall@5 упав 0.90 → 0.77.
        #
        # Наступний крок — застосовувати категорії як ПІДНЯТТЯ, а не фільтр:
        # вигадана категорія тоді не може нічого виключити, а справжня все
        # одно допомагає порядку. Перелік і `category_conditions()` лишаються
        # готовими для цього.
        for entity, raw_conditions in zip(
            [*parsed.must, *([None] * len(category_terms))],
            [*parsed.entity_conditions(), *category_terms],
        ):
            conditions, on_the_fly = self._split_conditions(raw_conditions)
            unknown += [k for k, _ in on_the_fly]
            if not conditions and not on_the_fly:
                continue
            # Обидва канали й тут. Раніше жорсткий прохід ранжував відібране
            # ЛИШЕ оригінальним запитом, тож на «дівчина в синій спідниці»
            # фільтр залишав 97 кадрів зі 117 і впорядковував їх українським
            # каналом, який цього поняття не розрізняє. Переклад був
            # правильний, але до ранжування просто не доходив.
            entity_filter = build_bound_filter(conditions) if conditions else None
            runs = []
            for candidate in [vector, *(alt_vectors or [])]:
                found = self.store.search(
                    schema.REGIONS, candidate, limit=fetch,
                    query_filter=entity_filter,
                    oversampling=oversampling,
                    with_vectors=bool(on_the_fly),
                )
                runs.append(self._refine_on_the_fly(found, on_the_fly))

            # Ранг рахується по КАДРАХ, а не по регіонах: кадр із двадцятьма
            # плитками інакше зайняв би двадцять перших позицій і роздув би
            # свій внесок у злиття.
            fused = _rrf_by_frame(runs, canonical=self._canonical_index,
                                  weight=self._canonical_weight)
            best: dict[str, Hit] = {}
            #: Усі ділянки кадру, що пройшли фільтр ЦІЄЇ сутності — з них
            #: обирається та, яку показати.
            shown: dict[str, list[Hit]] = {}
            canonical_run = self._canonical_index % len(runs)
            for index, run in enumerate(runs):
                for hit in run:
                    key = hit.payload.get("frame_id", "")
                    if not key:
                        continue
                    siblings.setdefault(key, []).append(hit)
                    shown.setdefault(key, []).append(hit)
                    # Представник кадру — регіон, обраний КАНОНІЧНИМ каналом:
                    # саме канонічним текстом його потім і оцінюватимуть.
                    better = key not in best or hit.score > best[key].score
                    if index == canonical_run:
                        if key not in canonical_seen or better:
                            best[key] = hit
                        canonical_seen.add(key)
                    elif key not in canonical_seen and better:
                        best[key] = hit
            for key, hit in best.items():
                hit.fusion_score = fused.get(key)
                # Латиниця, і це не стиль, а обмеження: підпис малюється
                # ПІКСЕЛЯМИ поверх фото, а шрифт, вбудований у Pillow, не має
                # кирилиці — українські підписи виходили порожніми
                # прямокутниками. Системний шрифт брати не можна: середовище
                # виконання без мережі може не мати нічого, крім образу.
                # Підпис під фото використовує те саме слово, щоб рамку й
                # рядок під нею не доводилося зіставляти в голові.
                # Ранжування лишається за цією ділянкою, а підпис отримує та,
                # яку показуємо: найпомітніша серед упевнених у цій сутності.
                wanted = entity.count if entity is not None else 1
                instances = _distinct_instances(
                    shown.get(key, [hit]), raw_conditions, wanted
                )
                if len(instances) < wanted:
                    # Кадр не дав стільки РІЗНИХ екземплярів, скільки просить
                    # запит. Це НЕ підстава його викинути: система не вміє
                    # рахувати екземпляри надійно (ADR-022), тож нестача може
                    # означати як «їх справді менше», так і «детектор дав одну
                    # рамку на всю фразу». Слідчий бачить помітку й вирішує сам.
                    shortfall[key] = (len(instances), wanted)
                    hit.payload["instances_found"] = f"{len(instances)} з {wanted}"
                display = instances[0] if instances else hit
                for extra in instances[1:]:
                    extra.entity = (
                        entity.describe(ascii_only=True) if entity is not None else ""
                    )
                    extra.entity_score = _facet_confidence(extra.payload, raw_conditions)
                    kin = siblings.setdefault(key, [])
                    if extra in kin:
                        kin.remove(extra)
                    kin.insert(0, extra)
                display.entity = (
                    entity.describe(ascii_only=True) if entity is not None
                    # Категорія підписується власним іменем: рамка має казати
                    # «nudity», а не «object».
                    else raw_conditions[0][0].removeprefix("cat_").replace("_", " ")
                )
                display.entity_score = _facet_confidence(display.payload, raw_conditions)
                if display is not hit:
                    kin = siblings.setdefault(key, [])
                    if display in kin:
                        kin.remove(display)
                    kin.insert(0, display)
            per_entity.append(best)
            if entity is not None:
                entity_labels.append(entity.describe(ascii_only=True))

        # Заперечення рахуємо завжди й першими. Кадр вибуває, якщо хоч один
        # його регіон підпадає під заборонену умову — саме тому must_not
        # перевіряється на регіонах: на рівні кадру він означав би «в кадрі
        # взагалі немає такої ознаки», що не те саме.
        #
        # Ліміт навмисно щедрий: пропустити заборонений кадр гірше, ніж
        # витратити зайвий обхід. Заперечення, яке інколи не спрацьовує,
        # непридатне для розслідування.
        banned_frames: set[str] = set()
        for raw_conditions in [*parsed.excluded_conditions(), *(extra_excluded or [])]:
            conditions, on_the_fly = self._split_conditions(raw_conditions)
            if not conditions and not on_the_fly:
                continue
            if conditions and not on_the_fly:
                # ТОЧНЕ виключення (M7c). Заборона спирається лише на
                # проіндексовані ознаки, отже питання «які кадри її мають»
                # має точну відповідь, і брати замість неї top-N найсхожіших
                # немає жодних підстав.
                #
                # Стенд масштабу показав, що межа вже пройдена: за
                # `cat_underwear` підпадає 11 508 точок при вибірці 2 000, —
                # тобто заперечення вже сьогодні ймовірнісне, просто на 3861
                # регіоні вибірка накривала весь індекс.
                found, truncated = self.store.frame_ids_matching(
                    schema.REGIONS, build_bound_filter(conditions),
                    cap=NEGATION_SCAN_CAP,
                )
                banned_frames |= found
                if truncated:
                    self._last_unknown.append(
                        f"заборона за {conditions[0][0]} обірвана на "
                        f"{NEGATION_SCAN_CAP} кадрах — виключення неповне"
                    )
                continue

            # Прототип на льоту вимагає ВЕКТОРІВ, тож точний обхід тут
            # неможливий: доводиться оцінювати підняте. Заперечення за такою
            # умовою лишається ймовірнісним, і це названо вголос, а не
            # приховано за однаковим виглядом результату.
            banned = self.store.search(
                schema.REGIONS, vector, limit=fetch,
                query_filter=build_bound_filter(conditions) if conditions else None,
                oversampling=oversampling,
                with_vectors=True,
            )
            # У заперечення підняття не годиться: тут потрібне саме рішення
            # «має ознаку чи ні». Беремо верхню чверть за оцінкою прототипу —
            # решта надто невпевнена, щоб виключати за нею кадр.
            banned = self._refine_on_the_fly(banned, on_the_fly)
            banned = banned[: max(1, len(banned) // 4)]
            banned_frames |= {hit.payload.get("frame_id", "") for hit in banned}

        if not per_entity:
            # Позитивних умов немає — сам лише текст плюс заперечення.
            # Випадок «машина, але не червона» цілком звичайний, тож щільний
            # прохід тут не короткий шлях, а правильна поведінка; заборонені
            # кадри все одно мають бути прибрані.
            dense = self._dense_pass(
                vector, limit + len(banned_frames), scope=SCOPE_REGIONS,
                query_filter=extra, alt_vectors=alt_vectors,
            )
            kept = [
                hit for hit in dense
                if (hit.payload.get("frame_id") or hit.point_id) not in banned_frames
            ]
            return kept[:limit]

        # Кадр підходить, лише якщо в ньому знайшлися ВСІ сутності запиту.
        common = set(per_entity[0])
        for mapping in per_entity[1:]:
            common &= set(mapping)
        common -= banned_frames

        # Оцінка кадру — найслабша з його сутностей: запит виконано настільки,
        # наскільки виконана найгірше підтверджена його частина.
        scored: list[Hit] = []
        for frame_id in common:
            parts = [mapping[frame_id] for mapping in per_entity]
            weakest = min(parts, key=lambda h: h.order_key)
            # Спершу — по одній рамці на КОЖНУ сутність запиту, і лише потім
            # решта знахідок. Порядок тут визначає, що побачить людина: на
            # «дівчина з дитиною» кадр має показати рамку «жінка» і рамку
            # «дитина» окремо. Дві безіменні рамки не дають перевірити, чи
            # знайдено обох, чи двічі одну й ту саму людину.
            weakest.siblings = [p for p in parts if p is not weakest] + [
                h for h in siblings.get(frame_id, []) if h not in parts
            ]
            scored.append(weakest)
        if unknown:
            self._last_unknown = sorted(set(unknown))
            logger.info("індекс не знає фасетів %s — умови пропущено", self._last_unknown)
        # Скільки кадрів УЗАГАЛІ задовольнили умови — не лише показаних.
        self._last_total = len(scored)
        return sorted(scored, key=lambda h: -h.order_key)[:limit]

    def _apply_lexical(
        self, results: list["SearchResult"], query: str, parsed, query_filter
    ) -> list["SearchResult"]:
        """Долучити знахідки лексичного шару та переставити за силою свідчення.

        Точний рядок у прочитаному тексті — свідчення сильніше за будь-який
        косинус: «PXCLD-1624» на знімку або є, або немає. Тому впевненість
        лексичного збігу — це ЧАСТКА слів запиту, знайдених у тексті кадру, і
        вона стоїть на тій самій шкалі [0, 1], що й візуальна: обидві
        відповідають на те саме питання «наскільки система певна».

        Кадри, яких щільний пошук не підняв узагалі, додаються — саме заради
        цього шар і потрібен: за запитом «PXCLD-1624» потрібний знімок мав
        косинус 0.0706 і стояв другим ЗА ТЕКСТОМ, але жодна візуальна модель
        не поставила б його першим.
        """
        if self._sparse_query is None or self._sparse_query.is_empty:
            return results

        texts = [query]
        if parsed is not None and parsed.query_en:
            texts.append(parsed.query_en)
        probe = " ".join(texts)

        try:
            lexical_hits = self.store.search_sparse(
                schema.FRAMES, self._sparse_query,
                limit=LEXICAL_FETCH_LIMIT, query_filter=query_filter,
            )
        except Exception:  # noqa: BLE001 — без лексики видача лишається
            logger.warning("лексичний прохід не вдався", exc_info=True)
            return results

        by_frame = {r.frame_id: r for r in results}
        for hit in lexical_hits:
            # Впевненість рахується ЛИШЕ за написом, не за описом.
            #
            # Це різні за природою свідчення. Напис — точний рядок: «PXCLD-1624»
            # у кадрі або є, або немає, і збіг означає рівно те, що каже. Опис —
            # проза, і збіг у ній може статися на службових словах: «a woman in
            # a blue dress» перекривається з «a girl in a blue skirt» на «in» та
            # «a», що жодним свідченням не є.
            #
            # Виміряно: коли опис почав давати впевненість нарівні з написом,
            # Recall@5 упав 0.80 → 0.73. Опис лишається в розрідженому векторі
            # (він допомагає ЗНАЙТИ кадр), але впевненість дає лише напис.
            text = str(hit.payload.get("ocr_text") or "")
            share, found = lexical.coverage(probe, text)
            if share < MIN_LEXICAL_COVERAGE:
                continue
            frame_id = hit.payload.get("frame_id", "")
            result = by_frame.get(frame_id)
            if result is None:
                result = self._to_result(hit, dict(results[0].provenance) if results else {})
                result.probability = 0.0
                by_frame[frame_id] = result
                results.append(result)
            # Свідчення беруться найсильніше з наявних, а не додаються: кадр
            # не стає вдвічі релевантнішим від того, що його знайшли двома
            # способами.
            # Косинус лишається справжнім: він і далі чесно каже, наскільки
            # кадр схожий візуально. Змінюється лише ВПЕВНЕНІСТЬ — бо тепер
            # свідчень два, і береться сильніше.
            if share > result.probability:
                result.probability = share
            result.matched_attrs = {
                **result.matched_attrs,
                "ocr_match": ", ".join(found),
                "ocr_text": text[:200],
            }
        # Порядок, у якому результати ПРИЙШЛИ, зберігається: його вже визначило
        # злиття свідчень (`_fuse_evidence`). Лексика лише піднімає нагору те,
        # де знайдено точний рядок, і не переставляє решту — повне
        # пересортування тут тихо скасовувало б усе злиття.
        position = {id(r): i for i, r in enumerate(results)}
        return sorted(
            results,
            key=lambda r: (
                0 if r.matched_attrs.get("ocr_match") else 1,
                -r.probability if r.matched_attrs.get("ocr_match") else 0,
                position.get(id(r), len(results)),
            ),
        )

    def _rescore_canonical(self, hits: list[Hit], canonical) -> list[Hit]:
        """Переоцінити готову верхівку ОДНИМ текстом і за ним же впорядкувати.

        Без цього кроку показане число й порядок видачі — різні величини.
        RRF упорядковує за рангами, а користувач бачить косинус того каналу,
        який дав вищий бал, — і у видачі 6% опиняється ВИЩЕ за 13%. Виглядає
        як поламане ранжування, хоча ранжування правильне; поламане саме
        пояснення.

        Розподіл ролей після цього чіткий: **RRF відбирає кандидатів** (обидва
        канали додають те, що знайшли — це повнота), а **канонічний текст їх
        упорядковує** (одна шкала — це зрозумілість). Той самий підхід, що й в
        уточненні детекцією: міряти всіх однією мірою, а не кожного своєю.

        Коштує один запит до Qdrant по щонайбільше `limit` ідентифікаторах.
        """
        if canonical is None or not hits:
            return hits

        by_collection: dict[str, list[str]] = {}
        for hit in hits:
            if hit.vector is None and hit.collection:
                by_collection.setdefault(hit.collection, []).append(hit.point_id)
        vectors: dict[str, list[float]] = {}
        for collection, ids in by_collection.items():
            try:
                vectors.update(self.store.fetch_vectors(collection, ids))
            except Exception:  # noqa: BLE001 — без переоцінки видача лишається
                logger.warning("не вдалося дістати вектори з %s", collection, exc_info=True)

        for hit in hits:
            raw = hit.vector if hit.vector is not None else vectors.get(hit.point_id)
            if raw is None:
                continue
            hit.score = float(np.dot(np.asarray(raw, dtype="float32"), canonical))
        return sorted(hits, key=lambda h: -h.score)

    # ── уточнення обумовленою детекцією (ADR-012, M4e) ──────────────────

    def refine_by_detection(
        self,
        response: "SearchResponse",
        *,
        top_k: int = REFINE_TOP_K,
        detector: Any = None,
    ):
        """Переоцінити верхівку видачі точними рамками під слово запиту.

        Генератор: віддає видачу після КОЖНОГО уточненого кадру. Це не
        зручність, а вимога — детекція коштує ~1.6 с на кадр, тож на десятці
        це 16 секунд. Чекати на них із порожнім екраном неприйнятно, а
        відмовитися від уточнення означає лишити дрібні обʼєкти незнайденими.

        Чому це працює (вимір на знімку з двома синіми спідницями):

            цілий кадр                      0.2%
            плитка 288 px                  10.7%
            рамка обумовленої детекції     56.5%   ← 3.3% площі кадру

        Загальні пропозиції того самого детектора спідниць не знаходять
        взагалі: вони відповідають на питання «що тут головне». Різниця не в
        моделі й не в її здатності бачити дрібне, а в тому, що на етапі
        запиту слово вже відоме.

        Порядок міняється ЛИШЕ всередині уточненої верхівки й лише за однією
        мірою — косинусом кропа з канонічним текстом. Змішувати уточнені
        оцінки з неуточненими не можна: це та сама помилка порівняння різних
        шкал, що й у злитті максимумом (ADR-013).
        """
        from vsearch.ingest.images import load_checked
        from vsearch.represent.regions import TASK_OPEN_VOCAB, Florence2Proposer

        term = (response.parsed.query_en if response.parsed else "") or response.query
        head = response.results[:top_k]
        if not term.strip() or not head:
            yield response
            return

        # Детекція йде ПО СУТНОСТЯХ, коли запит їх називає. На «дівчина з
        # дитиною» це дає рамку жінки й рамку дитини окремо, а не одну рамку
        # на всю фразу. Саме цього бракувало найбільше: рамка була плиткою
        # 288 px, бо жодна пропозиція детектора не окреслювала людину, що
        # займає весь кадр, — а обумовлена детекція окреслює.
        entities = list(response.parsed.must) if response.parsed else []
        # Сутність без класу й без ознак не несе СЛОВА — її підпис це
        # заглушка «object». Заземлювати літеральне «object» безглуздо:
        # детектор чесно повертає якийсь обʼєкт, і на запит «соняшник» це був
        # мангал у кутку кадру. Для такої сутності словом лишається весь
        # `query_en`, як і сказано у вступі до цього методу.
        terms: list[str] = []
        for e in entities:
            described = e.describe(ascii_only=True)
            if e.object is ObjectClass.OTHER and not e.attributes:
                described = term
            if described and described not in terms:
                terms.append(described)
        terms = terms or [term]
        # Кожна сутність коштує окремого проходу детектора (~1.6 с на кадр),
        # тож глибина ділиться між ними: краще уточнити пʼять кадрів за обома
        # сутностями, ніж десять за однією.
        if len(terms) > 1:
            head = response.results[: max(2, top_k // len(terms))]

        detectors = {
            t: (detector if detector is not None and len(terms) == 1
                else Florence2Proposer(self.profile, task=TASK_OPEN_VOCAB, term=t))
            for t in terms
        }
        texts = dict(zip(terms, self.embedder.embed_texts(terms)))
        text = texts[terms[0]]
        refined: dict[str, float] = {}


        for result in head:
            path = Path(result.path)
            if not path.exists():
                continue
            try:
                image, _ = load_checked(path)
                # Рамки всіх сутностей одразу — вони й підуть у показ.
                per_entity_boxes: list[tuple[str, Any, float]] = []
                for name in terms:
                    boxes = detectors[name].propose(image)
                    if not boxes:
                        continue
                    scores_e = self.embedder.embed_images(
                        [_with_context(b).crop(image) for b in boxes]
                    ) @ texts[name]
                    pick_e = int(np.argmax(scores_e))
                    per_entity_boxes.append(
                        (name, boxes[pick_e], float(scores_e[pick_e]))
                    )
                found = [b for _, b, _ in per_entity_boxes]
                # Наявна найкраща рамка йде В ТІ САМІ кандидати. Без цього
                # уточнені кадри мали б оцінку за канонічним текстом, а
                # неуточнені — стару за змішаним каналом, і у видачі 98%
                # опинялося б НИЖЧЕ за 56%. Це та сама помилка порівняння
                # різних шкал, що й у злитті максимумом (ADR-013): міряти
                # треба всіх однією мірою, а не лише тих, кого чіпали.
                candidates = list(found)
                if result.bbox:
                    from vsearch.represent.tiling import Region

                    candidates.append(
                        Region(*result.bbox, kind=result.regions[0].kind
                               if result.regions else "region")
                    )
                if not candidates:
                    refined[result.frame_id] = float("-inf")
                    yield response
                    continue
                crops = [r.crop(image) for r in candidates]
                scores = self.embedder.embed_images(crops) @ text
            except Exception:  # noqa: BLE001 — один кадр не спиняє уточнення
                logger.warning("уточнення не вдалося для %s", path.name, exc_info=True)
                yield response
                continue

            pick = int(np.argmax(scores))
            region = candidates[pick]
            cosine = float(scores[pick])

            # ПЕРЕВІРКА НАЯВНОСТІ, а не лише перестановка (ADR-018).
            #
            # Якщо детектор не зміг заземлити слово запиту в кадрі — слова там
            # немає. Це та роль, яку в плані мав виконувати крок [5] через
            # Florence-2 VQA; такого завдання в моделі не існує, а обумовлена
            # детекція його замінює й розділяє начисто: на золотому наборі
            # пастки падають до 0–1%, тоді як цілі дають 11–51%.
            #
            # Межа — та сама `MIN_PROBABILITY`, за якою система вже вирішує,
            # чи взагалі щось знайдено. Нового підігнаного числа тут немає й
            # бути не повинно.
            grounded = max(
                (float(self.embedder.probability(box_score))
                 for _, _, box_score in per_entity_boxes),
                default=0.0,
            )
            if grounded < MIN_PROBABILITY:
                # ПОЗНАЧКА, а не пониження — і це висновок із виміру, а не
                # обережність.
                #
                # Спершу тут стояло сортування, що опускало непідтверджені
                # кадри. Проба на пастках виглядала переконливо: 4 з 6 запитів
                # розділялися начисто. Але вона порівнювала НАЙКРАЩУ ціль із
                # НАЙКРАЩОЮ пасткою на запит, а покадрово розподіли
                # перетинаються — цілі на 0–4% сидять там само, де пастки на
                # 0–1%. На золотому наборі пониження прибрало дві пастки
                # ціною падіння Recall@5 з 0.82 до 0.62.
                #
                # Тому кадр лишається на місці, а слідчий бачить, чого саме
                # система не підтвердила, і вирішує сам.
                result.matched_attrs = {
                    **result.matched_attrs, "not_grounded": term,
                }
            # Оцінка кадру НЕ ЧІПАЄТЬСЯ, і це не обережність, а те саме
            # правило, яке проєкт уже сформулював тричі: число поруч із
            # твердженням має походити з того самого виміру, що й саме
            # твердження. Порядок видачі визначило злиття свідчень; якщо
            # після уточнення підписати кадри косинусом кропа детектора —
            # іншим виміром, — то #1 отримає 3%, а #3 сорок, і видача
            # виглядатиме перерахованою, хоча жоден кадр не переставлявся.
            # Саме це користувач і бачив: «спочатку дає правильний
            # результат, а потім щось перераховується».
            #
            # Своє свідчення детекція віддає в окреме поле
            # `detection_confidence`, яке показується поруч і підписане тим,
            # на що відповідає.
            refined[result.frame_id] = cosine
            if pick < len(found):
                # Виграла рамка детектора — у показ ідуть рамки ВСІХ
                # сутностей, кожна зі своїм підписом. Одна рамка на фразу не
                # давала б перевірити, чи знайдено обох, чи двічі одного.
                result.bbox = region.bbox
                result.regions = [
                    RegionHit(
                        bbox=box.bbox, score=box_score, kind="detected",
                        label=box.label or name, entity=name,
                        probability=float(self.embedder.probability(box_score)),
                        entity_score=float(self.embedder.probability(box_score)),
                    )
                    for name, box, box_score in per_entity_boxes
                ][:MAX_SHOWN_REGIONS]
                result.provenance = {**result.provenance, "refined": term}
                # Провенанс має говорити про ту рамку, що показана. Інакше
                # результат казав би «плитка r3c1», показуючи рамку детектора,
                # і пояснення суперечило б картинці.
                result.matched_attrs = {
                    **result.matched_attrs,
                    "region": "detected",
                    "label": region.label or term,
                }
            # Виграла вже наявна рамка: детектор не переконав, і підміняти
            # знайдену плитку його здогадом було б погіршенням показу. Її
            # власні числа теж лишаються — вони походять із того відбору, що
            # цю рамку й обрав.

            # Уточнення НЕ ЧІПАЄ ПОРЯДОК. Його цінність — точна рамка, і це
            # виміряно тричі: як заміна порядку воно давало Recall@5 0.54
            # замість 0.79, з розширеним контекстом — 0.54, як третій канал
            # злиття — 0.51. Кроп детектора міряє інше, ніж решта видачі, і
            # додавання цього виміру в порядок щоразу псувало його.
            #
            # Побічний наслідок перестановок бачив користувач: кадр показувався
            # другим, а потім ЗНИКАВ із видачі, коли уточнення доходило до
            # нього. Для слідчого це виглядає як втрата доказу.
            result.detection_confidence = float(
                self.embedder.probability(refined[result.frame_id])
            )
            yield response

    def _feedback_vectors(
        self, positive: "Sequence[str]", negative: "Sequence[str]",
        missing: list[str],
    ) -> tuple[list["np.ndarray"], list["np.ndarray"]]:
        """Вектори відмічених кадрів — одним запитом на обидва списки.

        Кадр, якого в індексі немає, ПЕРЕЛІЧУЄТЬСЯ, а не мовчки ігнорується:
        «знайти схоже» без жодного зразка вироджується у звичайний пошук, і
        слідчий має право знати, що його позначку не враховано.
        """
        from vsearch.index.store import point_id as _pid

        wanted = [*positive, *negative]
        ids = {frame: _pid(frame) for frame in wanted}
        found = self.store.fetch_vectors(schema.FRAMES, list(ids.values()))
        missing.extend(f for f, pid in ids.items() if pid not in found)
        pos = [np.asarray(found[ids[f]], dtype=np.float32)
               for f in positive if ids[f] in found]
        neg = [np.asarray(found[ids[f]], dtype=np.float32)
               for f in negative if ids[f] in found]
        return pos, neg

    @staticmethod
    def _to_result(hit: Hit, provenance: dict[str, Any]) -> SearchResult:
        payload = hit.payload
        bbox = payload.get("bbox")
        return SearchResult(
            regions=_collect_regions(hit),
            asset_id=payload.get("asset_id", ""),
            frame_id=payload.get("frame_id", ""),
            score=hit.score,
            path=payload.get("path", ""),
            media_type=payload.get("media_type", "image"),
            ts_ms=payload.get("ts_ms"),
            bbox=tuple(bbox) if bbox else None,
            matched_attrs=(
                {"region": payload["region_type"], "label": payload.get("label", "")}
                | ({"shot_id": payload["shot_id"]} if payload.get("shot_id") else {})
                | (
                    {"instances_found": payload["instances_found"]}
                    if payload.get("instances_found")
                    else {}
                )
                if payload.get("region_type")
                else {}
            ),
            provenance=(
                {**provenance, "shot_id": payload["shot_id"]}
                if payload.get("shot_id") else provenance
            ),
        )


#: Ліниво створюваний пошуковець за замовчуванням — щоб `scripts/benchmark.py`
#: і крокові визначення не піднімали модель на кожен виклик.
_default: Searcher | None = None


def search(query: str, *, profile: str | None = None, limit: int = 10) -> SearchResponse:
    """Зручний фасад над Searcher для скриптів і тестів."""
    global _default
    wanted = get_profile(profile)
    if _default is None or _default.profile.name != wanted.name:
        _default = Searcher(profile=wanted)
    return _default.search(query, limit=limit)
