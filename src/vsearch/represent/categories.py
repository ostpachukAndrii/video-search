"""Категоризація (п.4): наперед задані категорії плюс власні від користувача.

Категорія — це прототип, а не модель. Наслідок, заради якого все так і
побудовано: додавання категорії до вже наявного індексу зводиться до множення
матриці збережених ембедингів на один вектор. Пікселі повторно не читаються,
моделі повторно не запускаються, переіндексації немає.
"""

from __future__ import annotations

from typing import Any

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Sequence

import numpy as np

from vsearch.represent.prototypes import (
    KIND_ATTRIBUTE,
    KIND_CATEGORY,
    Prototype,
    PrototypeBank,
)

if TYPE_CHECKING:
    from PIL.Image import Image

logger = logging.getLogger(__name__)


def _resolve_exclusive(
    payloads: list[dict],
    scores: dict,
    chosen: Sequence[str],
    region_types: list[str] | None = None,
) -> None:
    """Звести взаємовиключні фасети до одного ключа зі значенням-переможцем.

    Шість булевих `attr_color_*` замінюються на один `attr_color="white"`.
    Так фільтр «білий» стає точним вибором, а не «є трохи білого», і запит
    у парсері (`color=white`) лягає в payload буквально.
    """
    for group, members in EXCLUSIVE_GROUPS.items():
        present = [m for m in members if m in scores and m in chosen]
        if len(present) < 2:
            continue
        matrix = np.stack([scores[m].raw for m in present])  # (кольори, точки)
        order = np.argsort(-matrix, axis=0)
        winner, runner = order[0], order[1]
        for index, payload in enumerate(payloads):
            for member in present:
                payload.pop(f"attr_{member}", None)
                payload.pop(f"attr_{member}_score", None)
            # Ключ скидається ЯВНО, а не пропуском: set_payload лише
            # встановлює значення, тож без цього старий колір від попереднього
            # прогону лишався б назавжди.
            payload[f"attr_{group}"] = None
            payload[f"attr_{group}_confidence"] = None

            allowed_types = GROUP_REGION_TYPES.get(group)
            if region_types is not None and allowed_types is not None:
                kind = region_types[index] if index < len(region_types) else ""
                if kind not in allowed_types:
                    continue  # плитка: її колір про обʼєкт нічого не каже
            best = matrix[winner[index], index]
            second = matrix[runner[index], index]
            if best <= 0 or best - second < EXCLUSIVE_MARGIN:
                continue  # різнокольоровий кроп: однієї відповіді немає
            payload[f"attr_{group}"] = present[winner[index]].split("_", 1)[1]
            payload[f"attr_{group}_confidence"] = round(float(best - second), 4)


def _category(name: str, *phrases: str, description: str = "") -> Prototype:
    return Prototype(name=name, positive=phrases, kind=KIND_CATEGORY, description=description)


def _attribute(
    name: str, positive: Sequence[str], negative: Sequence[str], gate: str = "person"
) -> Prototype:
    """Бінарний атрибут із явним негативом.

    Для «в окулярах / без окулярів» явна протилежність надійніша за загальне
    тло: спільна складова формулювань (тут — «людина») скорочується, лишається
    саме та ознака, яку розрізняємо. Це і є механізм заперечення з п.11.
    """
    return Prototype(
        name=name, positive=tuple(positive), negative=tuple(negative),
        kind=KIND_ATTRIBUTE, gate=gate,
    )


#: Наперед задані категорії для розслідувань, згруповані за призначенням.
#:
#: Перелік розширений навмисно: категорія коштує майже нічого (текстовий
#: прототив плюс матричне множення по вже збережених векторах), а брак
#: категорії означає, що слідчий не знайде матеріал взагалі. Тому дешевше
#: мати зайву категорію, ніж пропущену.
#:
#: Там, де хибне спрацювання дорого коштує, негатив задано ЯВНО, а не взято
#: загальне тло. Найважливіший приклад — стан одягу: без протиставлення
#: пляжному й спортивному одягу під «спідню білизну» потрапляє купа
#: безневинних фото, і слідчий витрачає час на їх відсів.

def _contrast(name: str, positive: Sequence[str], negative: Sequence[str],
              description: str = "") -> Prototype:
    """Категорія з явним протиставленням — там, де тло замалим."""
    return Prototype(
        name=name, positive=tuple(positive), negative=tuple(negative),
        kind=KIND_CATEGORY, description=description,
    )


#: Стан одягу. Ключова група для форензіки, і водночас найризикованіша щодо
#: хибних спрацювань, тому кожна категорія протиставлена сусіднім.
CLOTHING_CATEGORIES: tuple[Prototype, ...] = (
    _contrast(
        "underwear",
        ["a person in underwear", "a person in lingerie", "a person in bra and panties"],
        # Негативи мають бути КОНКУРЕНТНИМИ гіпотезами для того самого кропа.
        # «Порожня кімната» такою не є: кроп людини завжди ближчий до
        # «людина в білизні», ніж до порожньої кімнати, тож такий негатив
        # нічого не розрізняє й лише завищує спрацювання.
        ["a fully clothed person", "a person in swimwear at the beach",
         "a person in sportswear", "a person in a dress or a suit"],
        description="Спідня білизна — протиставлено пляжному й спортивному одягу",
    ),
    _contrast(
        "nudity",
        ["a nude person", "an unclothed human body", "bare skin of a naked person"],
        ["a fully clothed person", "a person in underwear",
         "a person in swimwear", "a portrait of a dressed person"],
        description="Оголення — протиставлено білизні, щоб розрізняти ступінь",
    ),
    _contrast(
        "swimwear",
        ["a person in swimwear", "a person in a bikini at the beach"],
        ["a person in underwear indoors", "a fully clothed person"],
        description="Пляжний одяг — існує саме щоб знімати хибні спрацювання underwear",
    ),
    _category("uniform", "a person in uniform", "a police or military uniform",
              description="Формений одяг"),
)

#: Предмети, що цікавлять слідство.
OBJECT_CATEGORIES: tuple[Prototype, ...] = (
    _category("weapon", "a weapon", "a firearm or a knife", "a gun",
              description="Зброя загалом"),
    _category("firearm", "a firearm", "a pistol or rifle", "a handgun",
              description="Вогнепальна зброя — окремо від холодної"),
    _category("blade", "a knife", "a blade or sharp weapon",
              description="Холодна зброя"),
    _category("ammunition", "ammunition", "bullets or cartridges",
              description="Боєприпаси"),
    _category("money", "banknotes and cash", "a stack of money",
              description="Готівка"),
    _category("drugs", "narcotic substances", "packets of powder or pills",
              description="Наркотичні речовини"),
    _category("syringe", "a syringe", "a hypodermic needle",
              description="Шприци, ін'єкційне приладдя"),
    _category("document", "a document", "a passport, id card or paper document",
              description="Документи, посвідчення"),
    _category("license_plate", "a vehicle license plate", "a car number plate",
              description="Номерні знаки"),
    _category("phone", "a mobile phone", "a smartphone in hand",
              description="Телефон"),
    _category("electronics", "a laptop or computer", "electronic devices",
              description="Електроніка, носії даних"),
    _category("storage_media", "a usb flash drive", "a memory card or hard drive",
              description="Носії інформації"),
    _category("bag", "a bag, backpack or suitcase", "luggage",
              description="Сумки, багаж"),
    _category("jewelry", "jewelry", "a gold ring, necklace or watch",
              description="Коштовності"),
    _category("vehicle", "a vehicle", "a car, truck or motorcycle",
              description="Транспортні засоби"),
    _category("animal", "an animal", "a dog, cat or other animal",
              description="Тварини"),
)

#: Обстановка й контекст кадру.
SCENE_CATEGORIES: tuple[Prototype, ...] = (
    _category("indoor", "an indoor scene", "the interior of a room",
              description="Приміщення"),
    _category("outdoor", "an outdoor scene", "a street or open area",
              description="Вулиця, відкрита місцевість"),
    _category("night", "a scene at night", "a dark night photo",
              description="Нічна зйомка"),
    _category("bedroom", "a bedroom", "a room with a bed",
              description="Спальня — контекст для матеріалів про експлуатацію"),
    _category("bathroom", "a bathroom", "a shower or bathtub",
              description="Ванна кімната"),
    _category("vehicle_interior", "the interior of a car", "a view from inside a vehicle",
              description="Салон автомобіля"),
    _category("building", "a building", "a house or a wall with doors",
              description="Будівлі, споруди"),
    _category("crowd", "a crowd of people", "many people gathered",
              description="Скупчення людей"),
    _category("blood", "blood", "a bloodstain or wound",
              description="Кров, поранення"),
    _category("damage", "destroyed property", "a damaged building or burned car",
              description="Руйнування, пошкодження майна"),
)

#: Ознаки для встановлення особи та текстові матеріали.
IDENTITY_CATEGORIES: tuple[Prototype, ...] = (
    _category("person", "a person", "a human being", "a man or a woman",
              description="Наявність людини — передумова атрибутів"),
    _category("face_visible", "a clear photo of a human face", "a close-up portrait",
              description="Чітко видиме обличчя — придатне для впізнання"),
    _category("child", "a child", "a young kid or a baby",
              description="Дитина — вимагає особливого поводження з матеріалом"),
    _category("tattoo", "a tattoo on skin", "tattooed body art",
              description="Татуювання — стійка ознака особи"),
    _category("screenshot", "a screenshot of a phone screen", "a chat conversation on screen",
              description="Знімки екрана, листування"),
    _category("handwriting", "handwritten text on paper", "a handwritten note",
              description="Рукописний текст"),
)

DEFAULT_CATEGORIES: tuple[Prototype, ...] = (
    IDENTITY_CATEGORIES + CLOTHING_CATEGORIES + OBJECT_CATEGORIES + SCENE_CATEGORIES
)

#: Атрибути людини — фундамент заперечення (п.11). Кожен має явну
#: протилежність, а не загальне тло.
DEFAULT_ATTRIBUTES: tuple[Prototype, ...] = (
    _attribute("glasses",
               ["a person wearing glasses", "a face with eyeglasses"],
               ["a person without glasses", "a face with no eyeglasses"]),
    _attribute("headwear",
               ["a person wearing a hat", "a person with a cap or hood"],
               ["a person with bare head", "a person without a hat"]),
    _attribute("mask",
               ["a person wearing a face mask", "a covered face"],
               ["a person with an uncovered face", "a person without a mask"]),
    _attribute("beard",
               ["a bearded man", "a face with a beard"],
               ["a clean shaven face", "a face without a beard"]),
    # Стать і вік — пряма вимога п.13. Досі парсер їх витягував, а індекс не
    # рахував, тож запит «чоловік» шукав просто «людину», і в видачу
    # потрапляли жінки. Фасет булевий: True означає «чоловік», False —
    # «жінка», і саме тому негатив тут явний, а не загальне тло.
    _attribute("gender_male",
               ["a man", "a male person", "an adult man"],
               ["a woman", "a female person", "an adult woman"]),
    _attribute("adult",
               ["an adult person", "a grown up man or woman"],
               ["a child", "a little kid", "a baby"]),
)

#: Колір як ФАСЕТ, а не як частина текстового запиту.
#:
#: Виміряно на реальному фото: «велосипед» дає впевненість 21.7%, а «білий
#: велосипед» — 3.5%. Додавання кольору псує ембединг ушестеро, бо композиція
#: «ознака + іменник» у CLIP-подібних моделях представлена погано.
#:
#: Розвʼязання те саме, що й для заперечення: шукати ембедингом сам обʼєкт,
#: а ознаку застосовувати фільтром. Кольори без передумови — вони стосуються
#: будь-якого обʼєкта, не лише людини.
COLOR_NAMES = (
    "white", "black", "red", "blue", "green", "yellow",
    # Розширено після реального запиту: «рожеве плаття» не знаходилось, бо
    # рожевого в переліку не було, і ознака зникала безслідно — фільтр
    # мовчки лишався з самим «жінка», якому відповідає 92 кадри зі 116.
    "pink", "orange", "purple", "brown", "grey", "beige",
)


def _color(name: str) -> Prototype:
    others = [c for c in COLOR_NAMES if c != name]
    return Prototype(
        name=f"color_{name}",
        positive=(f"a mostly {name} object", f"something {name} in colour"),
        negative=tuple(f"a mostly {c} object" for c in others),
        kind=KIND_ATTRIBUTE,
    )


COLOR_ATTRIBUTES: tuple[Prototype, ...] = tuple(_color(c) for c in COLOR_NAMES)

#: Колір волосся — окремий взаємовиключний вибір, прив'язаний до людини.
#: «Брюнетка» й «блондинка» це найчастіший спосіб описати особу словами, а
#: до цього ознака не існувала взагалі й запит її просто губив.
#: Темне волосся НЕ ділиться на каштанове й чорне.
#:
#: Спершу класів було пʼять, і на реальному фото це зламалося: один регіон
#: жінки модель віднесла до brunette, три — до black_hair, тож запит
#: «брюнетка» кадру не знаходив. Межа між темно-каштановим і чорним у моделі
#: проходить інакше, ніж у людини, а коли людина каже «брюнетка», вона має на
#: увазі «темноволоса, не блондинка». Розрізняти тут — вигадувати різницю,
#: якої в запиті немає.
HAIR_NAMES = ("blonde", "dark", "red_hair", "grey_hair")

_HAIR_PHRASES = {
    "blonde": "a person with blonde or light hair",
    "dark": "a person with dark brown or black hair",
    "red_hair": "a person with red ginger hair",
    "grey_hair": "a person with grey or white hair",
}


def _hair(name: str) -> Prototype:
    others = [h for h in HAIR_NAMES if h != name]
    return Prototype(
        name=f"hair_{name}",
        positive=(_HAIR_PHRASES[name],),
        negative=tuple(_HAIR_PHRASES[h] for h in others),
        kind=KIND_ATTRIBUTE,
        gate="person",
    )


HAIR_ATTRIBUTES: tuple[Prototype, ...] = tuple(_hair(h) for h in HAIR_NAMES)

#: Атрибути ЛЮДИНИ — мають передумову `person`, бо «без окулярів» на кадрі без
#: людини означало б «це порожня кімната» (див. ADR-005).
PERSON_ATTRIBUTES: tuple[Prototype, ...] = DEFAULT_ATTRIBUTES

#: Атрибути ОБʼЄКТА — передумови не мають: колір стосується чого завгодно,
#: а не лише людини. Натомість вони обмежені типом регіону через
#: COLOR_REGION_TYPES: на плитці колір нічого не означає.
OBJECT_ATTRIBUTES: tuple[Prototype, ...] = COLOR_ATTRIBUTES

#: Що справді передобчислюється при індексації.
#:
#: Роль передобчислених фасетів звузилася до однієї: **звуження на мільйонах**
#: до пошуку по ANN. Прототип на льоту працює по вже піднятих кандидатах, тож
#: відсіяти 20M до 10K він не може — а фільтр може.
#:
#: Для цього потрібні десятки ознак, а не сотні. Кольори й волосся сюди більше
#: НЕ входять: вони були перелічені руками, «рожевого» в переліку не було, і
#: запит його мовчки губив. Тепер вони рахуються з тексту запиту за частки
#: мілісекунди, і перелічувати їх наперед немає потреби.
DEFAULT_ATTRIBUTES = PERSON_ATTRIBUTES

#: Доступні для оцінювання на льоту, але не передобчислювані.
ON_DEMAND_ATTRIBUTES: tuple[Prototype, ...] = OBJECT_ATTRIBUTES + HAIR_ATTRIBUTES

#: Колір — ВЗАЄМОВИКЛЮЧНИЙ вибір, а не шість незалежних ознак.
#:
#: Незалежні пороги тут не працюють: кроп, у якому є хоч трохи білого, ближчий
#: до «переважно білого», ніж до «переважно зеленого», тож кожен колір
#: спрацьовував на 85–87% регіонів. Замість шести булевих ставимо ОДИН ключ
#: `attr_color` зі значенням-переможцем — і лише коли він відірвався від
#: другого достатньо, щоб вибір був осмисленим.
EXCLUSIVE_GROUPS: dict[str, tuple[str, ...]] = {
    "color": tuple(f"color_{c}" for c in COLOR_NAMES),
    "hair": tuple(f"hair_{h}" for h in HAIR_NAMES),
}

#: Наскільки переможець має випереджати другого, щоб колір узагалі присвоївся.
#: Менший відрив означає, що кроп різнокольоровий і однієї відповіді немає.
EXCLUSIVE_MARGIN = 0.01

#: Типи регіонів, для яких колір узагалі має сенс.
#:
#: Плитка займає близько третини кадру, і її переважний колір нічого не каже
#: про дрібний обʼєкт усередині: за запитом «білий велосипед» фільтр
#: спрацьовував на плитці зі світлим тлом, а велосипед у ній був іншого
#: кольору. Колір — ознака ОБʼЄКТА, тож і рахувати його можна лише на
#: тісному кропі обʼєкта.
COLOR_REGION_TYPES = frozenset({"object", "person"})

#: Для волосся обмеження те саме: на плитці визначати колір волосся немає сенсу.
GROUP_REGION_TYPES = {"color": COLOR_REGION_TYPES, "hair": COLOR_REGION_TYPES}


@dataclass
class ApplyReport:
    """Що саме зробило застосування категорії до індексу."""

    name: str
    scanned: int = 0
    updated: int = 0
    positive: int = 0
    #: Скільки точок не отримали фасета, бо не пройшли передумову.
    gated_out: int = 0
    #: Межа, виведена з корпусу (None — визначити не вдалося).
    threshold: float | None = None
    #: True, якщо межі на цьому корпусі не існує: збережено лише оцінку.
    unthresholded: bool = False
    elapsed_s: float = 0.0
    reindexed_pixels: bool = False  # завжди False — і це головне

    @property
    def rate(self) -> float:
        return self.scanned / self.elapsed_s if self.elapsed_s else 0.0

    def summary(self) -> str:
        parts = [
            f"{self.name}: переглянуто {self.scanned}",
            f"позитивних {self.positive} ({self.positive / max(1, self.scanned):.1%})",
        ]
        if self.unthresholded:
            parts.append("межі не визначено — лише оцінка")
        elif self.threshold is not None:
            parts.append(f"межа з корпусу {self.threshold:.4f}")
        if self.gated_out:
            parts.append(f"без передумови {self.gated_out}")
        parts.append(f"{self.elapsed_s:.1f}с ({self.rate:.0f} векторів/с)")
        return ", ".join(parts)


class CategoryEngine:
    """Реєстр прототипів і застосування їх до наявного індексу."""

    def __init__(self, bank: PrototypeBank, store=None) -> None:
        self.bank = bank
        self.store = store

    @classmethod
    def with_defaults(cls, embedder, store=None) -> "CategoryEngine":
        bank = PrototypeBank(embedder)
        bank.add_many(DEFAULT_CATEGORIES)
        bank.add_many(DEFAULT_ATTRIBUTES)
        return cls(bank, store)

    # ── користувацькі категорії ─────────────────────────────────────────────

    def add_user_category(
        self,
        name: str,
        text: str,
        *,
        examples: Sequence["Image"] = (),
        extra_phrases: Iterable[str] = (),
        description: str = "",
    ) -> Prototype:
        """Додати категорію, описану користувачем звичайним текстом.

        Негатив не потрібен: тло підставляється автоматично. Приклади
        необовʼязкові й лише уточнюють формулювання там, де слова заслабкі.
        """
        phrases = (text, *extra_phrases)
        return self.bank.add(
            Prototype(name=name, positive=phrases, kind=KIND_CATEGORY, description=description),
            examples=examples,
        )

    # ── застосування до індексу ─────────────────────────────────────────────

    #: Наскільки має вирости корпус, щоб межу варто було виводити наново.
    #: Менший приріст не змінює ФОРМИ розподілу, з якої вона й береться, тож
    #: перевиведення коштувало б проходу по індексу без жодного виграшу.
    THRESHOLD_REFRESH_GROWTH = 0.25

    def _remembered_threshold(
        self, collection: str, name: str, sample: int, catalog
    ) -> float | None:
        """Корпусна межа: збережена, якщо корпус істотно не виріс.

        Без цього інкрементальний перерахунок був би НЕПРАВИЛЬНИМ, а не просто
        повільним: межа виводиться з розподілу, і взята з однієї нової партії
        вона означала б інше, ніж межа, за якою розмічений увесь індекс.
        """
        if catalog is None:
            return self._corpus_threshold(collection, name, sample)

        total = self.store.count(collection, exact=False) if self.store else 0
        remembered = catalog.corpus_threshold(collection, name)
        if remembered is not None:
            value, at_points = remembered
            grown = at_points and (total - at_points) / at_points
            if not grown or grown < self.THRESHOLD_REFRESH_GROWTH:
                return value

        derived = self._corpus_threshold(collection, name, sample)
        catalog.save_corpus_threshold(collection, name, derived, total)
        return derived

    def _corpus_threshold(self, collection: str, name: str, sample: int) -> float | None:
        """Вивести межу з розподілу самого корпусу.

        Типова межа «різниця > 0» для категорій непридатна, і це виміряно:
        на 116 реальних фото `underwear` спрацьовував на 82%, `nudity` на 33%,
        `swimwear` на 24%. Такий фасет не звужує пошук, а засмічує його — а у
        форензік-інструменті хибне спрацювання коштує часу слідчого.

        Розривне правило на тих самих даних дало 2%, 3% і 1% відповідно, а
        `person` лишило на 89% — бо люди справді є на більшості знімків. Тобто
        воно ріже шум, не чіпаючи того, що дійсно поширене.

        Межа рахується на ВИБІРЦІ: на мільйонах точок повний прохід заради
        одного порога не окупається, а форма розподілу від вибірки не залежить.
        """
        collected = []
        for _, vectors in self.store.iter_vectors(collection, batch_size=512):
            collected.extend(vectors)
            if len(collected) >= sample:
                break
        if len(collected) < 8:
            return None
        matrix = np.asarray(collected[:sample], dtype=np.float32)
        threshold, _ = self.bank.suggest_threshold(name, matrix)
        return threshold

    def apply_to_collection(
        self,
        collection: str,
        name: str,
        *,
        batch_size: int = 512,
        store_score: bool = True,
        calibrate_on_corpus: bool = True,
        sample: int = 5000,
        #: Обмежити перерахунок частиною колекції — зазвичай щойно доданими
        #: точками. Межа при цьому лишається КОРПУСНОЮ (див. `_corpus_threshold`):
        #: інакше той самий фасет означав би на нових точках не те, що на старих.
        scope_filter: Any = None,
        catalog=None,
    ) -> ApplyReport:
        """Проставити фасет усім точкам колекції.

        Читаються ЛИШЕ вектори з індексу — ані вихідних файлів, ані моделей
        зображень. Тому вартість не залежить від того, скільки важать
        оригінали, і нова категорія застосовується до готового індексу за
        секунди, а не за години переіндексації.
        """
        if self.store is None:
            raise RuntimeError("CategoryEngine створено без сховища")

        prototype = self.bank.get(name)
        report = ApplyReport(name=name)
        started = time.perf_counter()

        # Калібрований на розмітці поріг завжди має пріоритет: він знає більше
        # за будь-яку евристику. Розривний застосовується лише там, де розмітки
        # не було, тобто майже завжди для категорій.
        threshold = prototype.threshold
        if calibrate_on_corpus and not prototype.calibrated:
            derived = self._remembered_threshold(collection, name, sample, catalog)
            if derived is not None:
                threshold = derived
                report.threshold = derived

        for ids, vectors in self.store.iter_vectors(
            collection, batch_size=batch_size, query_filter=scope_filter
        ):
            if not len(ids):
                continue
            matrix = np.asarray(vectors, dtype=np.float32)
            scores = self.bank.score(matrix, name)
            if threshold is not None:
                values = scores.raw if prototype.is_contrastive else scores.probability
                scores.decision = values > threshold

            # Передумова: фасет має сенс лише там, де вона спрацювала.
            # Де не спрацювала — значення лишається НЕВІДОМИМ, а не «ні»:
            # це різні речі, і плутанина між ними зламала б заперечення.
            if prototype.gate:
                allowed = self.bank.score(matrix, prototype.gate).decision
            else:
                allowed = np.ones(len(matrix), dtype=bool)

            payloads: list[dict] = []
            for index in range(len(ids)):
                if not allowed[index]:
                    payloads.append({prototype.payload_key: None})
                    continue
                payload = {prototype.payload_key: bool(scores.decision[index])}
                if store_score:
                    payload[f"{prototype.payload_key}_score"] = float(scores.probability[index])
                payloads.append(payload)

            self.store.set_payloads(collection, list(ids), payloads)
            report.scanned += len(ids)
            report.updated += len(ids)
            report.gated_out += int((~allowed).sum())
            report.positive += int((scores.decision & allowed).sum())

        report.elapsed_s = time.perf_counter() - started
        logger.info(report.summary())
        return report

    def apply_many(
        self,
        collection: str,
        names: Sequence[str] | None = None,
        *,
        kinds: tuple[str, ...] | None = None,
        batch_size: int = 512,
        sample: int = 5000,
        store_scores: bool = True,
        #: Обмежити перерахунок частиною колекції — зазвичай новими точками.
        scope_filter: Any = None,
        catalog=None,
    ) -> dict[str, ApplyReport]:
        """Застосувати ВСІ прототипи за один прохід по колекції.

        Поштучне застосування робить K проходів по векторах і K оновлень
        payload на кожну точку. Для 42 прототипів це десятки секунд, для
        таксономії на тисячу класів — десятки хвилин, тобто таксономію
        довелося б обирати заздалегідь замість приміряти.

        Тут прохід один: усі прототипи оцінюються одним множенням матриць, і
        всі фасети точки пишуться однією операцією.

        **Категорії зберігаються розріджено** — лише ті, що спрацювали. При
        тисячі категорій щільний запис означав би близько десяти кілобайт на
        точку, тобто десятки гігабайт payload на мільйонах регіонів. Атрибути
        лишаються щільними: їх мало, і `False` там несе зміст («людина без
        окулярів»), тоді як у категорії відсутність і «ні» рівнозначні.
        """
        if self.store is None:
            raise RuntimeError("CategoryEngine створено без сховища")

        if names is not None:
            chosen = list(names)
        elif kinds is not None:
            chosen = [p.name for p in self.bank if p.kind in kinds]
        else:
            chosen = self.bank.names
        reports = {name: ApplyReport(name=name) for name in chosen}
        started = time.perf_counter()

        thresholds = self._remembered_thresholds(collection, chosen, sample, catalog)
        gates = {self.bank.get(n).gate for n in chosen if self.bank.get(n).gate}
        needed = list(dict.fromkeys([*gates, *chosen]))

        for ids, vectors, region_types in self._iter_with_types(
            collection, batch_size, scope_filter=scope_filter
        ):
            if not len(ids):
                continue
            matrix = np.asarray(vectors, dtype=np.float32)
            scores = self.bank.score_many(matrix, needed)
            # Передумова мусить використовувати ТУ САМУ межу, що й збережений
            # фасет. Інакше регіон отримає attr_glasses, але не отримає
            # cat_person — і фільтр «людина без окулярів», який вимагає обох,
            # його загубить. Розбіжність тут не дає помилки, лише порожні
            # місця у видачі.
            gate_pass = {}
            for gate in gates:
                if gate not in scores:
                    continue
                gate_threshold = thresholds.get(gate)
                if gate_threshold is None:
                    gate_pass[gate] = scores[gate].decision
                else:
                    prototype = self.bank.get(gate)
                    values = (
                        scores[gate].raw if prototype.is_contrastive
                        else scores[gate].probability
                    )
                    gate_pass[gate] = values > gate_threshold

            payloads: list[dict] = [{} for _ in ids]
            for name in chosen:
                prototype = self.bank.get(name)
                score = scores[name]
                report = reports[name]
                threshold = thresholds.get(name)
                report.threshold = threshold
                # Атрибути межі з корпусу не потребують: у них явна
                # протилежність («в окулярах» ↔ «без окулярів»), тож знак
                # різниці змістовний сам по собі. Категорії протиставлені
                # загальному тлу, і там знак нічого не означає.
                if prototype.kind == KIND_ATTRIBUTE:
                    threshold = prototype.threshold
                    report.threshold = threshold
                    values = score.raw if prototype.is_contrastive else score.probability
                    decision = values > threshold if threshold is not None else score.decision
                elif threshold is None and not prototype.calibrated:
                    # Межі З КОРПУСУ вивести не вдалося — беремо власне рішення
                    # прототипу, те саме, яким уже користується передумова
                    # (`gate_pass` вище).
                    #
                    # Раніше тут стояв `continue`: писалася лише оцінка, а
                    # булевий ключ не зʼявлявся взагалі. Наслідок був тихий і
                    # дорогий. Поріг виводиться з РОЗРИВУ в розподілі, а в
                    # `person` розриву немає — люди є майже на кожному знімку.
                    # Тож найважливіша категорія системи не потрапляла в індекс
                    # ніколи, і кожен запит про людину фільтрував за ключем,
                    # якого не існує: 28 категорій із 36 були відсутні на
                    # регіонах, серед них `cat_person` і `cat_bag`.
                    #
                    # Головне — це була РОЗБІЖНІСТЬ: передумова довіряла
                    # `score.decision` і пропускала 1814 регіонів як людей, а
                    # індекс про це мовчав. Якщо рішення достатньо надійне, щоб
                    # відкрити атрибути, воно достатньо надійне, щоб бути
                    # записаним. Некаліброваність лишається видимою в
                    # `report.unthresholded` і в збереженій оцінці.
                    report.unthresholded = True
                    decision = score.decision
                else:
                    values = score.raw if prototype.is_contrastive else score.probability
                    decision = values > threshold
                allowed = gate_pass.get(prototype.gate) if prototype.gate else None
                for index in range(len(ids)):
                    if allowed is not None and not allowed[index]:
                        report.gated_out += 1
                        continue
                    key = prototype.payload_key
                    if prototype.kind == KIND_ATTRIBUTE:
                        payloads[index][key] = bool(decision[index])
                        if store_scores:
                            payloads[index][f"{key}_score"] = round(
                                float(score.probability[index]), 3
                            )
                    elif decision[index]:
                        # Категорії — лише ті, що спрацювали.
                        payloads[index][key] = True
                        if store_scores:
                            payloads[index][f"{key}_score"] = round(
                                float(score.probability[index]), 3
                            )
                        report.positive += 1
                    if prototype.kind == KIND_ATTRIBUTE and decision[index]:
                        report.positive += 1
                    report.scanned += 1

            _resolve_exclusive(payloads, scores, chosen, region_types)
            self.store.set_payloads(collection, list(ids), payloads)
            for report in reports.values():
                report.updated += len(ids)

        elapsed = time.perf_counter() - started
        for report in reports.values():
            report.elapsed_s = elapsed / max(1, len(reports))
        logger.info(
            "застосовано %d прототипів до %s за %.1fс", len(chosen), collection, elapsed
        )
        return reports

    def _iter_with_types(self, collection: str, batch_size: int, *, scope_filter=None):
        """Вектори разом із типом регіону — потрібно для фасетів, що залежать
        від того, ЩО саме вирізано: колір обʼєкта осмислений, колір плитки ні.

        Обхід один на весь проєкт (`VectorStore.iter_vectors`). Тут колись
        жила його копія, і коли розріджений шар зробив вектори словниками,
        виправлення оригіналу цієї копії не торкнулося — фасети перестали
        рахуватися без жодної помилки в індексації.
        """
        for ids, vectors, payloads in self.store.iter_vectors(
            collection, batch_size, payload_fields=["region_type"],
            query_filter=scope_filter,
        ):
            yield ids, vectors, [p.get("region_type", "") for p in payloads]

    def _corpus_thresholds(
        self, collection: str, names: Sequence[str], sample: int
    ) -> dict[str, float]:
        """Межі для всіх прототипів одразу, з однієї вибірки.

        Вибірка береться з УСІЄЇ колекції, а не з обмеженої `scope_filter`
        частини: межа має описувати корпус, а не щойно додану партію.
        """
        collected: list = []
        for _, vectors in self.store.iter_vectors(collection, batch_size=512):
            collected.extend(vectors)
            if len(collected) >= sample:
                break
        if len(collected) < 8:
            return {}
        matrix = np.asarray(collected[:sample], dtype=np.float32)
        thresholds: dict[str, float] = {}
        for name in names:
            prototype = self.bank.get(name)
            if prototype.calibrated:
                continue
            threshold, _ = self.bank.suggest_threshold(name, matrix)
            if threshold is not None:
                thresholds[name] = threshold
        return thresholds

    def _remembered_thresholds(
        self, collection: str, names, sample: int, catalog
    ) -> dict[str, float]:
        """Корпусні межі: збережені, поки корпус істотно не виріс.

        Без цього інкрементальний прохід був би НЕПРАВИЛЬНИМ, а не просто
        швидшим: межа виводиться з розподілу, і взята з однієї нової партії
        вона розмітила б нові точки інакше, ніж уже розмічений індекс.
        """
        if catalog is None:
            return self._corpus_thresholds(collection, names, sample)

        total = self.store.count(collection, exact=False) if self.store else 0
        remembered: dict[str, float] = {}
        stale: list[str] = []
        for name in names:
            # Прототип, калібрований на розмітці, корпусної межі не потребує:
            # вона знає менше за розмітку і все одно була б відкинута. Раніше
            # такі імена вважалися «незбереженими» і через них запускався
            # повний обхід корпусу на КОЖНІЙ індексації — тобто інкремент
            # працював скрізь, крім найдорожчого місця.
            if self.bank.get(name).calibrated:
                continue
            found = catalog.corpus_threshold(collection, name)
            if found is None:
                stale.append(name)
                continue
            value, at_points = found
            grown = (total - at_points) / at_points if at_points else 1.0
            if grown >= self.THRESHOLD_REFRESH_GROWTH:
                stale.append(name)
            elif value is not None:
                remembered[name] = value
        if stale:
            fresh = self._corpus_thresholds(collection, stale, sample)
            # Записуємо і НЕВДАЛІ спроби. «Розриву немає» — це знання про
            # корпус, і без нього такі категорії щоразу тягли повний обхід.
            for name in stale:
                catalog.save_corpus_threshold(collection, name, fresh.get(name), total)
            remembered.update(fresh)
        return remembered

    def apply_all(
        self, collection: str, *, kinds: tuple[str, ...] | None = None, **kwargs
    ) -> list[ApplyReport]:
        """Застосувати прототипи вибраних типів, передумови — першими.

        `kinds` обмежує типи фасетів, і це не оптимізація, а вимога
        коректності. Атрибути людини мають сенс ЛИШЕ на регіонах: на рівні
        кадру `attr_glasses` разом із `cat_person` утворюють конʼюнкцію, яка
        стверджує те, чого в кадрі немає. Кадр із чоловіком у синій куртці та
        жінкою в червоній задовольнив би фільтр «чоловік у червоній куртці».

        Порядок теж має значення: атрибут спирається на фасет-передумову, тож
        той мусить бути порахований раніше.
        """
        selected = [p for p in self.bank if kinds is None or p.kind in kinds]
        # Передумови додаємо навіть якщо їхній тип не обраний: без них
        # атрибути не змогли б обчислитися взагалі.
        gates = {p.gate for p in selected if p.gate}
        names = {p.name for p in selected} | gates
        ordered = sorted(names, key=lambda n: (n not in gates, n))
        return [self.apply_to_collection(collection, name, **kwargs) for name in ordered]
