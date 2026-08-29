"""Схема розібраного запиту.

Окремий модуль без залежностей від LLM: за цією схемою будується GBNF-граматика,
за нею ж валідується результат, і її ж читають тести. Модель, що її заповнює,
може змінитися — контракт лишиться.

Ключове рішення — **закриті переліки**. Атрибути, предикати й класи обʼєктів
задані як Enum, а не вільним текстом. Під GBNF це означає, що модель фізично
не може вигадати неіснуючий атрибут: граматика не дозволить згенерувати такий
токен. Вільний текст тут був би джерелом мовчазних поломок пошуку.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class ObjectClass(str, Enum):
    """Класи обʼєктів, на які система вміє посилатися."""

    PERSON = "person"
    VEHICLE = "vehicle"
    BAG = "bag"
    WEAPON = "weapon"
    DOCUMENT = "document"
    PHONE = "phone"
    ANIMAL = "animal"
    BUILDING = "building"
    OTHER = "other"


class AttributeName(str, Enum):
    """Атрибути, які система вміє перевіряти фасетами.

    Перелік навмисно збігається з DEFAULT_ATTRIBUTES у categories.py:
    парсер не має права вимагати того, чого індекс не знає.
    """

    GLASSES = "glasses"
    HEADWEAR = "headwear"
    MASK = "mask"
    BEARD = "beard"
    GENDER = "gender"
    AGE_BAND = "age_band"
    COLOR = "color"
    HAIR = "hair"
    #: Як людина ВИГЛЯДАЄ — вільним текстом: одяг, оголений торс, татуювання.
    #:
    #: Решта ознак мають закриті значення, ця — ні, і це навмисно. Перелічити
    #: одяг неможливо (ADR-009), а купальник, спідниця чи камуфляж мусять
    #: перевірятися САМЕ НА КРОПІ ЛЮДИНИ, а не десь у кадрі: інакше «дівчина в
    #: купальнику» задовольняється дівчиною та чужим купальником поруч.
    #:
    #: Передобчисленого фасета для неї немає й не буде — вона йде прототипом
    #: на льоту по вже піднятих кандидатах.
    WEARING = "wearing"


#: Як атрибут із запиту лягає у фасет індексу.
#:
#: Парсер мислить категоріями («стать = чоловіча»), індекс — булевими
#: фасетами («gender_male = так»). Без цієї таблиці умова `attr_gender=male`
#: шукала б неіснуючий ключ, запобіжник її відкидав би, і запит «чоловік»
#: тихо перетворювався б на «людина» — саме так і було.
ATTRIBUTE_TO_FACET: dict[tuple[str, str], tuple[str, bool]] = {
    ("gender", "male"): ("gender_male", True),
    ("gender", "female"): ("gender_male", False),
    ("age_band", "adult"): ("adult", True),
    ("age_band", "child"): ("adult", False),
    ("age_band", "senior"): ("adult", True),
    # Колір — один ключ зі значенням, а не шість булевих: кроп має рівно один
    # переважний колір, і фільтр має бути точним вибором.
    **{
        ("color", c): ("color", c)
        for c in ("white", "black", "red", "blue", "green", "yellow",
                  "pink", "orange", "purple", "brown", "grey", "beige")
    },
    **{("hair", h): ("hair", h)
       for h in ("blonde", "dark", "red_hair", "grey_hair")},
    # Синоніми значень. Модель описує волосся як завгодно — «brown», «dark»,
    # «блондинка» — а індекс знає рівно пʼять значень. Без цієї таблиці
    # `attr_hair=brown` шукав би неіснуюче й мовчки нічого не знаходив,
    # як і сталося на запиті «брюнетка в рожевому платті».
    **{("hair", alias): ("hair", canonical) for alias, canonical in (
        ("blonde", "blonde"), ("blond", "blonde"), ("fair", "blonde"),
        ("блондинка", "blonde"), ("блондин", "blonde"), ("світле", "blonde"),
        ("brunette", "dark"), ("brown", "dark"), ("dark", "dark"),
        ("брюнетка", "dark"), ("брюнет", "dark"), ("темне", "dark"),
        ("black", "dark"), ("black_hair", "dark"), ("чорне", "dark"),
        ("red", "red_hair"), ("ginger", "red_hair"), ("red_hair", "red_hair"),
        ("руда", "red_hair"), ("рудий", "red_hair"),
        ("grey", "grey_hair"), ("gray", "grey_hair"), ("white", "grey_hair"),
        ("grey_hair", "grey_hair"), ("сивий", "grey_hair"), ("сиве", "grey_hair"),
    )},
    # Українські назви — парсер нормалізує в англійську, але не завжди.
    **{
        ("color", ua): ("color", en)
        for ua, en in (("білий","white"), ("чорний","black"), ("червоний","red"),
                       ("синій","blue"), ("зелений","green"), ("жовтий","yellow"),
                       ("рожевий","pink"), ("помаранчевий","orange"),
                       ("коричневий","brown"), ("сірий","grey"), ("бежевий","beige"))
    },
}


class CategoryName(str, Enum):
    """Категорії сцени, які індекс уміє перевіряти фільтром.

    Перелік закритий і збігається з `DEFAULT_CATEGORIES` у categories.py. Під
    GBNF це означає, що модель фізично не може попросити неіснуючу категорію.

    Потрібне тому, що класів обʼєктів у запиті девʼять, а категорій — 36.
    Двадцять вісім із них не мали жодного способу потрапити в запит: система
    рахувала `swimwear`, `nudity`, `blood`, `tattoo`, `crowd` при кожній
    індексації й ніколи ними не користувалася. «Дівчина в купальнику»
    зводилася до «дівчина плюс щось», бо купальник ставав безіменним `other`.
    """

    PERSON = "person"
    FACE_VISIBLE = "face_visible"
    CHILD = "child"
    TATTOO = "tattoo"
    SCREENSHOT = "screenshot"
    HANDWRITING = "handwriting"
    UNDERWEAR = "underwear"
    NUDITY = "nudity"
    SWIMWEAR = "swimwear"
    UNIFORM = "uniform"
    WEAPON = "weapon"
    FIREARM = "firearm"
    BLADE = "blade"
    AMMUNITION = "ammunition"
    MONEY = "money"
    DRUGS = "drugs"
    SYRINGE = "syringe"
    DOCUMENT = "document"
    LICENSE_PLATE = "license_plate"
    PHONE = "phone"
    ELECTRONICS = "electronics"
    STORAGE_MEDIA = "storage_media"
    BAG = "bag"
    JEWELRY = "jewelry"
    VEHICLE = "vehicle"
    ANIMAL = "animal"
    INDOOR = "indoor"
    OUTDOOR = "outdoor"
    NIGHT = "night"
    BEDROOM = "bedroom"
    BATHROOM = "bathroom"
    VEHICLE_INTERIOR = "vehicle_interior"
    BUILDING = "building"
    CROWD = "crowd"
    BLOOD = "blood"
    DAMAGE = "damage"


class Predicate(str, Enum):
    """Просторові відношення між обʼєктами.

    Розділені за вартістю перевірки, і це не дрібниця:

    * геометричні (`inside`, `left_of`, `right_of`, `above`, `below`, `near`)
      виводяться з координат рамок і коштують нуль;
    * семантичні (`holding`, `wearing`, `entering`) з геометрії не виводяться
      й потребують VLM на етапі верифікації.
    """

    INSIDE = "inside"
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    ABOVE = "above"
    BELOW = "below"
    NEAR = "near"
    HOLDING = "holding"
    WEARING = "wearing"


#: Предикати, які перевіряються самою лише геометрією рамок.
GEOMETRIC_PREDICATES = frozenset({
    Predicate.INSIDE, Predicate.LEFT_OF, Predicate.RIGHT_OF,
    Predicate.ABOVE, Predicate.BELOW, Predicate.NEAR,
})


class Attribute(BaseModel):
    name: AttributeName
    value: str = Field(description="значення: true/false для булевих, інакше слово")

    @property
    def as_bool(self) -> bool | None:
        lowered = self.value.strip().lower()
        if lowered in ("true", "yes", "так"):
            return True
        if lowered in ("false", "no", "ні"):
            return False
        return None


class Entity(BaseModel):
    """Обʼєкт запиту разом із ознаками, привʼязаними САМЕ до нього.

    Звʼязування живе тут: `person{gender:male, color:red}` означає одну людину
    з обома ознаками, а не «десь у кадрі є чоловік і десь є червоне».
    """

    object: ObjectClass
    attributes: list[Attribute] = Field(default_factory=list)
    #: Скільки таких обʼєктів має бути в кадрі.
    #:
    #: «дві дівчини в синіх спідницях» і «дівчина в синій спідниці» — різні
    #: запити, і досі система бачила їх однаково. Кількість перевіряється
    #: РІЗНИМИ регіонами одного кадру: два різні кропи, кожен із власним
    #: підтвердженням. Одна людина, порахована двічі, кількості не утворює.
    #: Закритий перелік, а не число з межами: GBNF обмежує ФОРМУ, а не
    #: діапазон, тож при `int` модель згенерувала «2222222222222222» і розбір
    #: упав на валідації. Перелік вона фізично не може порушити — це той
    #: самий прийом, що й для атрибутів (ADR-007).
    #: Скільки таких сутностей просить запит («дві дівчини» → 2).
    #:
    #: Це ПОМІТКА, а не фільтр. Порахувати екземпляри система не вміє:
    #: обумовлена детекція заземлює фразу в ОДНУ ділянку, а плитки сітки
    #: різні за геометрією, а не за вмістом (ADR-022). Кадр, де підтверджено
    #: менше, лишається у видачі з поміткою «екземплярів 1 з 2».
    #:
    #: Тип перелічувальний, бо GBNF обмежує ФОРМУ, а не діапазон: без нього
    #: модель видавала `2222222222222222`.
    count: Literal[1, 2, 3, 4, 5] = 1

    def attribute(self, name: AttributeName) -> Attribute | None:
        return next((a for a in self.attributes if a.name == name), None)

    def describe(self, *, ascii_only: bool = False) -> str:
        """Короткий людський опис сутності — підпис рамки в інтерфейсі.

        Слідчому потрібно бачити не «person{gender=female}», а «жінка». І
        головне — бачити РІЗНІ підписи на різних рамках: на запит «дівчина з
        дитиною» кадр має показати рамку «жінка» й рамку «дитина» окремо,
        інакше неможливо перевірити, чи система знайшла обох, чи двічі одну.

        `ascii_only` дає латинський варіант — для тексту, який малюється
        ПІКСЕЛЯМИ поверх фото. Шрифт, вбудований у Pillow, кирилиці не має, і
        українські підписи виходили порожніми прямокутниками. Системний шрифт
        брати не можна: середовище виконання не має мережі й може не мати
        нічого, крім образу, тож знайдений на ноутбуці шрифт зник би в
        продакшені. Підпис під фото — звичайний markdown, його малює браузер,
        і там лишається українська.
        """
        names = _OBJECT_NAMES_EN if ascii_only else _OBJECT_NAMES
        heads = _PERSON_HEADS_EN if ascii_only else _PERSON_HEADS
        attrs = _ATTRIBUTE_NAMES_EN if ascii_only else _ATTRIBUTE_NAMES

        gender = self.attribute(AttributeName.GENDER)
        age = self.attribute(AttributeName.AGE_BAND)

        if self.object is ObjectClass.PERSON:
            if age and age.value == "child":
                head = heads["child"]
            elif gender and gender.value == "female":
                head = heads["female"]
            elif gender and gender.value == "male":
                head = heads["male"]
            elif age and age.value == "senior":
                head = heads["senior"]
            else:
                head = heads["person"]
        else:
            head = names.get(self.object, self.object.value)

        extras = [
            attrs.get((a.name.value, a.value.strip().lower()))
            or attrs.get((a.name.value, ""), "").format(value=a.value)
            for a in self.attributes
            if a.name not in (AttributeName.GENDER, AttributeName.AGE_BAND)
        ]
        extras = [e for e in extras if e]
        # Кількість у підпис РАМКИ не входить: рамка позначає один екземпляр,
        # а «2×» — властивість запиту. Інакше на кадрі стояло б «2× woman»
        # двічі, ніби там чотири людини.
        return f"{head} ({', '.join(extras)})" if extras else head

    def describe_query(self, *, ascii_only: bool = False) -> str:
        """Опис для панелі розбору, де кількість якраз доречна."""
        described = self.describe(ascii_only=ascii_only)
        return f"{self.count}× {described}" if self.count > 1 else described


#: Голова опису людини. Вибір за ознаками, а не за класом.
_PERSON_HEADS: dict[str, str] = {
    "child": "дитина", "female": "жінка", "male": "чоловік",
    "senior": "літня людина", "person": "людина",
}
_PERSON_HEADS_EN: dict[str, str] = {
    "child": "child", "female": "woman", "male": "man",
    "senior": "senior", "person": "person",
}

#: Латинські назви — для тексту, що малюється поверх фото (див. `describe`).
_OBJECT_NAMES_EN: dict = {
    ObjectClass.VEHICLE: "vehicle",
    ObjectClass.BAG: "bag",
    ObjectClass.WEAPON: "weapon",
    ObjectClass.DOCUMENT: "document",
    ObjectClass.PHONE: "phone",
    ObjectClass.ANIMAL: "animal",
    ObjectClass.BUILDING: "building",
    ObjectClass.OTHER: "object",
}

_ATTRIBUTE_NAMES_EN: dict[tuple[str, str], str] = {
    ("glasses", "true"): "glasses",
    ("glasses", "false"): "no glasses",
    ("headwear", "true"): "headwear",
    ("headwear", "false"): "no headwear",
    ("mask", "true"): "mask",
    ("mask", "false"): "no mask",
    ("beard", "true"): "beard",
    ("beard", "false"): "no beard",
    ("color", ""): "{value}",
    ("hair", ""): "{value} hair",
    ("wearing", ""): "{value}",
}

#: Людські назви класів обʼєктів для підписів у видачі.
_OBJECT_NAMES: dict = {
    ObjectClass.VEHICLE: "транспорт",
    ObjectClass.BAG: "сумка",
    ObjectClass.WEAPON: "зброя",
    ObjectClass.DOCUMENT: "документ",
    ObjectClass.PHONE: "телефон",
    ObjectClass.ANIMAL: "тварина",
    ObjectClass.BUILDING: "будівля",
    ObjectClass.OTHER: "обʼєкт",
}

#: Людські назви ознак. Ключ ("color", "") — шаблон для будь-якого значення.
_ATTRIBUTE_NAMES: dict[tuple[str, str], str] = {
    ("glasses", "true"): "в окулярах",
    ("glasses", "false"): "без окулярів",
    ("headwear", "true"): "у головному уборі",
    ("headwear", "false"): "без головного убору",
    ("mask", "true"): "у масці",
    ("mask", "false"): "без маски",
    ("beard", "true"): "з бородою",
    ("beard", "false"): "без бороди",
    ("color", ""): "колір: {value}",
    ("hair", ""): "волосся: {value}",
    ("wearing", ""): "у {value}",
}


class Relation(BaseModel):
    """Відношення між двома обʼєктами запиту, заданими їхніми позиціями."""

    subject: int = Field(ge=0, description="позиція обʼєкта у must")
    predicate: Predicate
    target: int = Field(ge=0, description="позиція обʼєкта у must")

    @property
    def is_geometric(self) -> bool:
        return self.predicate in GEOMETRIC_PREDICATES


class StructuredQuery(BaseModel):
    """Результат розбору запиту будь-якою мовою."""

    query_en: str = Field(description="канонічний опис англійською для ембедера")
    language: str = Field(default="unknown", description="мова оригіналу, код ISO")
    must: list[Entity] = Field(default_factory=list)
    must_not: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    categories: list[CategoryName] = Field(
        default_factory=list,
        description="категорії сцени з закритого переліку, які мають бути в кадрі",
    )

    @property
    def is_compositional(self) -> bool:
        """Чи привʼязує запит ознаки до обʼєктів.

        Від цього залежить, де шукати: композитний запит іде по регіонах, бо
        на рівні кадру ознаки не звʼязані (див. ADR-006). Сценовий запит
        («нічна вулиця») може використовувати й кадри.
        """
        if self.relations or self.must_not:
            return True
        return any(entity.attributes for entity in self.must)

    @property
    def has_negation(self) -> bool:
        return bool(self.must_not)

    @property
    def needs_vlm_verification(self) -> bool:
        """Чи є умови, які геометрія й фасети перевірити не можуть."""
        return any(not relation.is_geometric for relation in self.relations)

    def entity_conditions(self) -> list[list[tuple[str, object]]]:
        """Умови `must` — ОКРЕМИМ списком на кожну сутність.

        Не одним пласким списком, і це принципово. Запит «чоловік біля
        червоної машини» містить дві сутності; зліплені в один фільтр, вони
        вимагали б регіону, який водночас є людиною і автомобілем — тобто
        не знайшли б нічого ніколи.

        Правильно інакше: кожна сутність шукається окремо по регіонах своїм
        конʼюнктивним фільтром (звʼязування в межах однієї точки), а потім
        множини кадрів перетинаються за `frame_id`. Саме цей другий прохід і
        закриває обмеження, зафіксоване на M4a.
        """
        return [_conditions(entity) for entity in self.must]

    def excluded_conditions(self) -> list[list[tuple[str, object]]]:
        return [_conditions(entity) for entity in self.must_not]

    def category_conditions(self) -> list[list[tuple[str, object]]]:
        """Категорії сцени — окремими умовами, по одній на список.

        Кожна вимагає СВОГО регіону в кадрі, як і сутність: купальник і людина
        можуть бути різними ділянками. Обʼєднати їх в одну умову означало б
        вимагати, щоб одна ділянка була водночас людиною й купальником.

        Категорії, що дублюють клас обʼєкта з `must`, відкидаються: інакше
        «жінка з сумкою» додала б до сумки ще одну вимогу сумки й нічого не
        змінила б, лише звузивши вибірку вдвічі без причини.
        """
        already = {
            entity.object.value for entity in self.must
            if entity.object is not ObjectClass.OTHER
        }
        return [
            [(f"cat_{name.value}", True)]
            for name in dict.fromkeys(self.categories)
            if name.value not in already
        ]


#: Слова запиту, які ДОСЛІВНО називають категорію, що індекс уже обчислює.
#:
#: Спроба довірити вибір категорій парсерові провалилася: він їх вигадував —
#: «дівчина в синій спідниці» давала `underwear, swimwear, outdoor`, і три
#: неіснуючі умови прибрали потрібне фото з видачі (Recall@5 0.90 → 0.77).
#:
#: Тут інакше: збіг ДОСЛІВНИЙ, ціле слово в тексті запиту. Вигадувати нема
#: чого — якщо людина написала «nudity», вона написала саме це. Умова додається
#: не фільтром, а ознакою: її впевненість бере участь у злитті свідчень
#: (ADR-019) нарівні з іншими.
_CATEGORY_WORDS = {name.value.replace("_", " "): name for name in CategoryName}


def categories_in_query(*texts: str) -> list[CategoryName]:
    """Категорії, дослівно названі в тексті запиту.

    Порівнюються ЦІЛІ слова, а не підрядки: інакше «underwear» знаходився б
    усередині випадкових слів, а «bag» — у «baggage».
    """
    found: dict[CategoryName, None] = {}
    for text in texts:
        if not text:
            continue
        words = [w for w in re.split(r"[^\w]+", text.lower()) if w]
        joined = " ".join(words)
        for phrase, name in _CATEGORY_WORDS.items():
            if " " in phrase:
                if f" {phrase} " in f" {joined} ":
                    found[name] = None
            elif phrase in words:
                found[name] = None
    return list(found)


def _conditions(entity: Entity) -> list[tuple[str, object]]:
    """Ознаки ОДНІЄЇ сутності — усі мають виконатися на одному регіоні."""
    pairs: list[tuple[str, object]] = []
    if entity.object is not ObjectClass.OTHER:
        pairs.append((f"cat_{entity.object.value}", True))
    for attribute in entity.attributes:
        mapped = ATTRIBUTE_TO_FACET.get(
            (attribute.name.value, attribute.value.strip().lower())
        )
        if mapped is not None:
            facet, expected = mapped
            pairs.append((f"attr_{facet}", expected))
            continue
        as_bool = attribute.as_bool
        value = as_bool if as_bool is not None else attribute.value.lower()
        pairs.append((f"attr_{attribute.name.value}", value))
    return pairs


#: Порожній розбір — те, що повертається, коли модель недоступна.
#: Пошук має деградувати до щільного, а не падати.
EMPTY = StructuredQuery(query_en="", language="unknown")


#: Як ознака з запиту перетворюється на текст для прототипу.
#:
#: Це ключ до відкритого словника. Ознака, якої немає в жодному переліку
#: («рожевий», «камуфляжний», «подертий»), стає прототипом просто з власного
#: значення — і оцінює вже піднятих кандидатів за частки мілісекунди.
#: Перелічувати наперед більше не треба.
_PHRASE_TEMPLATES: dict[str, tuple[str, ...]] = {
    "color": ("a mostly {value} object", "something {value} in colour"),
    "hair": ("a person with {value} hair",),
    "gender": ("a {value} person",),
    "age_band": ("a {value} person",),
    "glasses": ("a person wearing glasses",),
    "headwear": ("a person wearing a hat",),
    "mask": ("a person wearing a face mask",),
    "beard": ("a bearded person",),
    # Три формулювання, бо ознака покриває і одяг, і вигляд тіла. «a person
    # wearing a bare torso» звучить дивно, «a person with a bare torso» — ні,
    # а прототип усереднює всі три, тож зайве формулювання лише пом'якшує
    # похибку одного.
    "wearing": (
        "a person wearing {value}",
        "someone in {value}",
        "a person with {value}",
    ),
}

#: Загальний шаблон для ознак, яких немає у переліку вище. Саме він робить
#: словник відкритим: будь-яке слово з запиту стає придатним прототипом.
_GENERIC_TEMPLATE = "something {value}"

#: Проти ЧОГО протиставляти ознаку. Порожньо означає «проти загального тла».
#:
#: Для одягу тло не годиться, і це виміряно. Контраст «людина в купальнику
#: проти фотографії взагалі» відповідає на питання «чи це людина», а не «чи це
#: купальник»: кроп будь-якої людини далекий від «a photo», тож прототип
#: насичувався — 100% і на купальнику, і на ногах, і 87% там, де купальника
#: немає.
#:
#: Протиставляти треба В МЕЖАХ КАТЕГОРІЇ. Розрив між «є» і «немає» на трьох
#: типах одягу:
#:
#:     ознака            тло    одягнена людина   інший одяг
#:     купальник         41%          59%            47%
#:     синя спідниця     34%          39%            46%
#:     чорна сукня        7%          34%            28%
#:
#: «Одягнена людина» виграє в середньому (44% проти 27%) і не програє тлу
#: ЖОДНОГО разу.
_NEGATIVE_TEMPLATES: dict[str, tuple[str, ...]] = {
    "wearing": ("a fully dressed person", "a person wearing ordinary clothes"),
}


def attribute_negatives(name: str) -> tuple[str, ...]:
    """Проти чого протиставляти ознаку. Порожньо — проти загального тла."""
    return _NEGATIVE_TEMPLATES.get(name, ())


def attribute_phrases(name: str, value: object) -> tuple[str, ...]:
    """Ознака → формулювання для текстового прототипу.

    Булеві ознаки (окуляри, борода) описуються самою назвою; ознаки зі
    значенням (колір, волосся) підставляють значення у шаблон. Невідома
    ознака описується узагальнено — гірше за спеціальний шаблон, але значно
    краще за відсутність ознаки взагалі.
    """
    templates = _PHRASE_TEMPLATES.get(name)
    if templates is None:
        templates = (_GENERIC_TEMPLATE,)

    # Для булевої ознаки змістовна сама НАЗВА, а не значення: «камуфляжний»
    # описується словом «camouflage», а не словом «true». Без цього невідома
    # булева ознака давала прототип «something true», тобто ніщо.
    lowered = str(value).strip().lower()
    if isinstance(value, bool) or lowered in ("true", "false", "yes", "no"):
        text = name.replace("_", " ")
    else:
        text = lowered.replace("_", " ")
    return tuple(t.format(value=text).replace("  ", " ").strip() for t in templates)


def object_phrases(name: str) -> tuple[str, ...]:
    """Клас обʼєкта → формулювання для прототипу."""
    readable = name.replace("_", " ")
    return (f"a {readable}", f"a photo of a {readable}")
