"""Юніт-тести механізму прототипів — без завантаження моделей.

Логіка оцінювання, порогів і калібрування перевіряється на підставному
ембедері: вона не залежить від того, яка саме модель дала вектори, і має
лишатися перевіреною навіть там, де 4.5 ГБ ваг немає.
"""

from __future__ import annotations

import numpy as np
import pytest

from vsearch.represent.prototypes import (
    BACKGROUND_PROMPTS,
    KIND_ATTRIBUTE,
    Prototype,
    PrototypeBank,
)


class FakeEmbedder:
    """Ембедер із наперед заданою відповідністю «фраза → вектор»."""

    dim = 4

    def __init__(self, mapping: dict[str, list[float]]):
        self.mapping = mapping
        self.calibration = (10.0, 0.0)

    def embed_texts(self, texts, use_template=True):
        rows = [self.mapping.get(t, [0.0, 0.0, 0.0, 1.0]) for t in texts]
        matrix = np.asarray(rows, dtype=np.float32)
        return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)

    def embed_images(self, images):
        matrix = np.asarray(images, dtype=np.float32)
        return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


@pytest.fixture
def bank():
    return PrototypeBank(
        FakeEmbedder({
            "cat": [1.0, 0.0, 0.0, 0.0],
            "dog": [0.0, 1.0, 0.0, 0.0],
            "feline animal": [0.9, 0.1, 0.0, 0.0],
        })
    )


class TestРеєстрація:
    def test_тло_підставляється_автоматично(self, bank):
        # Без негативу фасет непридатний — виміряно на M3 (F1 0.00–0.20).
        prototype = bank.add(Prototype(name="cat", positive=("cat",)))
        assert prototype.negative == BACKGROUND_PROMPTS
        assert prototype.is_contrastive

    def test_явний_негатив_не_замінюється(self, bank):
        prototype = bank.add(
            Prototype(name="cat", positive=("cat",), negative=("dog",))
        )
        assert prototype.negative == ("dog",)

    def test_абсолютний_режим_вмикається_свідомо(self, bank):
        prototype = bank.add(
            Prototype(name="cat", positive=("cat",), auto_background=False)
        )
        assert not prototype.is_contrastive

    def test_ансамбль_формулювань_усереднюється(self, bank):
        bank.add(Prototype(name="cat", positive=("cat", "feline animal")))
        vector = bank._positive["cat"]
        assert np.isclose(np.linalg.norm(vector), 1.0)

    def test_невідомий_прототип_підказує_відомі(self, bank):
        bank.add(Prototype(name="cat", positive=("cat",)))
        with pytest.raises(KeyError, match="cat"):
            bank.get("немає")

    def test_приклади_підмішуються_і_рахуються(self, bank):
        prototype = bank.add(
            Prototype(name="cat", positive=("cat",)),
            examples=[[1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0]],
        )
        assert prototype.example_count == 2


class TestКлючіPayload:
    def test_категорія_і_атрибут_у_різних_просторах_імен(self):
        category = Prototype(name="glasses", positive=("x",))
        attribute = Prototype(name="glasses", positive=("x",), kind=KIND_ATTRIBUTE)
        assert category.payload_key == "cat_glasses"
        assert attribute.payload_key == "attr_glasses"
        assert category.payload_key != attribute.payload_key


class TestОцінювання:
    def test_контрастний_режим_розрізняє(self, bank):
        bank.add(Prototype(name="cat", positive=("cat",), negative=("dog",)))
        vectors = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        scores = bank.score(vectors, "cat")
        assert scores.decision.tolist() == [True, False]
        assert scores.raw[0] > scores.raw[1]

    def test_ймовірність_у_межах_нуль_один(self, bank):
        bank.add(Prototype(name="cat", positive=("cat",), negative=("dog",)))
        vectors = np.random.default_rng(0).normal(size=(20, 4)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        scores = bank.score(vectors, "cat")
        assert (scores.probability >= 0).all() and (scores.probability <= 1).all()

    def test_поріг_зсуває_рішення(self, bank):
        # Різниця косинусів лежить у [-2, 2]; поріг 1.5 недосяжний для
        # ортогональних позитиву й негативу, де максимум становить 1.0.
        bank.add(Prototype(name="cat", positive=("cat",), negative=("dog",), threshold=1.5))
        vectors = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        scores = bank.score(vectors, "cat")
        assert scores.raw[0] == pytest.approx(1.0)
        assert not scores.decision[0]


class TestПорігБезРозмітки:
    def test_відсутня_категорія_дає_порожньо(self, bank):
        # Знак різниці сам відкидає те, чого в корпусі немає.
        bank.add(Prototype(name="dog", positive=("dog",), negative=("cat",)))
        vectors = np.array([[1.0, 0.0, 0.0, 0.0]] * 6, dtype=np.float32)
        threshold, decision = bank.suggest_threshold("dog", vectors)
        assert threshold == 0.0
        assert not decision.any()

    def test_виразний_розрив_знаходиться(self, bank):
        bank.add(Prototype(name="cat", positive=("cat",), negative=("dog",)))
        vectors = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.98, 0.02, 0.0, 0.0]]
            + [[0.5, 0.5, 0.0, 0.0]] * 10,
            dtype=np.float32,
        )
        threshold, decision = bank.suggest_threshold("cat", vectors)
        assert threshold > 0
        assert decision[:2].all(), "дві виразні цілі мали потрапити"


class TestКалібрування:
    def test_поріг_підбирається_і_позначається(self, bank):
        bank.add(Prototype(name="cat", positive=("cat",), negative=("dog",)))
        vectors = np.array(
            [[1.0, 0.0, 0.0, 0.0]] * 5 + [[0.0, 1.0, 0.0, 0.0]] * 5, dtype=np.float32
        )
        labels = [True] * 5 + [False] * 5
        prototype, metrics = bank.calibrate("cat", vectors, labels)
        assert prototype.calibrated
        assert metrics["f1"] == pytest.approx(1.0)
        assert metrics["positives"] == 5 and metrics["negatives"] == 5

    def test_замало_прикладів_це_помилка(self, bank):
        bank.add(Prototype(name="cat", positive=("cat",), negative=("dog",)))
        vectors = np.array([[1.0, 0.0, 0.0, 0.0]] * 4, dtype=np.float32)
        with pytest.raises(ValueError, match="замало"):
            bank.calibrate("cat", vectors, [True, True, False, False])

    def test_розбіжність_довжин_це_помилка(self, bank):
        bank.add(Prototype(name="cat", positive=("cat",), negative=("dog",)))
        with pytest.raises(ValueError, match="розбіжність"):
            bank.calibrate("cat", np.zeros((3, 4), dtype=np.float32), [True, False])


class TestПередумова:
    def test_атрибути_людини_мають_передумову(self):
        from vsearch.represent.categories import PERSON_ATTRIBUTES

        for prototype in PERSON_ATTRIBUTES:
            assert prototype.gate == "person", (
                f"{prototype.name} без передумови: «без окулярів» означало б "
                f"і «на кадрі немає людини»"
            )

    def test_атрибути_обʼєкта_передумови_не_мають(self):
        """Колір стосується чого завгодно, не лише людини.

        Передумова `person` тут була б помилкою в інший бік: колір велосипеда
        чи сумки не визначався б узагалі, бо вони не люди.
        """
        from vsearch.represent.categories import OBJECT_ATTRIBUTES

        for prototype in OBJECT_ATTRIBUTES:
            assert not prototype.gate, (
                f"{prototype.name} має передумову {prototype.gate!r}: колір "
                f"обʼєкта не залежить від наявності людини"
            )

    def test_передумова_є_серед_категорій(self):
        from vsearch.represent.categories import DEFAULT_ATTRIBUTES, DEFAULT_CATEGORIES

        names = {p.name for p in DEFAULT_CATEGORIES}
        for prototype in DEFAULT_ATTRIBUTES:
            if not prototype.gate:
                continue
            assert prototype.gate in names, f"передумову {prototype.gate!r} не зареєстровано"


class TestСинхронізаціяСловників:
    """Парсер не має права вимагати того, чого індекс не знає.

    Розсинхрон тут не дає помилки — фільтр на невідомий фасет просто ніколи не
    збігається. Жорсткий прохід повертає нуль, пошук мовчки падає в мʼякий, і
    користувач бачить лише «застосовано мʼякий пошук», не знаючи чому.
    Саме так і сталося з класами `animal`, `building` і `phone`.
    """

    def test_кожен_клас_обʼєкта_має_категорію_в_індексі(self):
        from vsearch.represent.categories import DEFAULT_CATEGORIES
        from vsearch.search.query_model import ObjectClass

        indexed = {p.name for p in DEFAULT_CATEGORIES}
        # OTHER навмисно без категорії: це «щось, чого ми не класифікуємо»,
        # і умови для нього не будуються взагалі.
        required = {c.value for c in ObjectClass} - {ObjectClass.OTHER.value}
        missing = required - indexed
        assert not missing, (
            f"парсер може вимагати {sorted(missing)}, але індекс таких категорій "
            f"не рахує — фільтр не збігатиметься ніколи"
        )

    def test_кожен_атрибут_парсера_дає_придатне_формулювання(self):
        """Передобчислений прототип більше НЕ потрібен — потрібне формулювання.

        Раніше тут перевірялося, що кожен атрибут має прототип в індексі, і це
        загонило в перелік: «рожевого» не було, тож ознака губилася. Тепер
        ознака стає прототипом із власного тексту, тож вимога інша — з неї має
        виходити осмислена фраза.
        """
        from vsearch.search.query_model import AttributeName, attribute_phrases

        for attribute in AttributeName:
            phrases = attribute_phrases(attribute.value, "true")
            assert phrases and all(p.strip() for p in phrases), (
                f"{attribute.value} не дає формулювання для прототипу"
            )
            assert not any("true" in p for p in phrases), (
                f"{attribute.value} описується словом «true» замість власної "
                f"назви — такий прототип нічого не означає"
            )

    def test_довільна_ознака_теж_дає_формулювання(self):
        """Відкритий словник: ознака, якої немає в жодному переліку.

        Саме заради цього прибрано перелічені кольори. Користувач пише
        «пастельно-рожевий» або «морквяний» — і воно має працювати.
        """
        from vsearch.search.query_model import attribute_phrases

        for value in ("pastel pink", "carrot orange", "camouflage"):
            phrases = attribute_phrases("color", value)
            assert any(value in p for p in phrases), (
                f"відтінок {value!r} не потрапив у формулювання прототипу"
            )

    def test_клас_other_не_породжує_умов(self):
        from vsearch.search.query_model import Entity, ObjectClass, StructuredQuery

        query = StructuredQuery(
            query_en="x", must=[Entity(object=ObjectClass.OTHER, attributes=[])]
        )
        assert query.entity_conditions() == [[]], (
            "OTHER не має давати фільтра: інакше кожен банер чи парасолька "
            "звужували б пошук до нуля"
        )
