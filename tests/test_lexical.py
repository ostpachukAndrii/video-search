"""Тести лексичного шару — чиста арифметика, без моделей."""

from __future__ import annotations

from vsearch.represent import lexical


class TestТокенізація:
    def test_літери_й_цифри_розділяються(self):
        """У номері літерна й цифрова частини помиляються незалежно."""
        terms = [t for t in lexical.tokenize("AA1234BB") if not t.startswith("#")]
        assert terms == ["aa", "1234", "bb"]

    def test_регістр_не_має_значення(self):
        assert lexical.tokenize("МЮНХЕН") == lexical.tokenize("мюнхен")

    def test_діакритика_складається(self):
        """OCR віддає розкладену форму, запит — складену; це те саме слово."""
        decomposed = "München"
        assert lexical.tokenize(decomposed) == lexical.tokenize("München")

    def test_односимвольні_слова_відкидаються(self):
        assert "a" not in lexical.tokenize("a sign")

    def test_окрема_цифра_лишається(self):
        assert "5" in lexical.tokenize("квартира 5")


class TestСтійкістьДоПомилокOCR:
    """Заради цього n-грами й додані.

    Помилка в одному символі робить пошук за цілим словом безрезультатним —
    саме в тому випадку, заради якого лексичний шар і потрібен.
    """

    def test_помилка_в_символі_лишає_спільні_терми(self):
        correct = set(lexical.tokenize("AA1234BB"))
        misread = set(lexical.tokenize("AA1Z34BB"))
        assert correct & misread, "жодного спільного терма — n-грами не працюють"

    def test_короткі_слова_на_нграми_не_розкладаються(self):
        assert not [t for t in lexical.tokenize("код") if t.startswith("#")]


class TestІдентифікаториТермів:
    def test_ідентифікатор_стабільний(self):
        assert lexical.term_id("münchen") == lexical.term_id("münchen")

    def test_ідентифікатор_невідємний_і_ненульовий(self):
        for term in ("a", "münchen", "1234", "#abc"):
            assert lexical.term_id(term) > 0

    def test_різні_терми_різні_ідентифікатори(self):
        ids = {lexical.term_id(t) for t in ("aa", "bb", "1234", "мюнхен")}
        assert len(ids) == 4


class TestПобудова:
    def test_порожній_текст_дає_порожній_вектор(self):
        assert lexical.build("").is_empty
        assert lexical.build("   ").is_empty

    def test_частота_зберігається(self):
        vec = lexical.build("сигнал сигнал шум")
        by_id = dict(zip(vec.indices, vec.values))
        assert by_id[lexical.term_id("сигнал")] == 2.0
        assert by_id[lexical.term_id("шум")] == 1.0

    def test_у_запиті_повтор_нічого_не_додає(self):
        vec = lexical.build_query("номер номер номер")
        assert set(vec.values) == {1.0}

    def test_індекси_впорядковані(self):
        vec = lexical.build("одне два три чотири пʼять")
        assert list(vec.indices) == sorted(vec.indices)


class TestПокриття:
    """Впевненість лексичного шару — величина, зіставна з візуальною."""

    def test_точний_рядок_дає_повне_покриття(self):
        share, found = lexical.coverage("PXCLD-1624", "тікет PXCLD-1624 у роботі")
        assert share == 1.0
        assert "pxcld" in found and "1624" in found

    def test_часткове_покриття_між_нулем_і_одиницею(self):
        share, _ = lexical.coverage("Augustiner Bräu München", "Augustiner вивіска")
        assert 0.0 < share < 1.0

    def test_відсутність_збігу_дає_нуль(self):
        share, found = lexical.coverage("гелікоптер", "краєвид без напису")
        assert share == 0.0 and found == ()

    def test_у_поясненні_немає_нграм(self):
        _, found = lexical.coverage("PXCLD-1624", "тікет PXCLD-1624")
        assert not [t for t in found if t.startswith("#")], (
            "n-грами потрібні пошуку, але в поясненні для людини безглузді"
        )


class TestТекстНаЗображенні:
    """Усе, що малюється ПІКСЕЛЯМИ, має бути видимим вбудованим шрифтом.

    Шрифт, вбудований у Pillow, не має кирилиці: підпис «жінка» малювався
    рядком однакових порожніх прямокутників. Помилка не давала ані винятку,
    ані попередження — лише нечитабельну картинку.

    Системний шрифт узяти не можна: середовище виконання без мережі може не
    мати нічого, крім образу. Тому перевіряється саме те, що обмежує: чи
    вбудований шрифт має гліфи для тексту, який ми збираємося намалювати.
    """

    def test_підписи_сутностей_малюються_без_порожніх_прямокутників(self):
        from PIL import ImageFont

        from vsearch.search.query_model import (
            Attribute, AttributeName, Entity, ObjectClass,
        )

        font = ImageFont.load_default(size=24)
        tofu = font.getbbox("\ufffe")[2]

        entities = [
            Entity(object=cls, attributes=attrs)
            for cls in ObjectClass
            for attrs in (
                [],
                [Attribute(name=AttributeName.GENDER, value="female")],
                [Attribute(name=AttributeName.AGE_BAND, value="child")],
                [Attribute(name=AttributeName.GLASSES, value="true")],
                [Attribute(name=AttributeName.COLOR, value="black")],
            )
        ]
        broken = []
        for entity in entities:
            text = entity.describe(ascii_only=True)
            for char in text:
                if char == " ":
                    continue
                if font.getbbox(char)[2] == tofu and font.getmask(char).getbbox():
                    broken.append((text, char))
        assert not broken, (
            f"шрифт не має гліфів для підписів, які ми малюємо: {broken[:5]}. "
            f"На фото це виглядає як рядок порожніх прямокутників."
        )
