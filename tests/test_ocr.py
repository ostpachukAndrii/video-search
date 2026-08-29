"""Тести обробки результату OCR — без завантаження моделі."""

from __future__ import annotations

from vsearch.represent.ocr import EMPTY_MARKERS, MAX_TEXT_LEN, OcrResult, clean


class TestОчищення:
    def test_заглушка_відсутності_тексту_не_стає_текстом(self):
        """Florence-2 віддає «-» там, де тексту немає.

        Потрапивши в індекс, така заглушка стала б термом, спільним для всіх
        кадрів без тексту, — і збігалася б із чим завгодно.
        """
        for marker in EMPTY_MARKERS:
            assert clean(marker) == ""

    def test_рядок_без_літер_і_цифр_це_не_текст(self):
        """Перелік заглушок неповний за побудовою.

        Модель віддавала і «-», і «.». Надійніша ознака — відсутність будь-якої
        літери чи цифри: такий рядок не дасть жодного терма, зате виглядатиме
        в матеріалах справи як прочитаний текст.
        """
        for junk in (".", "...", "!?", "  ~  "):
            assert clean(junk) == ""

    def test_перенос_рядка_стає_пробілом(self):
        assert clean("18:53\n5G\nShare") == "18:53 5G Share"

    def test_довгий_текст_обрізається(self):
        assert len(clean("слово " * 5000)) <= MAX_TEXT_LEN

    def test_справжній_текст_зберігається(self):
        assert clean("  Augustiner-Bräu   München  ") == "Augustiner-Bräu München"


class TestРезультат:
    def test_порожній_текст_це_відсутність_тексту(self):
        assert not OcrResult("", "florence2").has_text

    def test_непокриті_писемності_повідомляються(self):
        """Порожньо через «немає тексту» і через «не вміє читати» — різне.

        Без цієї різниці запит івритом мовчки нічого не знаходив би, і
        виглядало б це як відсутність матеріалу, а не як межа рушія.
        """
        result = OcrResult("", "florence2", unsupported=("hebrew",))
        assert "hebrew" in result.unsupported
