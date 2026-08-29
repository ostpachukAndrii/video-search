"""Лексичний шар: точний збіг там, де щільні ембединги системно слабкі.

Ембединг відповідає на питання «про що це зображення». На питання «чи є тут
рядок AA1234BB» він не відповідає ніколи: номер, прізвище чи напис — це не
семантика, а послідовність символів, і найкраща у світі візуальна модель
розрізняє «AA1234BB» та «AA1234BC» гірше, ніж звичайний пошук підрядка.

Тому текст із кадру (OCR, згодом ASR) індексується ОКРЕМИМ розрідженим
вектором, а не домішується до щільного. Два ранжування зливаються RRF — так
само, як два мовні канали (ADR-013), і з тієї ж причини: величини з різних
шкал порівнювати не можна, ранги можна.

Розріджені вектори будуються КЛІЄНТОМ, а IDF рахує Qdrant (модифікатор `idf`
на розрідженому індексі). Це навмисно: серверне обчислення BM25 у Qdrant
вимагає інференсу моделі на його боці, а середовище виконання не має мережі.
Частоти термів — чиста арифметика, і жодних ваг для неї не треба.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

#: Токен — послідовність літер АБО цифр. Розділяти їх принципово: у номері
#: «AA1234BB» літерна й цифрова частини читаються OCR по-різному й помиляються
#: незалежно, тож як окремі терми вони дають більше шансів на збіг.
_TOKEN = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)

#: Коротші відкидаються: одиничні літери є в будь-якому тексті й лише
#: розмивають IDF. Цифри — виняток, бо «5» у номері квартири змістовне.
MIN_TOKEN_LEN = 2

#: Довжина символьних n-грам для стійкості до помилок OCR.
#:
#: OCR регулярно плутає 0/O, 1/I/l, 5/S. Пошук за цілим словом на такій
#: помилці не спрацьовує взагалі — а це саме той випадок, заради якого
#: лексичний шар і додається. Трійки символів дають часткове перекриття:
#: «AA1234BB» і «AA1Z34BB» ділять 4 спільні трійки з 6.
NGRAM = 3

#: З якої довжини слово додатково розкладається на n-грами. Короткі слова
#: розкладати немає сенсу: вони й так знаходяться цілком.
NGRAM_MIN_LEN = 4


@dataclass(frozen=True)
class SparseVec:
    """Розріджений вектор у вигляді, який приймає Qdrant."""

    indices: tuple[int, ...]
    values: tuple[float, ...]

    def __len__(self) -> int:
        return len(self.indices)

    @property
    def is_empty(self) -> bool:
        return not self.indices


def normalize(text: str) -> str:
    """Нижній регістр і NFKC.

    NFKC складає діакритику в канонічну форму: «München» з OCR і
    «München» із запиту інакше були б різними рядками, і збіг не відбувся б
    попри те, що для людини це те саме слово.
    """
    return unicodedata.normalize("NFKC", text).casefold()


def tokenize(text: str) -> list[str]:
    """Текст → терми, включно з символьними n-грамами довгих слів."""
    terms: list[str] = []
    for match in _TOKEN.finditer(normalize(text)):
        token = match.group()
        if len(token) < MIN_TOKEN_LEN and not token.isdigit():
            continue
        terms.append(token)
        if len(token) >= NGRAM_MIN_LEN:
            terms.extend(
                f"#{token[i:i + NGRAM]}" for i in range(len(token) - NGRAM + 1)
            )
    return terms


def term_id(term: str) -> int:
    """Терм → стабільний невідʼємний ідентифікатор.

    Власний хеш, а не `hash()`: той рандомізується між процесами (PYTHONHASHSEED),
    і збережений індекс перестав би збігатися із запитом після перезапуску —
    мовчки, без жодної помилки.
    """
    value = 2166136261
    for byte in term.encode("utf-8"):
        value = ((value ^ byte) * 16777619) & 0xFFFFFFFF
    # Qdrant приймає uint32; нуль лишаємо вільним як «немає терма».
    return value or 1


def build(text: str) -> SparseVec:
    """Текст → розріджений вектор частот термів.

    Зберігається саме ЧАСТОТА, без нормування: IDF-складову рахує Qdrant за
    модифікатором `idf`, а нормування довжиною тут зашкодило б — довгий
    транскрипт відео не мусить важити менше за короткий напис, якщо шуканий
    терм є в обох.
    """
    counts = Counter(tokenize(text))
    if not counts:
        return SparseVec((), ())
    items = sorted((term_id(t), float(n)) for t, n in counts.items())
    return SparseVec(tuple(i for i, _ in items), tuple(v for _, v in items))


def build_query(text: str) -> SparseVec:
    """Запит → розріджений вектор.

    Відрізняється від індексного одним: повтор терма в запиті нічого не
    додає. «номер номер номер» шукає те саме, що «номер», і множити на три
    означало б лише зіпсувати злиття.
    """
    terms = set(tokenize(text))
    if not terms:
        return SparseVec((), ())
    items = sorted((term_id(t), 1.0) for t in terms)
    return SparseVec(tuple(i for i, _ in items), tuple(v for _, v in items))


def words(text: str) -> set[str]:
    """Лише СЛОВА, без n-грам — для пояснення збігу людині.

    N-грами потрібні пошуку (стійкість до помилок OCR), але показувати
    «збіглося #aa1, #a12, #123» безглуздо: у матеріалах справи має стояти
    слово, яке справді знайдено.
    """
    return {t for t in tokenize(text) if not t.startswith("#")}


def coverage(query: str, text: str) -> tuple[float, tuple[str, ...]]:
    """Яка частка слів запиту справді є в тексті, і які саме.

    Це і є впевненість лексичного шару — величина, яку можна поставити поруч
    із візуальною, бо обидві означають те саме: наскільки система певна, що
    знайшла саме те. Для точного рядка вона дорівнює одиниці, і це чесно:
    «PXCLD-1624» у прочитаному тексті — свідчення сильніше за будь-який
    косинус.
    """
    wanted = words(query)
    if not wanted:
        return 0.0, ()
    found = tuple(sorted(wanted & words(text)))
    return len(found) / len(wanted), found
