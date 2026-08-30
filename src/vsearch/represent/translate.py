"""Переклад запиту в англійську спеціалізованою моделлю.

Навіщо окрема модель, коли є LLM-парсер. Парсер робив дві різні роботи —
перекладав і розбирав структуру, — і обидві страждали.

**Переклад.** Qwen3-4B не знає побутової української лексики. Виміряно на 16
іменниках: 8 правильних. «Полуниця» перетворювалася на `a bench`, `a mushroom`,
`a flower`, а на прямий запит без нашого промпту модель відповіла `bluegrass`.
Наслідки бачив користувач: пошук полуниці показував квітку в кутку кадру,
«літак» знаходив фото дівчини з упевненістю 26%.

OPUS-MT на тих самих 16 словах дає **16 із 16** при 76 млн параметрів проти
4 млрд. Спеціалізована модель у 50 разів менша й удвічі точніша, бо робить
одну річ.

**Структура.** Промпт парсера досяг межі ємності: за одну сесію правки ламали
сусідні випадки шість разів. Знявши з нього переклад, ми звільняємо половину
зразків — це не оптимізація, а спосіб припинити обмін одного випадку на інший.

Перевірені й відхилені альтернативи, обидві на тому самому наборі:

* **M2M-100 418M** (MIT, 100 мов) — **1 із 16**. «Полуниця» → `the fool`,
  «білка» → `protein`, «сходи» → `go down`.
* **opus-mt-mul-en** (Apache-2.0, багатомовна) — **0 із 10**.

Обидві тренувалися на реченнях і на окремих словах розсипаються, а запит до
пошуку зображень — це майже завжди кілька слів. Тому ціна рішення прийнята
свідомо: **модель на кожну пару мов**, по ~300 МБ.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

#: Пара мов → назва в маніфесті. Розширюється додаванням запису в
#: `models/manifest.lock`, а не зміною коду.
MODELS = {"uk": "translate_uk_en"}

#: Скільки токенів дозволяємо на переклад. Запит до пошуку зображень — це
#: кілька слів; довший переклад означає, що модель пішла вигадувати.
MAX_TOKENS = 48

_loaded: dict[str, tuple] = {}

_CYRILLIC = re.compile(r"[Ѐ-ӿ]")
_HEBREW = re.compile(r"[֐-׿]")
#: Літери, яких немає в російській — найдешевша ознака української.
_UKRAINIAN_ONLY = re.compile(r"[іїєґІЇЄҐ]")


def detect_language(text: str) -> str:
    """Мова запиту за писемністю, без окремої моделі.

    Навмисно груба: розрізняти треба лише те, для чого в нас Є перекладач.
    Повноцінний визначник мови був би ще однією вагою в образі заради
    відповіді, яку дає один регулярний вираз.
    """
    if _HEBREW.search(text):
        return "he"
    if _CYRILLIC.search(text):
        # Українська й російська ділять абетку; для нас обидві йдуть у ту
        # саму модель, тож розрізняти їх не потрібно — але позначаємо чесно.
        return "uk"
    return "en"


def is_available(language: str = "uk") -> bool:
    """Чи є ваги перекладача на диску. Мережі не торкається."""
    from vsearch.backends.registry import ModelNotFetched, get_registry

    name = MODELS.get(language)
    if not name:
        return False
    try:
        return bool(get_registry().local_path(name))
    except (ModelNotFetched, KeyError):
        return False


def _model(language: str):
    if language in _loaded:
        return _loaded[language]
    from transformers import MarianMTModel, MarianTokenizer

    from vsearch.backends.registry import get_registry

    # ТІЛЬКИ локальний шлях і `local_files_only`: єдина точка завантаження
    # моделей у проєкті приймає шлях, а не repo_id, — інакше код тихо працює
    # на ноутбуці й падає в середовищі без мережі.
    path = str(get_registry().local_path(MODELS[language]))
    tokenizer = MarianTokenizer.from_pretrained(path, local_files_only=True)
    model = MarianMTModel.from_pretrained(path, local_files_only=True).eval()
    _loaded[language] = (tokenizer, model)
    return _loaded[language]


def translate(text: str, language: str | None = None) -> str:
    """Запит англійською. Повертає ОРИГІНАЛ, якщо перекласти нічим.

    Мовчазна відмова тут була б найгіршим варіантом: пошук працював би далі,
    просто гірше, і зрозуміти чому було б неможливо. Тому за відсутності ваг
    повертається вхідний текст, а причина йде в лог.
    """
    text = (text or "").strip()
    if not text:
        return text
    language = language or detect_language(text)
    if language == "en":
        return text
    if not is_available(language):
        logger.info("немає перекладача для %r — запит іде як є", language)
        return text

    try:
        tokenizer, model = _model(language)
        batch = tokenizer([text], return_tensors="pt", padding=True)
        output = model.generate(**batch, max_new_tokens=MAX_TOKENS, num_beams=4)
        result = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    except Exception:  # noqa: BLE001 — переклад не мусить валити пошук
        logger.warning("переклад не вдався, запит іде як є", exc_info=True)
        return text
    return result.rstrip(".") or text
