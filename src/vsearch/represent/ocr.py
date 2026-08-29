"""Читання тексту з кадру.

Рішення, що переглядає початковий план: **основний рушій — Florence-2**, який
уже є в образі. План передбачав EasyOCR плюс Tesseract, але обидва додали б
власні ваги, власні ліцензії та власні пастки з дозавантаженням у рантаймі, —
а Florence-2 уміє `<OCR>` тими самими вагами, що вже закріплені в маніфесті.

Виміряно на реальних знімках (0.4–0.9 с на кадр, 5.3 с на щільному
інтерфейсному знімку):

    'Münden / Augustiner-Brau Mündchen'      ← умляути читаються
    '18:53 / 5G / Share / Edit'              ← інтерфейс
    '-'                                      ← кадр без тексту, чесна відмова

Межа рішення: Florence-2 тренований переважно на латиниці. Для івриту (вимога
п.5) лишається окремий рушій — Tesseract `heb`, — і він свідомо винесений у
фіче-флаг: бінарник неможливо покласти через pip, тож його місце в Dockerfile,
а не в залежностях Python. Поки такого рушія немає, система має казати про це
вголос, а не тихо повертати порожній рядок.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from PIL.Image import Image

logger = logging.getLogger(__name__)

TASK_OCR = "<OCR>"

#: Завдання, що описує кадр словами. Не OCR, але живе поруч навмисно: обидва
#: годують ОДИН лексичний шар і обидва йдуть тими самими вагами.
#:
#: Підпис дає те, чого не дає ані ембединг, ані OCR: він читає етикетку й
#: називає її словами. На полиці, що впевнено спрацьовувала на «шампанське»,
#: підпис каже «a shelf filled with bottles of Havana Club Cuban Spiced Rum»
#: — тобто називає ром. Візуальна схожість цього не розрізняє в принципі:
#: пляшка й пляшка (ADR-020).
TASK_CAPTION = "<DETAILED_CAPTION>"

#: Що Florence-2 повертає, коли тексту немає. Це НЕ текст, і потрапивши в
#: індекс, воно стало б термом, який збігається з усім підряд.
EMPTY_MARKERS = {"", "-", "—", "–", "n/a"}

#: Стеля довжини. Щільний знімок інтерфейсу дає сотні термів, і всі вони
#: потрапляють в один розріджений вектор; без межі один такий кадр почав би
#: збігатися майже з будь-яким текстовим запитом.
MAX_TEXT_LEN = 2000


@dataclass(frozen=True)
class OcrResult:
    """Текст кадру разом із тим, чим його прочитано (п.7)."""

    text: str
    engine: str
    #: Писемності, які цей рушій НЕ покриває. Порожній текст на івриті й
    #: справжня відсутність тексту — різні речі, і плутати їх не можна.
    unsupported: tuple[str, ...] = ()

    @property
    def has_text(self) -> bool:
        return bool(self.text)


class OcrEngine(Protocol):
    def read(self, image: "Image") -> OcrResult: ...


class Florence2Ocr:
    """OCR і опис кадру тими самими вагами, що й пропозиції регіонів."""

    #: Писемності поза латиницею, на яких модель ненадійна.
    UNSUPPORTED = ("hebrew", "arabic")

    def __init__(self, profile, proposer=None) -> None:
        self.profile = profile
        self._proposer = proposer

    def _ensure(self):
        if self._proposer is None:
            from vsearch.represent.regions import Florence2Proposer

            self._proposer = Florence2Proposer(self.profile, task=TASK_OCR)
        self._proposer._ensure_loaded()
        return self._proposer

    def describe(self, image: "Image") -> str:
        """Опис кадру словами — для лексичного шару, не для показу.

        Виміряно на золотому наборі: підпис допомагає у 5 запитах із 14 і не
        шкодить у жодному. Виграш помірний, зате без ризику вигадування — на
        відміну від категорій, які довелося відкотити, підпис описує те, що
        в кадрі Є.

        Ціна на масштабі істотна: ~1.2 с на кадр, тобто на мільйоні знімків
        це два тижні однопотоково. Тому за прапорцем профілю, як і OCR.
        """
        return clean(self._generate(image, TASK_CAPTION, max_new_tokens=128))

    def _generate(self, image: "Image", task: str, *, max_new_tokens: int) -> str:
        """Один прохід Florence-2 для будь-якого текстового завдання."""
        import torch

        from vsearch.backends import device as device_mod

        proposer = self._ensure()
        torch_device, torch_dtype = device_mod.torch_device_and_dtype(proposer.device_spec)
        inputs = proposer._processor(
            text=task, images=image, return_tensors="pt"
        ).to(torch_device, torch_dtype)
        with torch.no_grad():
            generated = proposer._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=max_new_tokens,
                num_beams=3,
                do_sample=False,
            )
        decoded = proposer._processor.batch_decode(generated, skip_special_tokens=False)[0]
        parsed = proposer._processor.post_process_generation(
            decoded, task=task, image_size=(image.width, image.height)
        )
        return str(parsed.get(task, ""))

    def read(self, image: "Image") -> OcrResult:
        return OcrResult(
            text=clean(self._generate(image, TASK_OCR, max_new_tokens=256)),
            engine="florence2",
            unsupported=self.UNSUPPORTED,
        )


def clean(text: str) -> str:
    """Прибрати заглушки й обрізати до розумної довжини."""
    stripped = " ".join(text.split())
    if stripped.strip().lower() in EMPTY_MARKERS:
        return ""
    # Перелік заглушок неповний за побудовою: модель віддавала і «-», і «.».
    # Надійніша ознака — відсутність жодної літери чи цифри: такий рядок не
    # дасть жодного терма, зате роздує payload і виглядатиме як прочитаний
    # текст у матеріалах справи.
    if not any(ch.isalnum() for ch in stripped):
        return ""
    return stripped[:MAX_TEXT_LEN]
