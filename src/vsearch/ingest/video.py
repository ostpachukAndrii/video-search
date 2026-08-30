"""Відбір ключових кадрів із відео (M5a).

Відео не індексується покадрово, і це не оптимізація, а вимога коректності:
секунда матеріалу при 30 к/с дає тридцять майже однакових векторів, які
займуть усю верхівку видачі й витіснять інші кадри. Питання не «скільки
кадрів узяти», а «які кадри РІЗНІ».

**Гібридний відбір.** Кадр береться, коли сталася зміна сцени АБО минув
інтервал часу. Обидві умови потрібні:

* сама лише зміна сцени пропустила б усе, що знято одним дублем, — а це
  типовий випадок для матеріалів справи: камера спостереження й нагрудна
  камера пишуть годинами без жодного монтажного стику;
* сам лише інтервал пропустив би коротку подію між двома відліками.

**Один прохід декодування.** Межі сцен рахуються з ТІЄЇ САМОЇ перцептивної
різниці, що й дедуплікація, тож окремий прохід PySceneDetect не потрібен.
Декодування відео — найдорожча частина, і робити його двічі заради того, що
вже пораховано, було б платою без виграшу.

**Перцептивний хеш замість бібліотеки.** dHash 8×8 — це десяток рядків на
numpy, і він уже потрібен для дедуплікації. Тягнути заради нього окрему
залежність у ланцюг постачання, де кожна вага закріплена sha256, непослідовно.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import numpy as np

if TYPE_CHECKING:
    from PIL.Image import Image

logger = logging.getLogger(__name__)

#: Розширення, які вважаємо відео.
VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"})

#: Сторона зменшеного кадру для хешування. 9×8 дає 64 біти різниці по рядку.
_HASH_W, _HASH_H = 9, 8

#: Скільки бітів dHash мають різнитися, щоб вважати кадр НОВОЮ сценою.
#:
#: 64-бітний хеш; поріг ~18 бітів (28%) відповідає помітній зміні вмісту, а не
#: тремтінню камери чи зміні освітлення. Число не підганялося під конкретний
#: файл — воно виведене з того, що dHash сусідніх кадрів одного плану звичайно
#: різниться на одиниці бітів, а зміна плану дає десятки.
SCENE_DISTANCE = 18

#: Нижче цієї відстані кадр вважається ДУБЛІКАТОМ уже взятого й не береться,
#: навіть якщо інтервал минув. Саме це не дає годині статичного запису
#: перетворитися на тисячі однакових векторів.
DUPLICATE_DISTANCE = 6


@dataclass(frozen=True)
class Keyframe:
    """Один відібраний кадр із його місцем у часі."""

    image: "Image"
    ts_ms: int
    #: Порядковий номер сцени в цьому файлі. Кадри однієї сцени згортаються
    #: разом при показі, щоб одна подія не займала всю видачу.
    shot: int
    #: Чому кадр узято — для провенансу: слідчий має бачити, що система
    #: відібрала за зміною сцени, а що за таймером.
    reason: str
    #: Перцептивний хеш, за яким кадр і був відібраний.
    #:
    #: Він ВІДДАЄТЬСЯ, а не перераховується споживачем. Перша версія цього
    #: модуля мала дві реалізації dHash — через ffmpeg (швидку) і через PIL — і
    #: вони давали різні результати, бо масштабують різними фільтрами. Тест
    #: дедуплікації тоді перевіряв не той хеш, яким вона робиться. Одна логіка
    #: в двох місцях — помилка, за яку проєкт уже платив.
    digest: "np.ndarray"


def _dhash_frame(frame) -> np.ndarray:
    """64-бітний dHash прямо з декодованого кадру.

    Стійкий до зміни яскравості й масштабу, чутливий до зміни вмісту — саме
    те, що потрібно і для меж сцен, і для дедуплікації.

    `reformat` масштабує й переводить у відтінки сірого силами ffmpeg, тобто
    без розгортання повного RGB у Python. Саме це відрізняє обхід, що встигає
    за реальним часом, від того, що втричі відстає.
    """
    plane = frame.reformat(width=_HASH_W, height=_HASH_H, format="gray").to_ndarray()
    pixels = plane.astype(np.int16)
    return (pixels[:, 1:] > pixels[:, :-1]).flatten()


def _distance(a: np.ndarray | None, b: np.ndarray) -> int:
    return 64 if a is None else int(np.count_nonzero(a != b))


def sample_keyframes(
    path: Path | str,
    *,
    fps_floor: float = 0.5,
    max_frames: int | None = None,
    scene_distance: int = SCENE_DISTANCE,
    duplicate_distance: int = DUPLICATE_DISTANCE,
) -> Iterator[Keyframe]:
    """Ключові кадри відео: зміна сцени АБО інтервал, без дублікатів.

    `fps_floor` — скільки кадрів на секунду брати щонайменше. 0.5 означає
    «не рідше ніж раз на дві секунди», навіть якщо картинка не змінюється.

    Декодування йде потоком: увесь файл у памʼять не читається ніколи, тож
    тригодинний запис коштує стільки ж памʼяті, скільки хвилинний.
    """
    import av

    path = Path(path)
    interval_ms = int(1000 / fps_floor) if fps_floor > 0 else 0
    taken = 0
    shot = 0
    previous: np.ndarray | None = None
    last_taken_hash: np.ndarray | None = None
    last_taken_ms = -(10**9)

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        # Декодуємо лише відео: аудіо на цьому етапі не потрібне, а його
        # декодування коштувало б часу на кожному файлі (ASR — окремий крок).
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            ts_ms = int(frame.pts * float(stream.time_base) * 1000)
            # Хеш береться з ЗМЕНШЕНОЇ сірої площини, а не з повного кадру.
            # Різниця не косметична: `frame.to_image()` розгортає кожен кадр у
            # повнорозмірний RGB через PIL, і на 30 к/с це і є вся вартість
            # обходу. Перетворення в зображення робиться лише для кадрів, які
            # справді беремо, — а їх на два порядки менше.
            digest = _dhash_frame(frame)

            changed = _distance(previous, digest) >= scene_distance
            due = ts_ms - last_taken_ms >= interval_ms if interval_ms else False
            previous = digest
            if not (changed or due):
                continue

            # Дедуплікація перевіряється ОСТАННЬОЮ і стосується вже взятих
            # кадрів, а не сусідніх: інакше повільний наїзд камери давав би
            # ланцюжок «кожен схожий на попередній», і жоден кадр не взявся б,
            # хоча початок і кінець зовсім різні.
            if _distance(last_taken_hash, digest) < duplicate_distance:
                continue

            if changed:
                shot += 1
            yield Keyframe(
                image=frame.to_image(), ts_ms=ts_ms, shot=shot,
                reason="scene" if changed else "interval", digest=digest,
            )
            last_taken_hash = digest
            last_taken_ms = ts_ms
            taken += 1
            if max_frames is not None and taken >= max_frames:
                logger.warning(
                    "%s: досягнуто межі %d кадрів, решта відео не проіндексована",
                    path.name, max_frames,
                )
                return


def probe(path: Path | str) -> dict:
    """Тривалість, частота й розмір — без декодування вмісту."""
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        duration_s = float(container.duration / 1_000_000) if container.duration else 0.0
        return {
            "duration_s": duration_s,
            "fps": float(stream.average_rate) if stream.average_rate else 0.0,
            "width": stream.width,
            "height": stream.height,
            "frames": stream.frames or 0,
        }
