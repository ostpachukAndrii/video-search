"""Плиткування кадру заради дрібних обʼєктів (п.12).

Задача: обʼєкт розміром 20 пікселів у кадрі 4000×3000 після стискання під
модель займає частку пікселя і зникає з ембедингу повністю. Розвʼязання —
індексувати кадр ще й фрагментами, де той самий обʼєкт займає помітну частку.

Підхід SAHI: сітка з перекриттям. Перекриття потрібне, бо обʼєкт на межі двох
плиток інакше розрізався б навпіл і не знайшовся б у жодній.

Модуль свідомо не залежить від моделей: плитки — це чиста геометрія, і вона
має тестуватися без завантаження 4 ГБ ваг.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from PIL.Image import Image

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Region:
    """Прямокутник у частках кадру плюс те, звідки він узявся.

    Координати нормалізовані (0..1), а не в пікселях: результат має лишатися
    осмисленим після масштабування кадру, а рамка — придатною для показу
    поверх оригіналу будь-якого розміру.
    """

    x: float
    y: float
    w: float
    h: float
    kind: str  # frame | tile | object
    label: str = ""
    score: float = 1.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.x <= 1.0 and 0.0 <= self.y <= 1.0):
            raise ValueError(f"рамка поза кадром: x={self.x}, y={self.y}")
        if self.w <= 0 or self.h <= 0:
            raise ValueError(f"порожня рамка: w={self.w}, h={self.h}")

    @property
    def area_ratio(self) -> float:
        return self.w * self.h

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.w, self.h)

    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Нормалізована рамка → (left, top, right, bottom) у пікселях."""
        left = int(round(self.x * width))
        top = int(round(self.y * height))
        right = min(width, int(round((self.x + self.w) * width)))
        bottom = min(height, int(round((self.y + self.h) * height)))
        return left, top, max(right, left + 1), max(bottom, top + 1)

    def crop(self, image: "Image") -> "Image":
        return image.crop(self.to_pixels(image.width, image.height))

    def containment(self, other: "Region") -> float:
        """Яка частка ЦІЄЇ рамки лежить усередині іншої.

        IoU для вкладеності не працює: рамка обличчя всередині рамки людини
        дає низький IoU, бо площі різні, — і обидві лишаються у видачі. Але
        для показу це та сама знахідка, і дрібніша рамка лише збиває з пантелику.
        """
        ax2, ay2 = self.x + self.w, self.y + self.h
        bx2, by2 = other.x + other.w, other.y + other.h
        inter_w = max(0.0, min(ax2, bx2) - max(self.x, other.x))
        inter_h = max(0.0, min(ay2, by2) - max(self.y, other.y))
        return (inter_w * inter_h) / self.area_ratio if self.area_ratio else 0.0

    def iou(self, other: "Region") -> float:
        """Перетин до обʼєднання — для відсіву майже однакових рамок."""
        ax2, ay2 = self.x + self.w, self.y + self.h
        bx2, by2 = other.x + other.w, other.y + other.h
        inter_w = max(0.0, min(ax2, bx2) - max(self.x, other.x))
        inter_h = max(0.0, min(ay2, by2) - max(self.y, other.y))
        inter = inter_w * inter_h
        union = self.area_ratio + other.area_ratio - inter
        return inter / union if union > 0 else 0.0


WHOLE_FRAME = Region(0.0, 0.0, 1.0, 1.0, kind="frame")


def tile_pixels(
    width: int,
    height: int,
    tile_size: int = 384,
    overlap: float = 0.2,
    *,
    max_tiles: int = 64,
) -> list[Region]:
    """Сітка плиток ФІКСОВАНОГО розміру в пікселях, а не в частках кадру.

    Сітка «2×2» на знімку 1280×960 дає плитки 768×576, а на 4000×3000 — уже
    2400×1800. Модель бачить приблизно 384×384 незалежно від того, що їй дали,
    тож на великому фото плитка стискається вчетверо сильніше — і дрібний
    обʼєкт зникає саме там, де його найважче помітити. Частка кадру як міра
    приховує цю різницю.

    Тут навпаки: розмір плитки сталий, а їх кількість виводиться з розміру
    знімка. Тоді ефективна роздільність кожної плитки однакова для будь-якого
    вихідного файлу.

    `tile_size` варто тримати близьким до власної роздільності моделі: більша
    плитка стискатиметься, менша — розтягуватиметься без додавання інформації.

    Межа перекриття, яку варто знати: цілком гарантовано вміщується лише
    обʼєкт, вужчий за `tile_size × overlap` — при 288 px і 0.2 це 58 px.
    Ширший може лягти рівно на стик і не вміститися в жодну плитку. Це свідома
    межа: дрібні обʼєкти, заради яких усе й робиться, у неї вкладаються, а
    більші знаходяться на цілому кадрі. Але тихою вона бути не повинна.

    Повертає ПОРОЖНІЙ перелік, якщо знімок і так уміщається в одну плитку:
    цілий кадр індексується окремо, і дублювати його плиткою — зайвий вектор.
    """
    if tile_size <= 0:
        raise ValueError("розмір плитки має бути додатним")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("перекриття має бути в межах [0, 1)")
    if max_tiles < 1:
        raise ValueError("ліміт плиток має бути щонайменше 1")

    # Знімок, менший за плитку, ділити немає сенсу.
    if width <= tile_size and height <= tile_size:
        return []

    def layout(size: int) -> tuple[int, int, int]:
        step = max(1, int(size * (1.0 - overlap)))
        cols = -(-max(0, width - size) // step) + 1
        rows = -(-max(0, height - size) // step) + 1
        return size, step, rows * cols

    size, step, count = layout(tile_size)

    # Запобіжник від вибуху на панорамах і 12-мегапіксельних знімках. Росте
    # саме ПЛИТКА, а не крок: збільшений крок при незмінній плитці лишає між
    # плитками смуги, які не потрапляють у жодну, — і обʼєкт у такій смузі
    # зникає остаточно. Грубіша плитка гірша за дрібну, але покриття повне.
    while count > max_tiles and size < max(width, height):
        size, step, count = layout(int(size * 1.25) + 1)

    if size > tile_size:
        # Не тиха деградація: збільшена плитка стискається під модель сильніше,
        # і дрібні обʼєкти на такому знімку знайдуться гірше, ніж на меншому
        # тим самим профілем. Ліміт вибрано свідомо, але наслідок має бути
        # видно в логах, а не лише у гірших метриках через тиждень.
        _warn_enlarged(width, height, tile_size, size, max_tiles)

    tiles: list[Region] = []
    tile_w, tile_h = min(size, width), min(size, height)
    for row in range(rows_of(height, tile_h, step)):
        for col in range(rows_of(width, tile_w, step)):
            left = min(col * step, max(0, width - tile_w))
            top = min(row * step, max(0, height - tile_h))
            tiles.append(
                Region(
                    left / width,
                    top / height,
                    tile_w / width,
                    tile_h / height,
                    kind="tile",
                    label=f"r{row}c{col}",
                )
            )
    return tiles


def rows_of(extent: int, tile: int, step: int) -> int:
    """Скільки плиток довжиною `tile` із кроком `step` покривають `extent`."""
    return -(-max(0, extent - tile) // step) + 1


#: Попередження друкується раз на кожне поєднання «розмір знімка → розмір
#: плитки», а не раз на файл: інакше тека з тисячею однакових знімків
#: перетворила б лог на шум і сама себе б сховала.
_ENLARGED_SEEN: set[tuple[int, int, int]] = set()


def _warn_enlarged(width: int, height: int, wanted: int, actual: int, max_tiles: int) -> None:
    key = (width, height, wanted)
    if key in _ENLARGED_SEEN:
        return
    _ENLARGED_SEEN.add(key)
    logger.warning(
        "знімок %d×%d: ліміт %d плиток збільшив плитку %d → %d px; "
        "дрібні обʼєкти на таких файлах знайдуться гірше",
        width,
        height,
        max_tiles,
        wanted,
        actual,
    )


def deduplicate(
    regions: list[Region],
    iou_threshold: float = 0.85,
    containment_threshold: float | None = None,
) -> list[Region]:
    """Прибрати повторні рамки, лишаючи ту, що з вищою оцінкою.

    Дві різні форми повтору, і кожна потребує своєї міри:

    * **накладання** — дві рамки описують те саме місце; ловиться через IoU;
    * **вкладеність** — рамка обличчя всередині рамки людини; IoU тут малий
      через різницю площ, тому потрібна частка вкладеності.

    Вкладеність відсівається лише коли `containment_threshold` заданий. При
    індексації вона корисна: кроп обличчя несе інформацію, якої немає в кропі
    цілої людини. При показі — навпаки, лише засмічує кадр рамками.
    """
    kept: list[Region] = []
    for region in sorted(regions, key=lambda r: (-r.score, -r.area_ratio)):
        if any(region.iou(existing) >= iou_threshold for existing in kept):
            continue
        if containment_threshold is not None and any(
            region.containment(existing) >= containment_threshold
            or existing.containment(region) >= containment_threshold
            for existing in kept
        ):
            continue
        kept.append(region)
    return kept


def iter_crops(image: "Image", regions: list[Region]) -> Iterator[tuple[Region, "Image"]]:
    """Ліниво нарізати кадр за рамками."""
    for region in regions:
        yield region, region.crop(image)
