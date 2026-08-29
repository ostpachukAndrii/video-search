"""Профілі якість/швидкість.

Вимога п.10 — «швидко, але без втрати якості» — не має єдиної правильної точки,
тому вона параметризується.

Ключове рішення: усі профілі працюють на ОДНИХ вагах SigLIP 2 NaFlex і
різняться параметром `max_num_patches`. NaFlex приймає змінну довжину
послідовності, тож роздільність стає безперервним регулятором на тій самій
моделі — замість трьох різних чекпоінтів, які довелося б завантажувати,
тримати в памʼяті й індексувати в несумісні векторні простори.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

ProfileName = Literal["fast", "balanced", "quality"]

#: Розмірність вектора кожної моделі — потрібна для схеми Qdrant.
#: Змішувати ембедери в одній колекції не можна: простори не сумісні.
EMBED_DIMS: dict[str, int] = {
    "siglip2_so400m_naflex": 1152,
    "siglip2_base_naflex": 768,
}

#: SigLIP 2 тренували на тексті в нижньому регістрі, і зневага до цього тихо
#: погіршує якість, нічого не ламаючи явно.
TEXT_PROMPT_TEMPLATE = "this is a photo of {label}."


@dataclass(frozen=True)
class TilingConfig:
    """Плиткування кадру заради дрібних обʼєктів (п.12).

    Розмір задається В ПІКСЕЛЯХ, а не часткою кадру (ADR-010). Сітка 2×2 на
    знімку 1 Мпкс дає плитки 640 px, а на 12 Мпкс — 2000 px, і на другому
    дрібний обʼєкт зникає при стисканні під модель, хоча частка кадру та сама.
    Стала частка означає НЕсталу роздільність, і це виправлення саме її.
    """

    #: None вимикає плиткування: індексується лише цілий кадр.
    tile_size: int | None
    overlap: float = 0.2
    #: Стеля на знімок. Ліміт не ріже покриття — при його досягненні
    #: `tile_pixels` збільшує саму плитку, а не крок.
    max_tiles: int = 64

    @property
    def enabled(self) -> bool:
        return self.tile_size is not None


@dataclass(frozen=True)
class Profile:
    name: ProfileName
    embed_model: str
    #: Довжина послідовності NaFlex. Головний регулятор «якість ↔ швидкість»:
    #: обчислення ростуть приблизно лінійно, дрібні обʼєкти виграють найбільше.
    max_num_patches: int
    #: Скільки кандидатів тягнемо з ANN до rescore оригінальними векторами.
    ann_oversampling: int
    tiling: TilingConfig
    #: Region proposals від Florence-2 — найдорожчий крок індексації.
    use_region_proposals: bool
    #: VLM-верифікація заперечень на топі видачі (крок [5] архітектури).
    rerank_top_k: int
    #: Кадрів на секунду відео поза межами зміни сцени.
    video_fps_floor: float
    caption_regions: bool = False
    #: Читати текст із кадру (OCR) і будувати лексичний шар.
    #:
    #: Коштує 0.4–0.9 с на звичайний кадр і до 5 с на щільному знімку
    #: інтерфейсу — тобто подвоює час індексації. Але це єдиний шар, що
    #: взагалі здатен знайти номер авто чи прізвище на документі: щільний
    #: ембединг розрізняє «AA1234BB» і «AA1234BC» гірше за пошук підрядка.
    use_ocr: bool = False
    #: Описувати кадр словами й додавати опис у лексичний шар.
    #:
    #: Коштує ще ~1.2 с на кадр поверх OCR. Дає те, чого не дають ані
    #: ембединг, ані OCR: підпис читає етикетку й називає її. На полиці, що
    #: спрацьовувала на «шампанське», він каже «Havana Club Cuban Spiced Rum»
    #: — а візуальна схожість пляшку від пляшки не відрізняє (ADR-020).
    #:
    #: ⚠️ ТИПОВО ВИМКНЕНО за результатом виміру, а не з обережності.
    #:
    #: Попередня проба обіцяла виграш у 5 запитах із 14. Вона міряла частку
    #: слів запиту в підписі — і це виявилося хибною мірою: збіги припадали на
    #: службові слова. Коли підпис підключили по-справжньому, він почав давати
    #: впевненість нарівні з написом, і Recall@5 упав 0.80 → 0.73.
    #:
    #: Після виправлення (впевненість рахується лише за написом) підпис став
    #: НЕЙТРАЛЬНИМ: 2.4 / 0.80 / 8 пасток — рівно як без нього. При цьому він
    #: подвоює час індексації: 0.2 → 0.1 зображення за секунду.
    #:
    #: І він не розвʼязав задачі, заради якої вводився: полиця з ромом досі
    #: перша на «шампанське» з 52.5%. Підпис її НАЗИВАЄ ромом, але лексичний
    #: шар уміє лише піднімати за збігом, а не понижувати за суперечністю.
    #:
    #: Тобто механізм готовий і перевірений, але вмикати його поки нема за що.
    use_captions: bool = False
    extras: dict[str, object] = field(default_factory=dict)

    @property
    def embed_dim(self) -> int:
        return EMBED_DIMS[self.embed_model]

    @property
    def uses_rerank(self) -> bool:
        return self.rerank_top_k > 0

    def index_signature(self) -> dict[str, object]:
        """Параметри, які роблять індекс несумісним при зміні.

        Профіль впливає і на індексацію, і на пошук, але по-різному.
        `ann_oversampling` чи `rerank_top_k` можна змінити між запитами вільно.
        А ембедер, роздільність і сітку плиток змінити не можна: вектори,
        побудовані з іншими значеннями, лежать в іншому просторі. Тому підпис
        зберігається разом із колекцією й звіряється при відкритті.
        """
        return {
            "embed_model": self.embed_model,
            "embed_dim": self.embed_dim,
            "max_num_patches": self.max_num_patches,
            "tile_size": self.tiling.tile_size,
            "tiling_overlap": self.tiling.overlap,
            "tiling_max_tiles": self.tiling.max_tiles,
            "use_region_proposals": self.use_region_proposals,
            # Змінює склад колекції (зʼявляється розріджений вектор), тож
            # індекси з OCR і без нього несумісні.
            "use_ocr": self.use_ocr,
            "use_captions": self.use_captions,
        }


_PRIMARY = "siglip2_so400m_naflex"

PROFILES: dict[ProfileName, Profile] = {
    # Демо й попередній прохід по великих обсягах: кадр цілком, мінімальна
    # роздільність, без регіонів і без reranking.
    "fast": Profile(
        name="fast",
        embed_model=_PRIMARY,
        max_num_patches=256,
        ann_oversampling=2,
        # Плитка велика й лічильник тісний: це попередній прохід, який має
        # відсіяти явно нерелевантне, а не знайти дрібне.
        tiling=TilingConfig(tile_size=384, overlap=0.2, max_tiles=6),
        use_region_proposals=False,
        rerank_top_k=0,
        video_fps_floor=0.2,
    ),
    # Робоча точка за замовчуванням.
    #
    # ПЕРЕВІРЕНО НА РЕАЛЬНИХ ФОТО. Пропозиції Florence-2 на синтетиці не
    # окупалися (MRR 0.646 → 0.521), а на 117 реальних знімках дали 46% усіх
    # знахідок. Висновок синтетики був не просто неточним, а протилежним.
    #
    # Плитка 288 px — не кругле число, а виміряний обрив: на 288 фото з
    # сумками дає 54.9%, на 352 — 2.1%. Це нижче власної роздільності моделі
    # при 576 патчах (~384 px), бо працюють два різні механізми водночас:
    # менша плитка не лише зберігає пікселі, а й піднімає частку обʼєкта.
    "balanced": Profile(
        name="balanced",
        embed_model=_PRIMARY,
        max_num_patches=576,
        ann_oversampling=4,
        tiling=TilingConfig(tile_size=288, overlap=0.2, max_tiles=64),
        use_region_proposals=True,
        rerank_top_k=30,
        video_fps_floor=0.5,
        use_ocr=True,
    ),
    # Максимальна повнота: гранична роздільність NaFlex, пропозиції детектора
    # по плитках, підписи регіонів, глибший rerank.
    #
    # `max_tiles` тут навмисно високий: інакше стеля зрівняла б quality з
    # balanced на великих знімках, бо обидва впиралися б у неї і отримували б
    # ОДНАКОВУ збільшену плитку. Дрібніша плитка має сенс лише тоді, коли їй
    # дозволено бути дрібною до кінця.
    "quality": Profile(
        name="quality",
        embed_model=_PRIMARY,
        max_num_patches=1024,
        ann_oversampling=8,
        tiling=TilingConfig(tile_size=224, overlap=0.2, max_tiles=256),
        use_region_proposals=True,
        rerank_top_k=50,
        video_fps_floor=1.0,
        caption_regions=True,
        use_ocr=True,
    ),
}

DEFAULT_PROFILE: ProfileName = "balanced"


def get_profile(name: str | None = None) -> Profile:
    """Профіль за іменем → env VSEARCH_PROFILE → balanced."""
    chosen = name or os.environ.get("VSEARCH_PROFILE") or DEFAULT_PROFILE
    try:
        return PROFILES[chosen]  # type: ignore[index]
    except KeyError:
        raise ValueError(f"Невідомий профіль {chosen!r}; доступні: {', '.join(PROFILES)}") from None
