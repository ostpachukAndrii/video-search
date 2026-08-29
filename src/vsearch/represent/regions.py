"""Пропозиції регіонів: що саме нарізати з кадру.

Дві стратегії з різною вартістю й різними сильними сторонами:

* **Плитки** — чиста геометрія, нуль інференсу, покриття гарантоване. Не знає,
  що на зображенні, зате нічого не пропускає.
* **Пропозиції детектора** (Florence-2) — влучні рамки навколо справжніх
  обʼєктів, але коштують інференсу і, як показала перевірка на M2, дрібні
  обʼєкти на повному кадрі пропускають.

Тому вони не альтернативи, а шари: плитки дають покриття, детектор — точність.
За SAHI детектор варто запускати саме по плитках, а не по цілому кадру — тоді
дрібний обʼєкт для нього стає великим.
"""

from __future__ import annotations

from typing import Any

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from vsearch.config import Profile
from vsearch.represent.tiling import Region, deduplicate, tile_pixels

if TYPE_CHECKING:
    from PIL.Image import Image

logger = logging.getLogger(__name__)

#: Завдання Florence-2, що повертає рамки без прив'язки до заданого словника.
TASK_REGION_PROPOSAL = "<REGION_PROPOSAL>"
TASK_OBJECT_DETECTION = "<OD>"
TASK_DENSE_CAPTION = "<DENSE_REGION_CAPTION>"

#: Завдання, що приймає ТЕКСТ і шукає названим словом (ADR-012).
#:
#: Принципова відмінність від решти: воно єдине не має зміщення в бік
#: помітного. Загальні пропозиції відповідають на питання «що тут головне»,
#: і дрібна сумка на фоні в цю відповідь не потрапляє. Обумовлена детекція
#: відповідає на питання «де тут сумка», і фон їй не заважає.
#:
#: Наслідок для архітектури: застосовна лише на етапі ЗАПИТУ. При індексації
#: слова ще немає, і довелося б перелічувати словник — тобто повернутися до
#: проблеми, яку розвʼязує ADR-009.
TASK_OPEN_VOCAB = "<OPEN_VOCABULARY_DETECTION>"

#: Рамки, менші за це, майже завжди шум детектора: кроп такого розміру не несе
#: пізнаваного змісту навіть для людини.
MIN_OBJECT_AREA = 0.0002

#: Рамки, більші за це, вироджені: детектор пропонує «обʼєкт» завбільшки з
#: цілий кадр, який уже проіндексований у frames. Такий регіон не додає
#: інформації, зате додає ще один вектор, здатний випадково виграти ранжування.
MAX_OBJECT_AREA = 0.8


class RegionProposer(Protocol):
    """Спільний інтерфейс: кадр → перелік рамок."""

    def propose(self, image: "Image") -> list[Region]: ...


class TileProposer:
    """Сітка плиток із перекриттям. Без моделей і без інференсу."""

    def __init__(self, profile: Profile) -> None:
        self.profile = profile

    def propose(self, image: "Image") -> list[Region]:
        tiling = self.profile.tiling
        regions = [Region(0.0, 0.0, 1.0, 1.0, kind="frame")]
        if tiling.tile_size is not None:
            regions.extend(
                tile_pixels(
                    image.width,
                    image.height,
                    tile_size=tiling.tile_size,
                    overlap=tiling.overlap,
                    max_tiles=tiling.max_tiles,
                )
            )
        return regions


#: Ваги Florence-2, спільні для всіх екземплярів. Тут це важить навіть
#: більше, ніж для ембедера: обумовлена детекція створює НОВИЙ екземпляр на
#: кожну сутність запиту й на кожен профіль, тож без кешу модель на 0.8 ГБ
#: перезавантажувалася по кілька разів на один пошук.
_LOADED: dict[tuple[str, str, str], tuple[Any, Any]] = {}


class Florence2Proposer:
    """Пропозиції рамок від Florence-2.

    Використовує НАТИВНУ реалізацію transformers, без `trust_remote_code`:
    оригінальний чекпоінт microsoft/* йде через віддалений код, зламаний на
    transformers 5.x, а конвертований florence-community/* працює напряму.
    Побічний виграш — в образ не треба вендорити код моделі.
    """

    def __init__(
        self,
        profile: Profile,
        registry=None,
        device_spec=None,
        task: str = TASK_REGION_PROPOSAL,
        model_name: str = "florence2_large",
        term: str = "",
    ) -> None:
        from vsearch.backends import device as device_mod
        from vsearch.backends.registry import get_registry

        self.profile = profile
        self.registry = registry or get_registry()
        self.device_spec = device_spec or device_mod.resolve()
        self.task = task
        self.model_name = model_name
        #: Слово або фраза, якою обумовлена детекція. Порожнє для решти завдань.
        self.term = term
        self._model = None
        self._processor = None

    @property
    def prompt(self) -> str:
        """Текст, що подається моделі: завдання плюс слово запиту, якщо є."""
        return f"{self.task}{self.term}" if self.term else self.task

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoProcessor, Florence2ForConditionalGeneration

        from vsearch.backends import device as device_mod

        path = str(self.registry.local_path(self.model_name))
        torch_device, torch_dtype = device_mod.torch_device_and_dtype(self.device_spec)
        key = (path, str(torch_device), str(torch_dtype))
        cached = _LOADED.get(key)
        if cached is not None:
            self._processor, self._model = cached
            return

        logger.info("завантаження %s на %s", self.model_name, torch_device)
        self._processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        self._model = (
            Florence2ForConditionalGeneration.from_pretrained(
                path, local_files_only=True, dtype=torch_dtype
            )
            .to(torch_device)
            .eval()
        )
        _LOADED[key] = (self._processor, self._model)

    def propose(self, image: "Image") -> list[Region]:
        self._ensure_loaded()
        import torch

        from vsearch.backends import device as device_mod

        torch_device, torch_dtype = device_mod.torch_device_and_dtype(self.device_spec)
        inputs = self._processor(  # type: ignore[misc]
            text=self.prompt, images=image, return_tensors="pt"
        ).to(torch_device, torch_dtype)
        with torch.no_grad():
            generated = self._model.generate(  # type: ignore[union-attr]
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=512,
                num_beams=3,
                do_sample=False,
            )
        decoded = self._processor.batch_decode(generated, skip_special_tokens=False)[0]  # type: ignore[misc]
        parsed = self._processor.post_process_generation(  # type: ignore[misc]
            decoded, task=self.task, image_size=(image.width, image.height)
        )
        regions = self._to_regions(parsed.get(self.task, {}), image.width, image.height)
        if self.term:
            # Обумовленій детекції верхня межа площі не потрібна: коли спитали
            # «людина», рамка на пів кадру — правильна відповідь, а не
            # вироджена пропозиція. Для загальних пропозицій навпаки: рамка на
            # весь кадр не додає інформації до вже проіндексованого кадру.
            return regions
        return [r for r in regions if r.area_ratio <= MAX_OBJECT_AREA]

    @staticmethod
    def _to_regions(payload: dict, width: int, height: int) -> list[Region]:
        """Пікселі від Florence-2 → нормалізовані рамки.

        Обумовлена детекція повертає підписи під ключем `bboxes_labels`, решта
        завдань — під `labels`. Без цього рамки приходили б безіменними, і в
        показі зникло б головне пояснення: ЩО саме детектор тут знайшов.
        """
        boxes = payload.get("bboxes", []) if isinstance(payload, dict) else []
        labels = (
            payload.get("labels")
            or payload.get("bboxes_labels")
            or []
        ) if isinstance(payload, dict) else []
        regions: list[Region] = []
        for index, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            x, y = max(0.0, x1 / width), max(0.0, y1 / height)
            w, h = min(1.0 - x, (x2 - x1) / width), min(1.0 - y, (y2 - y1) / height)
            area = w * h
            if w <= 0 or h <= 0 or area < MIN_OBJECT_AREA:
                continue
            label = labels[index] if index < len(labels) else ""
            regions.append(Region(x, y, w, h, kind="object", label=str(label)))
        return regions


@dataclass
class CompositeProposer:
    """Плитки плюс, за потреби, пропозиції детектора — з відсівом дублікатів.

    Дублікати відсіваються не з естетичних міркувань: кожна зайва рамка це
    зайвий вектор, а на мільйонах активів саме кількість векторів вирішує,
    чи поміститься індекс у памʼять.
    """

    tiles: TileProposer
    detector: RegionProposer | None = None
    iou_threshold: float = 0.85
    #: Детектор по плитках, а не по кадру: підхід SAHI, завдяки якому дрібний
    #: обʼєкт стає для детектора великим.
    detect_on_tiles: bool = False

    def propose(self, image: "Image") -> list[Region]:
        regions = self.tiles.propose(image)
        if self.detector is None:
            return regions

        found: list[Region] = []
        if self.detect_on_tiles:
            for tile in (r for r in regions if r.kind == "tile"):
                for local in self.detector.propose(tile.crop(image)):
                    found.append(_map_into(local, tile))
        else:
            found.extend(self.detector.propose(image))

        # Плитки лишаються завжди: вони гарантують покриття, а детектор — ні.
        return regions + deduplicate(found, self.iou_threshold)


def _map_into(inner: Region, outer: Region) -> Region:
    """Рамка в координатах плитки → координати цілого кадру."""
    return Region(
        x=outer.x + inner.x * outer.w,
        y=outer.y + inner.y * outer.h,
        w=inner.w * outer.w,
        h=inner.h * outer.h,
        kind=inner.kind,
        label=inner.label,
        score=inner.score,
    )


def build_proposer(profile: Profile, *, detector: RegionProposer | None = None) -> CompositeProposer:
    """Зібрати пропонувальник згідно з профілем."""
    if profile.use_region_proposals and detector is None:
        detector = Florence2Proposer(profile)
    return CompositeProposer(
        tiles=TileProposer(profile),
        detector=detector if profile.use_region_proposals else None,
        detect_on_tiles=profile.name == "quality",
    )
