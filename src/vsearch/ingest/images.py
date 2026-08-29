"""Індексація зображень.

Активи ідентифікуються **sha256 вмісту, а не шляхом**. У розслідуванні той
самий файл приходить із різних джерел під різними іменами; хеш дає природну
дедуплікацію і водночас є тим, чим результат привʼязується до джерела в
матеріалах справи.
"""

from __future__ import annotations

import hashlib
import os
import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from vsearch.config import Profile, get_profile
from vsearch.index import schema
from vsearch.index.catalog import AssetRecord, Catalog
from vsearch.index.store import VectorStore
from vsearch.represent import lexical
from vsearch.represent.embed import Siglip2Embedder
from vsearch.represent.regions import CompositeProposer, build_proposer

if TYPE_CHECKING:
    from PIL.Image import Image

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}
)

#: Розширення, які ми ЗНАХОДИМО, але прочитати не можемо (ADR-028).
#:
#: HEIC лишається в обході навмисно. Прибрати його зі списку означало б, що
#: знімок з айфона просто зникає — жодного рядка, жодного лічильника, і
#: слідчий не дізнається, що доказ не потрапив в індекс. Краще гучний збій на
#: кожному такому файлі, ніж тиха відсутність.
UNDECODABLE_SUFFIXES = frozenset({".heic", ".heif"})

#: Пояснення замість «cannot identify image file». Загальна помилка Pillow
#: виглядає як пошкоджений файл, хоча файл цілий — бракує декодера, і його
#: свідомо немає.
UNDECODABLE_REASON = (
    "HEIC/HEIF не читається: декодера немає й не буде. Вільні реалізації "
    "копілефтні (libheif LGPL-3.0, libde265 LGPL-3.0), а HEVC ще й обтяжений "
    "патентними пулами — обидва порушують вимогу про комерційну придатність "
    "(ADR-028). Конвертуйте в JPEG або PNG перед індексацією: "
    "sips -s format jpeg ФАЙЛ.HEIC --out ФАЙЛ.jpg"
)


@dataclass
class IngestStats:
    scanned: int = 0
    indexed: int = 0
    #: Скільки файлів прочитано попри пошкодження.
    recovered: int = 0
    regions: int = 0
    facets: int = 0
    #: Скільки кадрів отримали опис словами.
    with_caption: int = 0
    #: Скільки кадрів дали непорожній текст. Число, за яким видно, чи
    #: окуповується OCR на цьому матеріалі взагалі.
    with_text: int = 0
    #: Чому не вдалося порахувати фасети. Порожньо — усе гаразд.
    facets_error: str = ""
    skipped_existing: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)
    elapsed_s: float = 0.0

    @property
    def regions_per_frame(self) -> float:
        """Коефіцієнт розростання індексу — головне число для оцінки масштабу.

        Саме воно вирішує, чи поміститься індекс у памʼять на мільйонах
        активів, тож міряється на кожному прогоні, а не оцінюється на око.
        """
        return self.regions / self.indexed if self.indexed else 0.0

    @property
    def rate(self) -> float:
        return self.indexed / self.elapsed_s if self.elapsed_s else 0.0

    def summary(self) -> str:
        parts = [
            f"переглянуто {self.scanned}",
            f"проіндексовано {self.indexed}",
        ]
        if self.regions:
            parts.append(f"регіонів {self.regions} ({self.regions_per_frame:.1f} на кадр)")
        if self.facets:
            parts.append(f"фасетів {self.facets}")
        if self.with_text:
            parts.append(f"з текстом {self.with_text}")
        if self.with_caption:
            parts.append(f"з описом {self.with_caption}")
        if self.recovered:
            parts.append(f"відновлено пошкоджених {self.recovered}")
        if self.skipped_existing:
            parts.append(f"пропущено як уже наявні {self.skipped_existing}")
        if self.failed:
            parts.append(f"з помилками {len(self.failed)}")
        if self.facets_error:
            parts.append(
                f"\n  ⚠ ФАСЕТИ НЕ ПОРАХОВАНО ({self.facets_error}). "
                f"Щільний пошук працює, але заперечення й фільтри за атрибутами — НІ"
            )
        if self.elapsed_s:
            parts.append(f"{self.rate:.1f} зобр/с")
        return ", ".join(parts)


def sha256_of(path: Path, chunk: int = 1 << 20, *, catalog=None) -> str:
    """sha256 вмісту — ідентифікатор активу.

    З каталогом хеш береться з кешу, якщо розмір і час зміни збігаються. Без
    цього кожен прогін перечитує КОЖЕН файл цілком лише для того, щоб
    відповісти «цей актив уже проіндексований» — тобто читає весь корпус, аби
    нічого не зробити (ADR-015).
    """
    stat = path.stat() if catalog is not None else None
    if stat is not None:
        cached = catalog.cached_hash(path, stat.st_size, stat.st_mtime)
        if cached:
            return cached
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    value = digest.hexdigest()
    if stat is not None:
        catalog.remember_hash(path, stat.st_size, stat.st_mtime, value)
    return value


def discover(root: Path) -> Iterator[Path]:
    """Знайти зображення в теці (або повернути сам файл, якщо це файл).

    Обхід ПОТОКОВИЙ: `sorted(root.rglob("*"))` матеріалізував увесь перелік
    файлів у памʼяті ще до першого кадру, тобто на мільйоні знімків платив
    памʼяттю до того, як зробив хоч щось.

    Порядок лишається сталим — теки й файли сортуються в межах кожного
    каталогу. Це не косметика: сталий порядок робить прогін відтворюваним, а
    відтворюваність тут вимога (п.7).
    """
    if root.is_file():
        if root.suffix.lower() in IMAGE_SUFFIXES:
            yield root
        return
    for directory, subdirs, files in os.walk(root):
        subdirs.sort()
        base = Path(directory)
        for name in sorted(files):
            if Path(name).suffix.lower() in IMAGE_SUFFIXES:
                yield base / name


def load_image(path: Path) -> "Image":
    """Відкрити зображення. Повертає лише картинку; про пошкодження — `load_checked`."""
    return load_checked(path)[0]


def load_checked(path: Path) -> tuple["Image", bool]:
    """Відкрити зображення, повернувши ще й ознаку пошкодження файлу.

    Два рішення, кожне з яких мовчки псує результат, якщо його не ухвалити:

    **EXIF-орієнтація.** Без `exif_transpose` фото з телефона індексується
    поверненим набік, і пошук по ньому просто не працює — без жодної помилки.

    **Обрізані файли.** PIL за замовчуванням відмовляється читати JPEG без
    маркера кінця зображення, навіть коли всі рядки пікселів на місці. У
    матеріалах справи такі файли звичайні — і відмовитися індексувати доказ
    через відсутній байт наприкінці гірше, ніж прочитати його з поміткою.
    Тому спершу пробуємо строго, а при невдачі — толерантно, і повертаємо
    `True`, щоб факт пошкодження потрапив у провенанс, а не загубився.
    """
    from PIL import Image, ImageFile, ImageOps

    def _open(tolerant: bool) -> "Image":
        previous = ImageFile.LOAD_TRUNCATED_IMAGES
        ImageFile.LOAD_TRUNCATED_IMAGES = tolerant
        try:
            with Image.open(path) as img:
                img.load()
                return ImageOps.exif_transpose(img).convert("RGB")
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = previous

    try:
        return _open(tolerant=False), False
    except OSError as exc:
        if "truncated" not in str(exc).lower():
            raise
        logger.warning("файл %s пошкоджений, читаємо частково: %s", path.name, exc)
        return _open(tolerant=True), True


def _assets_filter(asset_ids: list[str]):
    """Фільтр «точки цих активів» — обсяг інкрементального перерахунку."""
    from qdrant_client import models

    return models.Filter(must=[models.FieldCondition(
        key="asset_id", match=models.MatchAny(any=asset_ids)
    )])


def index_paths(
    root: Path | str,
    *,
    profile: Profile | None = None,
    store: VectorStore | None = None,
    catalog: Catalog | None = None,
    embedder: Siglip2Embedder | None = None,
    proposer: CompositeProposer | None = None,
    recreate: bool = False,
    batch_size: int = 8,
    skip_existing: bool = True,
    ocr_engine: object | None = None,
    index_regions: bool = True,
    apply_categories: bool = True,
) -> IngestStats:
    """Проіндексувати зображення з теки в колекцію `frames`."""
    profile = profile or get_profile()
    store = store or VectorStore()
    catalog = catalog or Catalog()
    embedder = embedder or Siglip2Embedder(profile=profile)
    # Регіони мають сенс лише тоді, коли профіль справді щось нарізає.
    wants_regions = index_regions and (profile.tiling.enabled or profile.use_region_proposals)
    if wants_regions and proposer is None:
        proposer = build_proposer(profile)

    # Підпис звіряється до першого запису: краще впасти зараз, ніж змішати
    # несумісні вектори і виявити це за якістю пошуку через тиждень.
    if recreate:
        store.drop_collection(schema.FRAMES)
        store.drop_collection(schema.REGIONS)
        # Каталог чиститься разом із векторами. Інакше він переживе видалення
        # колекцій і надалі стверджуватиме, що актив уже проіндексований, —
        # і наступний прогін мовчки пропустить файли, яких в індексі немає.
        removed = catalog.clear_assets()
        if removed:
            logger.info("перебудова індексу: з каталогу прибрано %d активів", removed)
        catalog.set_signature(profile.index_signature())
    else:
        catalog.assert_compatible(profile.index_signature())

    # Розріджений вектор існує лише там, де є що читати: він живе на кадрі,
    # а не на регіоні. Текст на плитці — той самий текст кадру, і дублювати
    # його по 33 регіонах означало б роздути індекс без жодного виграшу.
    store.ensure_collection(
        schema.FRAMES, profile.embed_dim, quantize=True,
        sparse=profile.use_ocr or profile.use_captions
    )
    if wants_regions:
        store.ensure_collection(schema.REGIONS, profile.embed_dim, quantize=True)

    info = embedder.info
    catalog.record_model("embed", info.model_name, info.repo_id, info.revision)

    reader = None
    if profile.use_ocr or profile.use_captions:
        from vsearch.represent.ocr import Florence2Ocr

        reader = ocr_engine or Florence2Ocr(profile)

    stats = IngestStats()
    started = time.perf_counter()
    #: Активи, записані САМЕ ЦИМ прогоном: лише їх і перераховуємо.
    fresh_assets: set[str] = set()
    pending: list[tuple[Path, str, "Image", bool]] = []

    def flush() -> None:
        if not pending:
            return
        vectors = embedder.embed_images([img for _, _, img, _ in pending])
        keys, payloads, sparse_vectors = [], [], []
        region_keys, region_crops, region_payloads = [], [], []

        for (path, asset_id, img, damaged), _ in zip(pending, vectors):
            fresh_assets.add(asset_id)
            frame_id = f"{asset_id}:0"
            keys.append(frame_id)
            ocr_text, ocr_engine_name, caption_text = "", "", ""
            if reader is not None:
                try:
                    result = reader.read(img)
                    ocr_text, ocr_engine_name = result.text, result.engine
                    if result.has_text:
                        stats.with_text += 1
                except Exception:  # noqa: BLE001 — один кадр не спиняє індексацію
                    logger.warning("OCR не вдався для %s", path.name, exc_info=True)
                if profile.use_captions:
                    try:
                        caption_text = reader.describe(img)
                        if caption_text:
                            stats.with_caption += 1
                    except Exception:  # noqa: BLE001
                        logger.warning("підпис не вдався для %s", path.name, exc_info=True)
            # Обидва джерела годують ОДИН розріджений вектор: для лексичного
            # пошуку немає різниці, звідки слово — з написа в кадрі чи з опису
            # того, що в ньому видно.
            lexical_source = " ".join(x for x in (ocr_text, caption_text) if x)
            sparse_vectors.append(
                lexical.build(lexical_source) if lexical_source else None
            )
            payloads.append(
                {
                    "asset_id": asset_id,
                    "frame_id": frame_id,
                    "media_type": "image",
                    "path": str(path),
                    "width": img.width,
                    "height": img.height,
                    "indexed_at": int(time.time()),
                    # Пошкодження джерела має дійти до матеріалів справи, а не
                    # лишитися в логах індексації.
                    "source_damaged": damaged,
                    # Текст зберігається й СЛОВАМИ, не лише розрідженим
                    # вектором: у матеріалах справи має бути видно, ЩО саме
                    # прочитано, а не тільки те, що збіг стався.
                    "ocr_text": ocr_text,
                    "ocr_engine": ocr_engine_name,
                    "caption": caption_text,
                }
            )
            if wants_regions and proposer is not None:
                # Цілий кадр уже лежить у frames — у regions він був би копією.
                for order, region in enumerate(
                    r for r in proposer.propose(img) if r.kind != "frame"
                ):  # noqa: E501
                    region_keys.append(f"{frame_id}:r{order}")
                    region_crops.append(region.crop(img))
                    region_payloads.append(
                        {
                            "asset_id": asset_id,
                            "frame_id": frame_id,
                            "path": str(path),
                            "region_type": region.kind,
                            "label": region.label,
                            "bbox": list(region.bbox),
                            "area_ratio": region.area_ratio,
                            "indexed_at": int(time.time()),
                        }
                    )
            catalog.add_asset(
                AssetRecord(
                    asset_id=asset_id,
                    path=str(path),
                    media_type="image",
                    size_bytes=path.stat().st_size,
                    width=img.width,
                    height=img.height,
                ),
                profile.name,
            )
            catalog.add_frame(frame_id, asset_id, ts_ms=None, width=img.width, height=img.height)
        store.upsert(
            schema.FRAMES, keys, vectors, payloads,
            sparse=sparse_vectors if reader is not None else None,
        )
        stats.indexed += len(keys)

        if region_crops:
            region_vectors = embedder.embed_images(region_crops)
            store.upsert(schema.REGIONS, region_keys, region_vectors, region_payloads)
            stats.regions += len(region_keys)
        pending.clear()

    for path in discover(Path(root)):
        stats.scanned += 1
        try:
            asset_id = sha256_of(path, catalog=catalog)
            if skip_existing and not recreate and catalog.has_asset(asset_id):
                stats.skipped_existing += 1
                continue
            image, damaged = load_checked(path)
            stats.recovered += damaged
            pending.append((path, asset_id, image, damaged))
        except Exception as exc:  # noqa: BLE001 — один зіпсований файл не спиняє прогін
            # Причина має бути ДІЄЮ, а не констатацією. «cannot identify image
            # file» виглядає як пошкоджений файл, хоча файл цілий і проблема
            # має відомий обхід.
            reason = (
                UNDECODABLE_REASON
                if path.suffix.lower() in UNDECODABLE_SUFFIXES
                else str(exc)
            )
            logger.warning("не вдалося прочитати %s: %s", path, reason)
            stats.failed.append((str(path), reason))
            continue
        if len(pending) >= batch_size:
            flush()
    flush()

    # Фасети проставляються ПІСЛЯ запису векторів, а не під час: це те саме
    # множення матриць, що й для нової категорії згодом, тож шлях один і той
    # самий і перевіряється тими самими сценаріями.
    if apply_categories and stats.indexed:
        from vsearch.represent.categories import CategoryEngine
        from vsearch.represent.prototypes import KIND_ATTRIBUTE, KIND_CATEGORY

        # Через apply_many, а не поштучно: воно робить один прохід по колекції
        # замість K і, головне, зводить взаємовиключні групи (колір, волосся)
        # до одного ключа. Поштучний шлях лишав сирі attr_color_* булеві, і
        # вони протікали туди, де їм не місце.
        #
        # kinds тут — вимога коректності, а не оптимізація: атрибути людини на
        # рівні кадру утворюють із cat_person конʼюнкцію, яка стверджує те,
        # чого в кадрі немає (ADR-005).
        # Збій тут НЕ мусить лишатися тихим. Вектори вже записані, тож індекс
        # виглядає справним і щільний пошук працює — а заперечення й фільтри
        # за статтю мовчки перестають діяти, бо фасетів просто немає. Саме так
        # і сталося при додаванні лексичного шару: вектори стали приходити
        # словником, обчислення фасетів упало всередині numpy, а індексація
        # доповіла про успіх.
        engine = CategoryEngine.with_defaults(embedder, store)
        # ІНКРЕМЕНТАЛЬНО: перераховуємо лише щойно додані активи. Раніше умовою
        # було «щось проіндексовано», тобто додавання одного знімка запускало
        # повний скрол по обох колекціях із вивантаженням кожного вектора
        # клієнту (ADR-015).
        #
        # Межі при цьому лишаються КОРПУСНИМИ: вони зберігаються в каталозі й
        # виводяться наново лише за суттєвого приросту. Інакше фасет на нових
        # точках означав би не те, що на старих, — це питання коректності, а
        # не швидкості.
        scope = _assets_filter(sorted(fresh_assets)) if fresh_assets else None
        try:
            engine.apply_many(
                schema.FRAMES, kinds=(KIND_CATEGORY,),
                scope_filter=scope, catalog=catalog,
            )
            if wants_regions:
                engine.apply_many(
                    schema.REGIONS, kinds=(KIND_CATEGORY, KIND_ATTRIBUTE),
                    scope_filter=scope, catalog=catalog,
                )
            # Індекси створюються ПІСЛЯ запису фасетів — раніше їх просто не
            # було б за чим будувати. Ідемпотентно, тож наявні не чіпаються.
            for collection in (schema.FRAMES, schema.REGIONS):
                store.ensure_payload_indexes(collection)
        except Exception as exc:  # noqa: BLE001 — але гучно
            logger.exception("обчислення фасетів не вдалося")
            stats.facets_error = f"{type(exc).__name__}: {exc}"
        for prototype in engine.bank:
            catalog.save_prototype(prototype)
        stats.facets = len(engine.bank)

    stats.elapsed_s = time.perf_counter() - started
    return stats
