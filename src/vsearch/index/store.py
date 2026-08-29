"""Обгортка над Qdrant.

Тримає три речі, які легко зробити неправильно й важко помітити:

* **бінарну квантизацію з rescore** — без `rescore=True` пошук іде лише по
  бінарних векторах, і якість тихо просідає;
* **payload-індекси** — без них фільтр працює, але вироджується в повний
  перебір, а фільтри це наш механізм заперечення (п.11);
* **детерміновані ідентифікатори точок** — Qdrant приймає лише UUID або ціле,
  а наші ключі рядкові, тож переіндексація має давати ті самі id, інакше
  замість оновлення виникнуть дублікати.
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from vsearch.index import schema

logger = logging.getLogger(__name__)

DEFAULT_URL = os.environ.get("VSEARCH_QDRANT_URL", "http://localhost:6333")

#: Префікс імен колекцій. Тести працюють на тому самому сервері Qdrant, що й
#: розробник, тому мусять мати власний простір імен — інакше прогін тестів
#: знищив би робочий індекс.
DEFAULT_PREFIX = os.environ.get("VSEARCH_COLLECTION_PREFIX", "")

#: Простір імен для uuid5. Фіксований і незмінний: від нього залежить,
#: чи повторна індексація оновить точку, чи створить другу.
_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")


def point_id(key: str) -> str:
    """Рядковий ключ → детермінований UUID точки."""
    return str(uuid.uuid5(_NAMESPACE, key))


@dataclass
class Hit:
    """Результат пошуку разом із тим, чому він знайдений (п.7)."""

    point_id: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)
    #: Інші регіони того самого кадру, що теж відповіли на запит.
    #: Заповнюється агрегацією; потрібне, щоб показати всі знахідки, а не одну.
    siblings: list["Hit"] = field(default_factory=list)
    #: Вектор точки. Піднімається на вимогу — потрібен, щоб оцінити кандидата
    #: прототипом, побудованим із запиту, без звернення до передобчислених
    #: фасетів.
    vector: list[float] | None = None
    #: Оцінка злиття кількох ранжувань (RRF), якщо воно застосовувалося.
    #:
    #: Тримається ОКРЕМО від `score` навмисно: `score` має лишатися справжнім
    #: косинусом, бо з нього рахується впевненість, яку бачить людина. А от
    #: порядок за косинусом порівнювати між каналами не можна — абсолютний
    #: рівень залежить від самого тексту запиту.
    fusion_score: float | None = None
    #: Колекція, з якої піднято точку. Потрібна, щоб пізніше дістати її вектор
    #: і переоцінити канонічним текстом: без цього довелося б угадувати між
    #: `frames` і `regions`.
    collection: str = ""
    #: Наскільки впевнено ця точка підтверджує свою сутність — за ФАСЕТАМИ,
    #: а не за косинусом із текстом запиту.
    #:
    #: Це різні виміри тієї самої моделі, і плутати їх не можна. Фасет
    #: контрастний: «людина» проти загального тла. Запит — абсолютна
    #: схожість із фразою. На дзеркальному селфі фасет каже «людина 0.85», а
    #: текст «a woman» дає 0.0% — бо модель тренована на підписах, де жінка
    #: означає видиму постать, а не фрагмент у дзеркалі. Підписати рамку
    #: «woman» і поставити поруч 4.7% означає показати число не про те.
    entity_score: float | None = None
    #: Яку СУТНІСТЬ запиту підтверджує ця точка («жінка», «дитина»).
    #:
    #: Порожньо для звичайного щільного пошуку. Заповнюється жорстким
    #: проходом, який шукає кожну сутність окремо, — і саме ця інформація
    #: дозволяє показати в кадрі рамку «жінка» і рамку «дитина» окремо,
    #: замість двох безіменних прямокутників.
    entity: str = ""

    @property
    def order_key(self) -> float:
        """За чим упорядковувати: рангова оцінка, якщо є, інакше косинус."""
        return self.fusion_score if self.fusion_score is not None else self.score

    @property
    def asset_id(self) -> str:
        return self.payload.get("asset_id", "")

    @property
    def frame_id(self) -> str:
        return self.payload.get("frame_id") or self.payload.get("id", "")


class VectorStore:
    """Підключення до Qdrant і робота з колекціями проєкту."""

    def __init__(
        self,
        url: str | None = None,
        timeout: int = 30,
        prefix: str | None = None,
    ) -> None:
        from qdrant_client import QdrantClient

        self.url = url or DEFAULT_URL
        self.prefix = DEFAULT_PREFIX if prefix is None else prefix
        self.client = QdrantClient(url=self.url, timeout=timeout)

    def name(self, collection: str) -> str:
        """Логічне імʼя колекції → фактичне з урахуванням префікса."""
        return f"{self.prefix}{collection}"

    # ── життєвий цикл колекцій ──────────────────────────────────────────────

    def ensure_collection(
        self,
        name: str,
        dim: int,
        *,
        quantize: bool = True,
        recreate: bool = False,
        sparse: bool = False,
    ) -> bool:
        """Створити колекцію з квантизацією та payload-індексами.

        Повертає True, якщо колекція була створена цим викликом.
        """
        from qdrant_client import models

        logical = name
        name = self.name(name)
        exists = self.client.collection_exists(name)
        if exists and not recreate:
            return False
        if exists:
            logger.warning("перестворення колекції %s — наявні вектори буде втрачено", name)
            self.client.delete_collection(name)

        params = schema.vector_params(dim, quantize=quantize)
        self.client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=params["size"],
                distance=models.Distance.COSINE,
                on_disk=params["on_disk"],
            ),
            quantization_config=(
                models.BinaryQuantization(
                    binary=models.BinaryQuantizationConfig(always_ram=True)
                )
                if quantize
                else None
            ),
            # IDF рахує Qdrant, частоти термів — клієнт. Так лексичний шар
            # обходиться без жодних ваг: серверне обчислення BM25 у Qdrant
            # вимагало б інференсу моделі, а середовище виконання без мережі
            # її не дістане.
            sparse_vectors_config=(
                {
                    schema.SPARSE_TEXT: models.SparseVectorParams(
                        modifier=models.Modifier.IDF,
                    )
                }
                if sparse
                else None
            ),
        )

        self.ensure_payload_indexes(logical)
        logger.info("створено колекцію %s (dim=%d, квантизація=%s)", name, dim, quantize)
        return True

    def ensure_payload_indexes(self, name: str) -> list[str]:
        """Створити payload-індекси, яких ще немає. Повертає додані.

        Викликається НЕ лише при створенні колекції. `ensure_collection`
        виходить одразу, якщо колекція вже є, — тож поле, додане в схему
        пізніше, ніколи не діставалося наявного індексу й мовчки лишалося без
        нього. Для фасетів це означало повний перебір при кожному фільтрі.

        Ідемпотентно: наявні індекси не чіпаються, тому виклик безпечний на
        кожній індексації.
        """
        from qdrant_client import models  # noqa: F401  (узгодженість імпортів)

        full = self.name(name)
        logical = name[len(self.prefix):] if self.prefix else name
        wanted = list(schema.PAYLOAD_BY_COLLECTION.get(logical, ()))
        if logical in (schema.FRAMES, schema.REGIONS):
            wanted += list(schema.facet_fields())

        try:
            existing = set(self.client.get_collection(full).payload_schema or {})
        except Exception:  # noqa: BLE001 — колекції ще може не бути
            existing = set()

        added: list[str] = []
        for field_def in wanted:
            if field_def.name in existing:
                continue
            try:
                self.client.create_payload_index(
                    collection_name=full,
                    field_name=field_def.name,
                    field_schema=field_def.schema,
                )
                added.append(field_def.name)
            except Exception:  # noqa: BLE001 — один індекс не спиняє решту
                logger.warning("не вдалося створити індекс %s", field_def.name)
        if added:
            logger.info("додано payload-індексів у %s: %d", full, len(added))
        return added

    def drop_collection(self, name: str) -> None:
        name = self.name(name)
        if self.client.collection_exists(name):
            self.client.delete_collection(name)

    def count(self, name: str, *, exact: bool = True) -> int:
        """Скільки точок у колекції.

        `exact=False` бере оцінку з метаданих замість повного підрахунку. Це
        не дрібниця вартості: точний підрахунок пробігає найбільшу колекцію
        цілком, а робиться він у щільному проході заради єдиного питання «чи
        вона не порожня». На мільйонах це ціна на КОЖНОМУ запиті (ADR-015).
        """
        name = self.name(name)
        if not self.client.collection_exists(name):
            return 0
        if not exact:
            return int(self.client.get_collection(name).points_count or 0)
        return self.client.count(collection_name=name, exact=True).count

    # ── запис ───────────────────────────────────────────────────────────────

    def upsert(
        self,
        name: str,
        keys: Sequence[str],
        vectors: np.ndarray,
        payloads: Sequence[dict[str, Any]],
        *,
        batch_size: int = 256,
        sparse: Sequence[Any] | None = None,
    ) -> int:
        """Записати вектори з payload. Ключі рядкові, id — детерміновані."""
        from qdrant_client import models

        if not (len(keys) == len(vectors) == len(payloads)):
            raise ValueError(
                f"розбіжність довжин: keys={len(keys)}, "
                f"vectors={len(vectors)}, payloads={len(payloads)}"
            )

        name = self.name(name)
        written = 0
        for start in range(0, len(keys), batch_size):
            stop = start + batch_size
            chunk = list(zip(
                keys[start:stop], vectors[start:stop], payloads[start:stop]
            ))
            sparse_chunk = list(sparse[start:stop]) if sparse is not None else None
            points = []
            for offset, (key, vector, payload) in enumerate(chunk):
                if sparse_chunk is None:
                    payload_vector: Any = vector.tolist()
                else:
                    # Безіменний щільний вектор адресується порожнім імʼям —
                    # так Qdrant дозволяє додати іменований розріджений, не
                    # ламаючи наявні запити до щільного.
                    payload_vector = {"": vector.tolist()}
                    vec = sparse_chunk[offset]
                    if vec is not None and not vec.is_empty:
                        payload_vector[schema.SPARSE_TEXT] = models.SparseVector(
                            indices=list(vec.indices), values=list(vec.values)
                        )
                points.append(
                    models.PointStruct(
                        id=point_id(key), vector=payload_vector, payload=payload
                    )
                )
            self.client.upsert(collection_name=name, points=points, wait=True)
            written += len(points)
        return written

    # ── читання ─────────────────────────────────────────────────────────────

    def search(
        self,
        name: str,
        vector: np.ndarray,
        *,
        limit: int = 10,
        query_filter: Any = None,
        oversampling: float = schema.DEFAULT_OVERSAMPLING,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> list[Hit]:
        """Пошук найближчих із rescore оригінальними векторами.

        `rescore=True` тут принциповий: бінарний вектор дає грубий відбір, а
        остаточний порядок рахується по оригіналах. Без цього економія памʼяті
        купується падінням якості, чого вимога п.10 не дозволяє.
        """
        from qdrant_client import models

        response = self.client.query_points(
            collection_name=self.name(name),
            query=vector.tolist(),
            limit=limit,
            query_filter=query_filter,
            with_payload=with_payload,
            with_vectors=with_vectors,
            search_params=models.SearchParams(
                quantization=models.QuantizationSearchParams(
                    ignore=False,
                    rescore=True,
                    oversampling=oversampling,
                )
            ),
        )
        return [
            Hit(
                point_id=str(p.id),
                score=float(p.score),
                payload=p.payload or {},
                vector=p.vector if with_vectors else None,
                collection=name,
            )
            for p in response.points
        ]

    def search_sparse(
        self,
        name: str,
        sparse_vec: Any,
        *,
        limit: int = 10,
        query_filter: Any = None,
    ) -> list[Hit]:
        """Пошук лише лексичним шаром. Оцінка — BM25, НЕ косинус.

        Віддається окремо від щільного навмисно. Спокуса злити їх усередині
        Qdrant велика (`FusionQuery` там є), але тоді назовні виходить RRF-бал
        у діапазоні 0.2–0.8, який далі змішується з косинусами регіонів
        (~0.09) — і кадр виграє не тому, що краще підходить, а тому, що його
        оцінка з іншої шкали. Саме така помилка вже траплялася двічі
        (ADR-013), тож шкали розділені на рівні API.
        """
        from qdrant_client import models

        if sparse_vec is None or sparse_vec.is_empty:
            return []
        response = self.client.query_points(
            collection_name=self.name(name),
            query=models.SparseVector(
                indices=list(sparse_vec.indices), values=list(sparse_vec.values)
            ),
            using=schema.SPARSE_TEXT,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )
        return [
            Hit(
                point_id=str(p.id), score=float(p.score),
                payload=p.payload or {}, collection=name,
            )
            for p in response.points
        ]

    def search_hybrid(
        self,
        name: str,
        vector: np.ndarray,
        sparse_vec: Any,
        *,
        limit: int = 10,
        query_filter: Any = None,
        oversampling: float = schema.DEFAULT_OVERSAMPLING,
        candidates: int = 100,
    ) -> list[Hit]:
        """Щільний і лексичний пошук зі злиттям за рангами на боці Qdrant.

        Злиття саме RRF, і з тієї самої причини, що й для двох мовних каналів
        (ADR-013): косинус щільного вектора та вага BM25 лежать у різних
        шкалах, тож будь-яке порівняння їхніх ВЕЛИЧИН довільне. Ранги
        порівнювані за побудовою.

        Якщо лексичний вектор порожній (у запиті немає жодного терма, або
        колекція без розрідженого індексу), запит вироджується у звичайний
        щільний — мовчки повертати нуль було б гірше за відсутність шару.
        """
        from qdrant_client import models

        if sparse_vec is None or sparse_vec.is_empty:
            return self.search(
                name, vector, limit=limit, query_filter=query_filter,
                oversampling=oversampling,
            )

        response = self.client.query_points(
            collection_name=self.name(name),
            prefetch=[
                models.Prefetch(
                    query=vector.tolist(),
                    using="",
                    limit=candidates,
                    filter=query_filter,
                    params=models.SearchParams(
                        quantization=models.QuantizationSearchParams(
                            ignore=False, rescore=True, oversampling=oversampling,
                        )
                    ),
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=list(sparse_vec.indices),
                        values=list(sparse_vec.values),
                    ),
                    using=schema.SPARSE_TEXT,
                    limit=candidates,
                    filter=query_filter,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        return [
            Hit(
                point_id=str(p.id), score=float(p.score),
                payload=p.payload or {}, collection=name,
            )
            for p in response.points
        ]

    @staticmethod
    def _dense_of(vector: Any) -> Any:
        """Витягти щільний вектор із того, що повернув Qdrant.

        У колекції БЕЗ розріджених векторів приходить простий список, у
        колекції з ними — словник `{"": щільний, "text": розріджений}`.
        Пропустити цю різницю коштувало дорого: додавання лексичного шару
        зробило вектори неоднорідними, і весь етап обчислення фасетів упав
        усередині numpy. Індексація при цьому завершилася з кодом 0, а фасети
        просто не зʼявилися — тобто заперечення й фільтр статі перестали
        працювати мовчки.
        """
        if isinstance(vector, dict):
            return vector.get("")
        return vector

    def frame_ids_matching(
        self, name: str, query_filter: Any, *, page: int = 4096,
        cap: int | None = None,
    ) -> tuple[set[str], bool]:
        """Усі `frame_id`, що підпадають під фільтр — ТОЧНО, не top-N.

        Для заперечення це принципово. Раніше заборонені кадри бралися як
        top-N НАЙСХОЖІШИХ на позитивний запит серед тих, що підпадають під
        заборону, — тобто кадр із забороненою ознакою за межами цих N тихо
        проходив. Стенд масштабу показав, що межа вже пройдена: за
        `cat_underwear` підпадає 11 508 точок при вибірці 2 000.

        Пропущене виключення дає ВПЕВНЕНО ХИБНИЙ результат — найгірший
        різновид помилки в розслідуванні (п.11), гірший за зайвого кандидата.

        Вектори не читаються: потрібен лише `frame_id`, тож ціна — обхід
        payload, а не 37 МБ векторів на запит.

        Повертає `(кадри, чи обірвано)`. Обрив НЕ мовчазний: викликач мусить
        сказати про нього вголос, інакше точність підміниться на здогад
        непомітно.
        """
        name = self.name(name)
        if not self.client.collection_exists(name):
            return set(), False
        found: set[str] = set()
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=name, scroll_filter=query_filter,
                limit=page, offset=offset,
                with_payload=["frame_id"], with_vectors=False,
            )
            found.update(
                str(p.payload.get("frame_id") or p.id) for p in points
            )
            if offset is None:
                return found, False
            if cap is not None and len(found) >= cap:
                return found, True

    def fetch_vectors(self, name: str, point_ids: Sequence[str]) -> dict[str, list[float]]:
        """Дістати вектори за ідентифікаторами точок — одним запитом.

        Потрібне для переоцінки верхівки одним каноничним текстом: інакше
        показане число й порядок видачі рахувалися б різними величинами.
        """
        if not point_ids:
            return {}
        records = self.client.retrieve(
            collection_name=self.name(name),
            ids=list(point_ids),
            with_vectors=True,
            with_payload=False,
        )
        out: dict[str, list[float]] = {}
        for record in records:
            vector = self._dense_of(record.vector)
            if vector is not None:
                out[str(record.id)] = list(vector)
        return out

    def iter_vectors(
        self, name: str, batch_size: int = 512, *,
        payload_fields: Sequence[str] | None = None, query_filter: Any = None,
    ):
        """Пройти колекцію, віддаючи (ідентифікатори, вектори[, payload]) партіями.

        Це шлях, яким нова категорія застосовується до готового індексу:
        читаються ЛИШЕ вектори, без вихідних файлів і без моделей зображень.
        Тому вартість не залежить від обсягу оригіналів.

        `payload_fields` існує, щоб цей обхід не доводилося переписувати
        деінде. Раніше поруч жила друга, майже така сама реалізація — і коли
        додавання розрідженого шару зробило вектори словниками, виправлення
        першої не полагодило другу. Одна помилка, два місця, полагоджене одне:
        фасети мовчки перестали рахуватися.
        """
        collection = self.name(name)
        if not self.client.collection_exists(collection):
            return
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=collection,
                limit=batch_size,
                offset=offset,
                scroll_filter=query_filter,
                with_payload=list(payload_fields) if payload_fields else False,
                with_vectors=True,
            )
            if not points:
                return
            ids = [p.id for p in points]
            vectors = [self._dense_of(p.vector) for p in points]
            if payload_fields is None:
                yield ids, vectors
            else:
                yield ids, vectors, [p.payload or {} for p in points]
            if offset is None:
                return

    def set_payloads(
        self,
        name: str,
        ids: Sequence[Any],
        payloads: Sequence[dict[str, Any]],
        *,
        wait: bool = True,
    ) -> int:
        """Проставити РІЗНІ payload різним точкам за один виклик.

        Точки з однаковим payload групуються: булеве рішення фасета дає лише
        два значення, а оцінку округлюємо до сотої. Замість N мережевих
        викликів виходить кількадесят — на мільйонах точок це різниця між
        хвилинами й годинами.
        """
        from qdrant_client import models

        if len(ids) != len(payloads):
            raise ValueError(f"розбіжність: {len(ids)} точок і {len(payloads)} payload")

        groups: dict[tuple, list[Any]] = {}
        for point_id, payload in zip(ids, payloads):
            key = tuple(sorted(
                (k, round(v, 2) if isinstance(v, float) else v) for k, v in payload.items()
            ))
            groups.setdefault(key, []).append(point_id)

        operations = [
            models.SetPayloadOperation(
                set_payload=models.SetPayload(payload=dict(key), points=list(point_ids))
            )
            for key, point_ids in groups.items()
        ]
        self.client.batch_update_points(
            collection_name=self.name(name), update_operations=operations, wait=wait
        )
        return len(operations)

    def scroll_payloads(self, name: str, limit: int = 100) -> list[dict[str, Any]]:
        """Прочитати payload перших точок — для діагностики та тестів."""
        points, _ = self.client.scroll(
            collection_name=self.name(name), limit=limit, with_payload=True
        )
        return [p.payload or {} for p in points]

    def collections(self) -> list[str]:
        return [c.name for c in self.client.get_collections().collections]

    def is_alive(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:  # noqa: BLE001 — будь-яка помилка означає «недоступний»
            return False


def build_bound_filter(
    conditions: Iterable[tuple[str, Any]],
    excluded: Iterable[tuple[str, Any]] = (),
) -> Any:
    """Фільтр, усі умови якого мають виконатися на ОДНІЙ точці.

    Це і є механізм звʼязування. Qdrant застосовує `must` до кожної точки
    окремо, тож `[("attr_gender","male"), ("attr_clothing_color","red")]`
    вимагає, щоб обидві ознаки належали одному кропу, а не просто зустрічалися
    десь у кадрі. Групування за `frame_id` робиться вже після пошуку.

    Через це фільтр зі звʼязаними умовами має застосовуватися до колекції
    `regions`: на `frames` та сама конʼюнкція означала б лише «обидві ознаки
    десь у кадрі є» і давала б хибні спрацювання.
    """
    return build_filter(must=conditions, must_not=excluded)


def build_filter(
    must: Iterable[tuple[str, Any]] = (),
    must_not: Iterable[tuple[str, Any]] = (),
) -> Any:
    """Зібрати фільтр Qdrant із пар (поле, значення).

    Саме тут «без окулярів» перетворюється на точний предикат замість здогадки
    за косинусом — механізм, заради якого фасети рахуються при індексації.
    """
    from qdrant_client import models

    def conditions(pairs: Iterable[tuple[str, Any]]) -> list[Any]:
        return [
            models.FieldCondition(key=key, match=models.MatchValue(value=value))
            for key, value in pairs
        ]

    must_list = conditions(must)
    must_not_list = conditions(must_not)
    if not must_list and not must_not_list:
        return None
    return models.Filter(must=must_list or None, must_not=must_not_list or None)
