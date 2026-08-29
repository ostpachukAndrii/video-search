"""Прототипи: механізм, на якому тримається і категоризація, і заперечення.

Фасет — це не окрема модель, а **калібрований косинус між уже збереженим
ембедингом кропа й текстовим прототипом**. Наслідок вирішальний для масштабу:
нова категорія чи новий атрибут застосовуються до всього наявного індексу
матричним множенням, без повторного читання пікселів.

Два режими оцінювання, бо задачі різні:

* **абсолютний** — «чи є на кадрі зброя?». Використовує власну сигмоїду
  SigLIP: модель тренована сигмоїдним лосом, тож `sigmoid(scale·cos + bias)`
  є поштучною ймовірністю пари, а не величиною, осмисленою лише відносно
  інших кандидатів. Тому працює без жодних розмічених даних — це критично
  для категорій, які користувач додає сам.
* **контрастний** — «в окулярах чи без?». Порівнює позитивний прототип із
  негативним і дивиться на різницю. Для бінарних атрибутів це надійніше за
  абсолютне значення, бо спільна для обох формулювань складова (тут — «людина»)
  скорочується.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Iterable, Sequence

import numpy as np

if TYPE_CHECKING:
    from PIL.Image import Image

    from vsearch.represent.embed import Siglip2Embedder

KIND_CATEGORY = "category"
KIND_ATTRIBUTE = "attribute"

#: Тло за замовчуванням для категорій, у яких негатив не вказано явно.
#:
#: Виміряно на M3: без негативу фасет непридатний (F1 0.00–0.20), бо власна
#: межа SigLIP розрахована на природні фото з описовими підписами й майже все
#: відсікає. Ці загальні фрази дають F1 0.955–0.981 — у межах 2–3% від ручно
#: дібраних негативів, і при цьому не потребують ані розмітки, ані зусиль
#: користувача. Саме це робить вимогу п.4 здійсненною: категорія, додана
#: самим лише текстом, працює одразу.
BACKGROUND_PROMPTS: tuple[str, ...] = (
    "a photo",
    "an image",
    "a picture of something",
    "an ordinary scene",
    "a background",
)


@dataclass(frozen=True)
class Prototype:
    """Опис фасета: як його сформулювати й де проходить межа рішення."""

    name: str
    positive: tuple[str, ...]
    negative: tuple[str, ...] = ()
    kind: str = KIND_CATEGORY
    #: Поріг рішення. None означає «взяти типовий для режиму»: 0.5 ймовірності
    #: для абсолютного, нульова різниця для контрастного.
    threshold: float | None = None
    #: Передумова: імʼя іншого прототипу, який має спрацювати, щоб цей фасет
    #: узагалі мав сенс. Атрибут «окуляри» осмислений лише там, де є людина.
    #:
    #: Без цього «attr_glasses=false» означало б «окулярів не видно», зокрема й
    #: на порожній кімнаті, — і запит «людина без окулярів» повертав би
    #: порожні кімнати першими. Тобто вимога п.11 виглядала б виконаною,
    #: насправді не працюючи.
    gate: str = ""

    #: Чи підібрано поріг на розмічених даних. Некалібрований фасет придатний
    #: для ранжування, але як жорсткий фільтр ненадійний — і споживач має про
    #: це знати, а не дізнаватися з поганих результатів.
    calibrated: bool = False
    #: Скільки прикладів-зображень підмішано в прототип (для провенансу).
    example_count: int = 0
    description: str = ""

    #: Чи дозволено підставити типове тло, якщо негатив не вказано.
    #: Вимикається лише свідомо — абсолютний режим потребує калібрування.
    auto_background: bool = True

    @property
    def is_contrastive(self) -> bool:
        return bool(self.negative)

    @property
    def payload_key(self) -> str:
        """Імʼя поля в payload Qdrant.

        Атрибути й категорії живуть у різних просторах імен, щоб фільтр
        `attr_glasses=false` не переплутався з категорією «glasses».
        """
        prefix = "attr_" if self.kind == KIND_ATTRIBUTE else "cat_"
        return f"{prefix}{self.name}"

    def default_threshold(self) -> float:
        if self.threshold is not None:
            return self.threshold
        return 0.0 if self.is_contrastive else 0.5


@dataclass
class PrototypeScores:
    """Результат оцінювання: сира величина, ймовірність і рішення."""

    name: str
    raw: np.ndarray          # різниця косинусів або сам косинус
    probability: np.ndarray  # у межах [0, 1]
    decision: np.ndarray     # булеве рішення за порогом

    def __len__(self) -> int:
        return len(self.raw)


class PrototypeBank:
    """Набір прототипів і оцінювання по них.

    Текстові вектори рахуються один раз при додаванні: далі оцінювання
    мільйона регіонів — це одне множення матриць.
    """

    def __init__(self, embedder: "Siglip2Embedder") -> None:
        self.embedder = embedder
        self._prototypes: dict[str, Prototype] = {}
        self._positive: dict[str, np.ndarray] = {}
        self._negative: dict[str, np.ndarray] = {}
        #: Кеш ембедингів наборів фраз. Типове тло однакове для всіх категорій,
        #: і без кешу воно рахувалося б заново на кожен прототип — на таксономії
        #: з тисячі класів це тисяча зайвих інференсів того самого.
        self._phrase_cache: dict[tuple[str, ...], np.ndarray] = {}

    def __len__(self) -> int:
        return len(self._prototypes)

    def __contains__(self, name: object) -> bool:
        return name in self._prototypes

    def __iter__(self):
        return iter(self._prototypes.values())

    @property
    def names(self) -> list[str]:
        return sorted(self._prototypes)

    def get(self, name: str) -> Prototype:
        try:
            return self._prototypes[name]
        except KeyError:
            raise KeyError(
                f"Прототип {name!r} не зареєстровано. Відомі: {', '.join(self.names)}"
            ) from None

    # ── реєстрація ──────────────────────────────────────────────────────────

    def add(self, prototype: Prototype, examples: Sequence["Image"] = ()) -> Prototype:
        """Зареєструвати прототип, за потреби підмішавши приклади-зображення.

        Приклади уточнюють формулювання там, де слова заслабкі: «саморобний
        вибуховий пристрій» текстом описується погано, а пʼятьма знімками —
        значно краще. Текст лишається завжди: без нього прототип не переносився б
        на матеріали, не схожі на приклади.
        """
        # Без негативу фасет майже завжди непридатний, тому тло підставляється
        # автоматично. Виняток лишається явним: auto_background=False.
        if not prototype.negative and prototype.auto_background:
            prototype = replace(prototype, negative=BACKGROUND_PROMPTS)

        positive = self._embed_phrases(prototype.positive)
        if len(examples):
            image_vectors = self.embedder.embed_images(list(examples))
            # Текст і приклади з рівною вагою: приклади уточнюють, а не
            # заміщають, інакше прототип звузився б до кількох знімків.
            positive = _unit(positive + _unit(image_vectors.mean(axis=0)))
            prototype = replace(prototype, example_count=len(examples))

        self._prototypes[prototype.name] = prototype
        self._positive[prototype.name] = positive
        if prototype.negative:
            self._negative[prototype.name] = self._embed_phrases(prototype.negative)
        return prototype

    def add_many(self, prototypes: Iterable[Prototype]) -> None:
        self.add_bulk(list(prototypes))

    def _embed_phrases(self, phrases: Sequence[str]) -> np.ndarray:
        """Ансамбль формулювань → один одиничний вектор.

        Усереднення кількох перефразувань помітно стабільніше за одну фразу:
        окреме формулювання може випадково зачепити нерелевантний напрямок.
        """
        key = tuple(phrases)
        cached = self._phrase_cache.get(key)
        if cached is not None:
            return cached
        vectors = self.embedder.embed_texts(list(phrases), use_template=False)
        result = _unit(vectors.mean(axis=0))
        self._phrase_cache[key] = result
        return result

    def add_bulk(self, prototypes: Sequence[Prototype]) -> None:
        """Зареєструвати багато прототипів, ембедячи всі фрази одним батчем.

        Поштучне додавання робить виклик моделі на кожен прототип. Для
        таксономії на тисячу класів це тисяча дрібних викликів там, де
        достатньо кількох великих.
        """
        prepared = [
            replace(p, negative=BACKGROUND_PROMPTS)
            if not p.negative and p.auto_background else p
            for p in prototypes
        ]
        groups: list[tuple[str, ...]] = []
        seen: set[tuple[str, ...]] = set()
        for prototype in prepared:
            for phrases in (tuple(prototype.positive), tuple(prototype.negative)):
                if phrases and phrases not in seen and phrases not in self._phrase_cache:
                    seen.add(phrases)
                    groups.append(phrases)

        flat = [phrase for group in groups for phrase in group]
        if flat:
            vectors = self.embedder.embed_texts(flat, use_template=False)
            offset = 0
            for group in groups:
                chunk = vectors[offset : offset + len(group)]
                self._phrase_cache[group] = _unit(chunk.mean(axis=0))
                offset += len(group)

        for prototype in prepared:
            self._prototypes[prototype.name] = prototype
            self._positive[prototype.name] = self._phrase_cache[tuple(prototype.positive)]
            if prototype.negative:
                self._negative[prototype.name] = self._phrase_cache[tuple(prototype.negative)]

    # ── оцінювання ──────────────────────────────────────────────────────────

    def score(self, vectors: np.ndarray, name: str) -> PrototypeScores:
        """Оцінити матрицю ембедингів (N, dim) за одним прототипом."""
        prototype = self.get(name)
        scale, bias = self.embedder.calibration
        cos_pos = vectors @ self._positive[name]

        if prototype.is_contrastive:
            cos_neg = vectors @ self._negative[name]
            raw = cos_pos - cos_neg
            # Різницю проганяємо через ту саму шкалу моделі, але без bias:
            # bias калібрує абсолютну відповідність, а тут порівнюються
            # два однаково зміщені значення, і він скоротився б.
            probability = _sigmoid(raw * scale)
        else:
            raw = cos_pos
            probability = _sigmoid(cos_pos * scale + bias)

        threshold = prototype.default_threshold()
        decision = (raw > threshold) if prototype.is_contrastive else (probability > threshold)
        return PrototypeScores(name=name, raw=raw, probability=probability, decision=decision)

    def score_many(
        self, vectors: np.ndarray, names: Sequence[str] | None = None
    ) -> dict[str, PrototypeScores]:
        """Оцінити всі прототипи ЗА ОДИН прохід матричного множення.

        Різниця не в самому множенні, а в тому, скільки разів доводиться
        торкатися даних. Поштучне оцінювання робить K проходів по векторах;
        тут їх два — по одному на позитивні й негативні прототипи. Для
        таксономії на тисячу класів це різниця між півгодиною і секундами,
        тобто між «таксономію треба обирати заздалегідь» і «її можна
        приміряти».
        """
        chosen = list(names) if names is not None else self.names
        if not chosen:
            return {}

        positive = np.stack([self._positive[name] for name in chosen])
        cos_pos = vectors @ positive.T  # (M, K)

        # Негативи є не в усіх; для тих, де їх немає, ставимо нульовий вектор,
        # і різниця вироджується в сам косинус — тобто в абсолютний режим.
        negative = np.stack([
            self._negative.get(name, np.zeros(vectors.shape[1], dtype=np.float32))
            for name in chosen
        ])
        cos_neg = vectors @ negative.T

        scale, bias = self.embedder.calibration
        result: dict[str, PrototypeScores] = {}
        for index, name in enumerate(chosen):
            prototype = self._prototypes[name]
            if prototype.is_contrastive:
                raw = cos_pos[:, index] - cos_neg[:, index]
                probability = _sigmoid(raw * scale)
            else:
                raw = cos_pos[:, index]
                probability = _sigmoid(raw * scale + bias)
            threshold = prototype.default_threshold()
            decision = (
                raw > threshold if prototype.is_contrastive else probability > threshold
            )
            result[name] = PrototypeScores(
                name=name, raw=raw, probability=probability, decision=decision
            )
        return result

    def score_all(self, vectors: np.ndarray) -> dict[str, PrototypeScores]:
        return self.score_many(vectors)

    # ── калібрування ────────────────────────────────────────────────────────

    def suggest_threshold(
        self,
        name: str,
        vectors: np.ndarray,
        *,
        max_positive_ratio: float = 0.25,
        min_gap: float = 0.01,
    ) -> tuple[float | None, np.ndarray]:
        """Запропонувати межу за розривом у розподілі — без жодної розмітки.

        Спирається на дві властивості, виміряні на M3:

        * **знак різниці відсікає відсутнє.** Категорія, якої в корпусі немає
          зовсім, дає всі різниці відʼємними — тож поріг ніколи не опускається
          нижче нуля, і «гелікоптер» серед намальованих фігур дає порожньо;
        * **між влучаннями і рештою є розрив.** Для «жовтого трикутника» дві
          цілі мали 0.115 і 0.094, а наступний кандидат — 0.053.

        Розрив шукається лише серед додатних різниць і лише у верхній частині
        корпусу: фасет, що вмикається для більш ніж чверті всього, нічого не
        розрізняє, і найбільший розрив у його «хвості» був би випадковим.
        """
        scores = self.score(vectors, name)
        values = np.sort(scores.raw)[::-1]
        positive = values[values > 0]
        if len(positive) < 1:
            return 0.0, np.zeros(len(scores.raw), dtype=bool)

        # Найінформативніший розрив часто саме на межі додатних і відʼємних:
        # коли цілей мало, обрив від останньої з них до решти і є відповіддю.
        # Без цього хвостового нуля така межа лишалася б поза розглядом.
        candidates = np.append(positive, 0.0)
        # Розрив шукаємо спершу у верхній чверті, потім ширше. Вибіркова
        # категорія дає обрив рано; поширена — пізніше, і обмежувати пошук
        # чвертю означало б оголошувати «межі немає» там, де вона просто далі.
        best, gaps, candidates_used = -1, None, candidates
        for ratio in (max_positive_ratio, 0.6):
            horizon = max(1, min(len(candidates) - 1, int(len(values) * ratio) + 1))
            trial = candidates[:horizon] - candidates[1 : horizon + 1]
            index = int(np.argmax(trial))
            if trial[index] >= min_gap:
                best, gaps = index, trial
                break
        if best < 0:
            # Виразного розриву немає — межі на цьому корпусі не існує.
            #
            # Раніше тут поверталася межа по нулю, і це було гірше за
            # відсутність відповіді: «різниця > 0» означає лише «ближче до
            # позитивного формулювання, ніж до негативного», тож фасет
            # спрацьовував на 84% регіонів і нічого не розрізняв.
            #
            # None означає «визначити не вдалося». Оцінка лишається — за нею
            # можна ранжувати; булеве твердження не робиться. Для слідчого
            # ранжований перелік чесніший за мітку, хибну в 8 випадках з 10.
            return None, scores.raw > 0.0
        threshold = float((candidates_used[best] + candidates_used[best + 1]) / 2)
        return threshold, scores.raw > threshold

    def calibrate(
        self,
        name: str,
        vectors: np.ndarray,
        labels: Sequence[bool],
        *,
        min_samples: int = 4,
    ) -> tuple[Prototype, dict[str, float]]:
        """Підібрати поріг за розміченими прикладами, максимізуючи F1.

        Типовий поріг працює без жодних даних — це і є цінність підходу. Але
        коли розмітка є, її гріх не використати: межа зсувається під конкретні
        матеріали, а не під «середню фотографію з інтернету».
        """
        prototype = self.get(name)
        labels_array = np.asarray(labels, dtype=bool)
        if len(labels_array) != len(vectors):
            raise ValueError(f"розбіжність: {len(vectors)} векторів і {len(labels_array)} міток")
        if labels_array.sum() < min_samples or (~labels_array).sum() < min_samples:
            raise ValueError(
                f"замало прикладів для калібрування {name!r}: "
                f"позитивних {int(labels_array.sum())}, негативних {int((~labels_array).sum())}, "
                f"потрібно щонайменше {min_samples} кожного"
            )

        scores = self.score(vectors, name)
        values = scores.raw if prototype.is_contrastive else scores.probability

        best_threshold, best_f1 = prototype.default_threshold(), -1.0
        for candidate in np.unique(values):
            f1 = _f1(values > candidate, labels_array)
            if f1 > best_f1:
                best_threshold, best_f1 = float(candidate), f1

        calibrated = replace(prototype, threshold=best_threshold, calibrated=True)
        self._prototypes[name] = calibrated
        metrics = {
            "threshold": best_threshold,
            "f1": best_f1,
            "f1_default": _f1(values > prototype.default_threshold(), labels_array),
            "positives": int(labels_array.sum()),
            "negatives": int((~labels_array).sum()),
        }
        return calibrated, metrics


def _unit(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return (vector / max(float(np.linalg.norm(vector)), eps)).astype(np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def _f1(predicted: np.ndarray, actual: np.ndarray) -> float:
    true_positive = int((predicted & actual).sum())
    if not true_positive:
        return 0.0
    precision = true_positive / int(predicted.sum())
    recall = true_positive / int(actual.sum())
    return 2 * precision * recall / (precision + recall)
