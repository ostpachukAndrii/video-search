"""Ембедер SigLIP 2 NaFlex — спільний векторний простір для зображень і тексту.

Три деталі, кожна з яких тихо псує якість, нічого не ламаючи явно:

1. **Нижній регістр.** Модель тренували на тексті в нижньому регістрі. Запит
   "Чоловік БЕЗ окулярів" без нормалізації дає гірший вектор, ніж той самий
   у нижньому регістрі, і жодної помилки при цьому не виникає.
2. **padding="max_length".** Текстова вежа SigLIP працює з фіксованою довжиною
   64 токени. Динамічний паддінг мовчки зміщує ембединги.
3. **L2-нормалізація.** Qdrant рахує косинус як скалярний добуток лише за умови
   одиничної норми; без неї бінарна квантизація теж деградує.
"""

from __future__ import annotations

from typing import Any

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np

from vsearch.backends import device as device_mod
from vsearch.backends.registry import ModelRegistry, get_registry
from vsearch.config import TEXT_PROMPT_TEMPLATE, Profile, get_profile

if TYPE_CHECKING:
    from PIL.Image import Image

logger = logging.getLogger(__name__)

#: Фіксована довжина текстової послідовності SigLIP.
TEXT_SEQ_LEN = 64


@dataclass(frozen=True)
class EmbedderInfo:
    """Те, що має потрапити в провенанс кожного результату (п.7)."""

    model_name: str
    repo_id: str
    revision: str
    dim: int
    max_num_patches: int
    device: str
    dtype: str


#: Завантажені ваги, спільні для всіх екземплярів: ключ — (модель, пристрій,
#: тип). Для інференсу ваги лише читаються, тож ділити їх безпечно, а не
#: ділити — дорого: кожен новий `Searcher` брав власну копію 1.6 ГБ.
#:
#: Знайдено на повному прогоні тестів: додавання двох файлів зі своїми
#: `Searcher()` довело памʼять прискорювача до 23.8 з 30.2 ГБ і дало 12
#: падінь, які виглядали як дефекти логіки. Це вже другий раз, коли
#: вичерпання памʼяті читалося як щось інше (ризик 14).
_LOADED: dict[tuple[str, str, str], tuple[Any, Any]] = {}


class Siglip2Embedder:
    """Обгортка над SigLIP 2 NaFlex.

    Модель завантажується лениво: `doctor`, перевірка ліцензій і валідація
    золотого набору не мають тягнути 1.6 ГБ у памʼять.
    """

    def __init__(
        self,
        profile: Profile | None = None,
        registry: ModelRegistry | None = None,
        device_spec: device_mod.DeviceSpec | None = None,
    ) -> None:
        self.profile = profile or get_profile()
        self.registry = registry or get_registry()
        self.device_spec = device_spec or device_mod.resolve()
        self._model = None
        self._processor = None

    # ── завантаження ────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModel, AutoProcessor

        name = self.profile.embed_model
        kwargs = self.registry.hf_kwargs(name)
        torch_device, torch_dtype = device_mod.torch_device_and_dtype(self.device_spec)

        key = (name, str(torch_device), str(torch_dtype))
        cached = _LOADED.get(key)
        if cached is not None:
            self._processor, self._model = cached
            return

        logger.info("завантаження %s на %s (%s)", name, torch_device, self.device_spec.dtype)
        self._processor = AutoProcessor.from_pretrained(**kwargs)
        self._model = AutoModel.from_pretrained(
            **kwargs,
            dtype=torch_dtype,
            attn_implementation="sdpa",
        )
        self._model.to(torch_device)
        self._model.eval()
        _LOADED[key] = (self._processor, self._model)

    @property
    def info(self) -> EmbedderInfo:
        entry = self.registry.entry(self.profile.embed_model)
        return EmbedderInfo(
            model_name=self.profile.embed_model,
            repo_id=entry.repo_id,
            revision=entry.revision,
            dim=self.profile.embed_dim,
            max_num_patches=self.profile.max_num_patches,
            device=self.device_spec.device,
            dtype=self.device_spec.dtype,
        )

    @property
    def dim(self) -> int:
        return self.profile.embed_dim

    @property
    def calibration(self) -> tuple[float, float]:
        """Власні параметри калібрування SigLIP: (logit_scale, logit_bias).

        SigLIP тренували сигмоїдним лосом, а не softmax по батчу, тому
        `sigmoid(scale * cos + bias)` — це справжня ймовірність того, що пара
        «зображення–текст» відповідна, поштучно й без огляду на решту батчу.
        Саме тому фасет може мати абсолютний поріг: у CLIP така величина була б
        осмисленою лише відносно інших кандидатів.
        """
        self._ensure_loaded()
        return (
            float(self._model.logit_scale.exp().item()),  # type: ignore[union-attr]
            float(self._model.logit_bias.item()),  # type: ignore[union-attr]
        )

    def probability(self, cosine):
        """Косинус → ймовірність відповідності за калібруванням моделі."""
        scale, bias = self.calibration
        return 1.0 / (1.0 + np.exp(-(np.asarray(cosine) * scale + bias)))

    # ── ембединги ───────────────────────────────────────────────────────────

    def embed_images(
        self,
        images: Sequence["Image"],
        *,
        batch_size: int = 16,
        max_num_patches: int | None = None,
    ) -> np.ndarray:
        """Зображення → матриця (N, dim) з одиничною нормою рядків.

        `max_num_patches` перекривається явно лише для замірів: у бойовому
        режимі він береться з профілю, бо має збігатися з тим, яким будувався
        індекс.
        """
        self._ensure_loaded()
        import torch

        patches = max_num_patches or self.profile.max_num_patches
        torch_device, _ = device_mod.torch_device_and_dtype(self.device_spec)
        chunks: list[np.ndarray] = []

        for start in range(0, len(images), batch_size):
            batch = list(images[start : start + batch_size])
            inputs = self._processor(  # type: ignore[misc]
                images=batch,
                max_num_patches=patches,
                return_tensors="pt",
            ).to(torch_device)
            with torch.no_grad():
                features = self._model.get_image_features(**inputs)  # type: ignore[union-attr]
            chunks.append(_l2_normalize(_as_matrix(features)))

        if not chunks:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack(chunks)

    def embed_texts(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 64,
        use_template: bool = True,
    ) -> np.ndarray:
        """Тексти → матриця (N, dim) з одиничною нормою рядків.

        `use_template` загортає текст у "this is a photo of {label}." — саме так
        формулювали підписи при навчанні, і для коротких запитів на кшталт «ніж»
        це помітно краще за голе слово. Для довгих описів шаблон вимикають.
        """
        self._ensure_loaded()
        import torch

        prepared = [self.prepare_text(t, use_template=use_template) for t in texts]
        torch_device, _ = device_mod.torch_device_and_dtype(self.device_spec)
        chunks: list[np.ndarray] = []

        for start in range(0, len(prepared), batch_size):
            batch = prepared[start : start + batch_size]
            inputs = self._processor(  # type: ignore[misc]
                text=batch,
                padding="max_length",
                max_length=TEXT_SEQ_LEN,
                truncation=True,
                return_tensors="pt",
            ).to(torch_device)
            with torch.no_grad():
                features = self._model.get_text_features(**inputs)  # type: ignore[union-attr]
            chunks.append(_l2_normalize(_as_matrix(features)))

        if not chunks:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack(chunks)

    @staticmethod
    def prepare_text(text: str, *, use_template: bool = True) -> str:
        """Нормалізація тексту під те, як тренували модель."""
        cleaned = " ".join(text.split()).lower()
        return TEXT_PROMPT_TEMPLATE.format(label=cleaned) if use_template else cleaned


def _as_matrix(features) -> np.ndarray:
    """Витягти матрицю (N, dim) з того, що повернула модель.

    transformers 4.x віддає тензор напряму, 5.x — обʼєкт виводу з pooler_output.
    Підтримуємо обидва: різниця версій не має протікати в решту коду.
    """
    if hasattr(features, "pooler_output"):
        features = features.pooler_output
    elif not hasattr(features, "shape"):
        features = features[0]
    return features.float().cpu().numpy()


def _l2_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return (matrix / np.maximum(norms, eps)).astype(np.float32)
