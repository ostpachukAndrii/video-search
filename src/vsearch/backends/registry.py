"""Реєстр моделей: єдиний дозволений спосіб дістати ваги.

Правило проєкту (п.9): у середовищі виконання мережі немає взагалі. Ваги
приїжджають на етапі збірки образу за `models/manifest.lock` і далі читаються
лише з диска.

Тому цей модуль приймає **тільки логічне імʼя моделі** і повертає локальний
шлях. Він свідомо не вміє приймати HuggingFace repo_id: виклик на кшталт
`from_pretrained("microsoft/Florence-2-base")`, залишений будь-де в коді,
мовчки працює на ноутбуці розробника і падає лише в ізольованому контурі —
тобто виявляється найдорожчим із можливих способів.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODELS_DIR = Path(os.environ.get("VSEARCH_MODELS_DIR", REPO_ROOT / "models"))
MANIFEST_NAME = "manifest.lock"

#: Змінні, які змушують бібліотеки HuggingFace навіть не намагатися вийти в мережу.
OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
}


class ModelNotFetched(RuntimeError):
    """Модель є в маніфесті, але її файлів немає на диску."""


class ManifestError(RuntimeError):
    """Маніфест відсутній, зіпсований або неповний."""


@dataclass(frozen=True)
class ModelFile:
    path: str
    sha256: str | None
    #: Заповнюється лише для source="url" (наприклад, ваги з OpenCV Zoo на GitHub).
    url: str | None = None

    @property
    def is_pinned(self) -> bool:
        return bool(self.sha256)


@dataclass(frozen=True)
class ModelEntry:
    """Один запис маніфесту — усе, що треба, щоб відтворити ваги побайтово."""

    name: str
    role: str
    #: "hf" — HuggingFace Hub; "url" — прямі посилання на файли (OpenCV Zoo тощо).
    source: str
    repo_id: str
    revision: str  # ЗАВЖДИ commit SHA, ніколи тег і ніколи "main"
    license: str
    license_url: str
    local_dir: str
    files: tuple[ModelFile, ...]
    optional: bool = False
    notes: str = ""

    @property
    def revision_is_pinned(self) -> bool:
        """Тег можна перезаписати, commit SHA — ні. Приймаємо лише друге."""
        return len(self.revision) == 40 and all(c in "0123456789abcdef" for c in self.revision)


def _load_manifest(models_dir: Path) -> dict[str, Any]:
    manifest_path = models_dir / MANIFEST_NAME
    if not manifest_path.exists():
        raise ManifestError(f"Немає {manifest_path}. Маніфест — частина репозиторію, не артефакт.")
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError(f"Зіпсований {manifest_path}: {exc}") from exc


class ModelRegistry:
    """Читає маніфест і видає локальні шляхи. Мережею не користується ніколи."""

    def __init__(self, models_dir: Path | str | None = None) -> None:
        self.models_dir = Path(models_dir) if models_dir else DEFAULT_MODELS_DIR
        raw = _load_manifest(self.models_dir)
        self._entries: dict[str, ModelEntry] = {}
        for name, spec in raw.get("models", {}).items():
            missing = {"repo_id", "revision", "license", "local_dir"} - spec.keys()
            if missing:
                raise ManifestError(f"Модель {name!r}: у маніфесті бракує полів {sorted(missing)}")
            self._entries[name] = ModelEntry(
                name=name,
                role=spec.get("role", ""),
                source=spec.get("source", "hf"),
                repo_id=spec["repo_id"],
                revision=spec["revision"],
                license=spec["license"],
                license_url=spec.get("license_url", ""),
                local_dir=spec["local_dir"],
                files=tuple(
                    ModelFile(path=f["path"], sha256=f.get("sha256"), url=f.get("url"))
                    for f in spec.get("files", [])
                ),
                optional=spec.get("optional", False),
                notes=spec.get("notes", ""),
            )

    def __iter__(self) -> Iterator[ModelEntry]:
        return iter(self._entries.values())

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    @property
    def names(self) -> list[str]:
        return sorted(self._entries)

    def entry(self, name: str) -> ModelEntry:
        try:
            return self._entries[name]
        except KeyError:
            raise KeyError(
                f"Модель {name!r} не описана в {self.models_dir / MANIFEST_NAME}. "
                f"Відомі: {', '.join(self.names)}"
            ) from None

    def local_path(self, name: str, *, require_present: bool = True) -> Path:
        """Локальна тека моделі. Єдиний спосіб отримати шлях до ваг."""
        entry = self.entry(name)
        path = self.models_dir / entry.local_dir
        if require_present and not self.is_fetched(name):
            raise ModelNotFetched(
                f"Ваг {name!r} немає в {path}.\n"
                f"Вони завантажуються ТІЛЬКИ на етапі збірки образу:\n"
                f"    python scripts/fetch_models.py --only {name}\n"
                f"У середовищі виконання мережі немає — це очікувана поведінка, не помилка."
            )
        return path

    def is_fetched(self, name: str) -> bool:
        """Чи всі файли моделі присутні на диску."""
        entry = self.entry(name)
        base = self.models_dir / entry.local_dir
        if not entry.files:
            return base.is_dir() and any(base.iterdir())
        return all((base / f.path).exists() for f in entry.files)

    def fetched_names(self) -> list[str]:
        return [n for n in self.names if self.is_fetched(n)]

    def verify_checksums(self, name: str) -> list[str]:
        """Звірити sha256 усіх файлів. Повертає перелік розбіжностей (порожній = все добре)."""
        entry = self.entry(name)
        base = self.models_dir / entry.local_dir
        problems: list[str] = []
        for f in entry.files:
            target = base / f.path
            if not target.exists():
                problems.append(f"{f.path}: файлу немає")
                continue
            if not f.is_pinned:
                problems.append(f"{f.path}: у маніфесті не записано sha256")
                continue
            actual = sha256_of(target)
            if actual != f.sha256:
                problems.append(f"{f.path}: sha256 {actual[:12]}… ≠ очікуваний {f.sha256[:12]}…")
        return problems

    def hf_kwargs(self, name: str) -> dict[str, Any]:
        """Аргументи для `from_pretrained`, які гарантовано не йдуть у мережу.

        Використовувати ТІЛЬКИ так:
            model = AutoModel.from_pretrained(**registry.hf_kwargs("siglip2"))
        """
        return {
            "pretrained_model_name_or_path": str(self.local_path(name)),
            "local_files_only": True,
        }


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def enforce_offline_env() -> None:
    """Виставити офлайн-змінні. Викликається на імпорті пакета vsearch."""
    for key, value in OFFLINE_ENV.items():
        os.environ.setdefault(key, value)
    os.environ.setdefault("TORCH_HOME", str(DEFAULT_MODELS_DIR / "torch"))


_registry: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    """Ліниво створений синглтон — маніфест читається один раз за процес."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
