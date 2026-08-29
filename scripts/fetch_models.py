#!/usr/bin/env python3
"""Завантаження ваг — ТІЛЬКИ на етапі збірки образу.

Це єдиний файл у проєкті, якому дозволено ходити в мережу. Середовище виконання
його не викликає й не містить (див. Dockerfile: скрипт живе у builder-шарі).

Режими:
    --update-lock   розвʼязати теги в commit SHA, порахувати sha256 → записати в маніфест
    --check         звірити контрольні суми вже завантаженого (мережа не потрібна)
    (без прапорців) завантажити за пінами з маніфесту

Розбіжність sha256 валить процес із ненульовим кодом, тобто валить збірку образу.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

MODELS_DIR = REPO_ROOT / "models"
MANIFEST = MODELS_DIR / "manifest.lock"


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def save_manifest(data: dict) -> None:
    MANIFEST.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def fetch_hf(name: str, spec: dict, *, update_lock: bool) -> list[str]:
    """Завантажити модель із HuggingFace Hub за закріпленою ревізією."""
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError:
        return [f"{name}: потрібен huggingface_hub (pip install huggingface_hub) для build-фази"]

    revision = spec.get("revision") or ""
    if not revision:
        if not update_lock:
            return [
                f"{name}: revision порожній. Спершу закріпіть ревізію:\n"
                f"    python scripts/fetch_models.py --update-lock --only {name}"
            ]
        # Розвʼязуємо поточний стан гілки в конкретний commit SHA один раз і назавжди.
        info = HfApi().model_info(spec["repo_id"])
        revision = info.sha
        spec["revision"] = revision
        print(f"  закріплено revision={revision}")

    target = MODELS_DIR / spec["local_dir"]
    patterns = spec.get("allow_patterns")
    print(f"  завантаження {spec['repo_id']}@{revision[:12]} → {target}")
    if patterns:
        print(f"  фільтр файлів: {patterns}")
    snapshot_download(
        repo_id=spec["repo_id"],
        revision=revision,
        local_dir=str(target),
        allow_patterns=patterns,
    )
    return []


def fetch_urls(name: str, spec: dict) -> list[str]:
    """Завантажити окремі файли за прямими посиланнями (OpenCV Zoo тощо)."""
    problems: list[str] = []
    target_dir = MODELS_DIR / spec["local_dir"]
    target_dir.mkdir(parents=True, exist_ok=True)
    for file_spec in spec.get("files", []):
        url = file_spec.get("url")
        if not url:
            problems.append(f"{name}/{file_spec['path']}: source=url, але url не вказано")
            continue
        target = target_dir / file_spec["path"]
        if target.exists():
            print(f"  вже є: {target.name}")
            continue
        print(f"  завантаження {url}")
        urllib.request.urlretrieve(url, target)  # noqa: S310 — лише build-фаза
    return problems


def record_checksums(spec: dict) -> None:
    """Записати sha256 усіх наявних файлів моделі в маніфест."""
    base = MODELS_DIR / spec["local_dir"]
    if not base.is_dir():
        return
    recorded = []
    for path in sorted(base.rglob("*")):
        if not path.is_file() or ".cache" in path.parts:
            continue
        recorded.append({"path": str(path.relative_to(base)), "sha256": sha256_of(path)})
    spec["files"] = recorded
    print(f"  записано контрольних сум: {len(recorded)}")


def check_only(names: list[str]) -> int:
    """Звірити контрольні суми без мережі. Придатне для CI та runtime-перевірки."""
    from vsearch.backends.registry import ModelRegistry

    registry = ModelRegistry(MODELS_DIR)
    failures = 0
    for name in names or registry.names:
        entry = registry.entry(name)
        if not registry.is_fetched(name):
            status = "пропущено (не завантажено)" if entry.optional else "ВІДСУТНЯ"
            print(f"[{'-' if entry.optional else '!'}] {name}: {status}")
            failures += 0 if entry.optional else 1
            continue
        problems = registry.verify_checksums(name)
        if problems:
            failures += 1
            print(f"[!] {name}: контрольні суми не збігаються")
            for problem in problems:
                print(f"      {problem}")
        else:
            print(f"[+] {name}: {len(entry.files)} файлів, sha256 збігається")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", default=[], help="обмежити перелік моделей")
    parser.add_argument("--update-lock", action="store_true", help="закріпити revision і sha256")
    parser.add_argument("--check", action="store_true", help="лише звірити суми, без мережі")
    parser.add_argument("--include-optional", action="store_true", help="тягнути й опційні моделі")
    args = parser.parse_args()

    if args.check:
        return 1 if check_only(args.only) else 0

    print("=" * 70)
    print("BUILD-ФАЗА: це єдиний момент, коли дозволено виходити в мережу.")
    print("Середовище виконання цього скрипта не містить (див. Dockerfile).")
    print("=" * 70)

    data = load_manifest()
    problems: list[str] = []
    for name, spec in data["models"].items():
        if args.only and name not in args.only:
            continue
        if spec.get("optional") and not args.include_optional and not args.only:
            print(f"[-] {name}: опційна, пропущено (--include-optional щоб узяти)")
            continue

        print(f"[*] {name} — {spec.get('role', '')}")
        if spec.get("source", "hf") == "url":
            problems += fetch_urls(name, spec)
        else:
            problems += fetch_hf(name, spec, update_lock=args.update_lock)

        if args.update_lock and not problems:
            record_checksums(spec)

    if args.update_lock:
        save_manifest(data)
        print(f"\nМаніфест оновлено: {MANIFEST}")

    if problems:
        print("\nПроблеми:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
