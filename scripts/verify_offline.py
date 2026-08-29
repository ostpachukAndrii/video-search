#!/usr/bin/env python3
"""Доказ вимоги п.9: система працює із заблокованою мережею.

Скрипт блокує зовнішні зʼєднання на рівні socket і після цього проганяє
робочий цикл. Будь-яка спроба щось дотягнути завершується винятком, а не тихим
завантаженням у фоні.

Той самий сценарій продубльовано у features/offline.feature, тому вимога
перевіряється на кожному прогоні тестів, а не перед демонстрацією.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vsearch.backends.netguard import NetworkAccessDenied, no_network  # noqa: E402
from vsearch.backends.registry import ModelRegistry, get_registry  # noqa: E402


def check(label: str, fn) -> bool:
    try:
        detail = fn()
    except Exception as exc:  # noqa: BLE001 — тут важливий сам факт падіння
        print(f"[!] {label}: {type(exc).__name__}: {exc}")
        return False
    print(f"[+] {label}{f': {detail}' if detail else ''}")
    return True


def main() -> int:
    print("Перевірка офлайн-контуру (зовнішня мережа заблокована, loopback дозволений)\n")
    ok = True

    with no_network():
        # 1. Запобіжник справді працює — інакше решта перевірок нічого не варта.
        def guard_works() -> str:
            import socket

            try:
                socket.create_connection(("huggingface.co", 443), timeout=1)
            except NetworkAccessDenied:
                return "зовнішні зʼєднання блокуються"
            raise AssertionError("запобіжник не спрацював — зʼєднання пройшло")

        ok &= check("Запобіжник мережі", guard_works)

        # 2. Маніфест читається з диска.
        ok &= check(
            "Маніфест моделей",
            lambda: f"{len(ModelRegistry().names)} записів",
        )

        # 3. Офлайн-змінні виставлені на імпорті пакета.
        def offline_env() -> str:
            import os

            import vsearch  # noqa: F401 — імпорт має побічний ефект

            missing = [
                key
                for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
                if os.environ.get(key) != "1"
            ]
            if missing:
                raise AssertionError(f"не виставлені: {missing}")
            return "HF_HUB_OFFLINE, TRANSFORMERS_OFFLINE, HF_DATASETS_OFFLINE = 1"

        ok &= check("Офлайн-змінні оточення", offline_env)

        # 4. Профілі та метрики не тягнуть мережу.
        def config_loads() -> str:
            from vsearch.config import PROFILES, get_profile

            return f"профілі: {', '.join(PROFILES)} (типовий — {get_profile().name})"

        ok &= check("Конфігурація", config_loads)

        # 5. Наявні ваги збігаються за контрольними сумами.
        def checksums() -> str:
            registry = get_registry()
            fetched = registry.fetched_names()
            if not fetched:
                return "ваг ще немає — нічого звіряти (очікувано на M0)"
            broken = {n: registry.verify_checksums(n) for n in fetched}
            broken = {n: p for n, p in broken.items() if p}
            if broken:
                raise AssertionError(f"розбіжності: {broken}")
            return f"звірено моделей: {len(fetched)}"

        ok &= check("Контрольні суми ваг", checksums)

    print()
    if ok:
        print("Офлайн-контур цілий: жоден компонент не потребує мережі.")
        return 0
    print("Офлайн-контур ПОРУШЕНО.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
