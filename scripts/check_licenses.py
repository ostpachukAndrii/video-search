#!/usr/bin/env python3
"""Інвентар ліцензій усіх ваг — доказ вимоги п.6.

Ліцензія моделі не менш обовʼязкова за її точність: система призначена для
комерційного використання, тож одна вага під CC-BY-NC або AGPL робить
непридатним увесь результат. Правило живе у vsearch.licensing, тут — лише CLI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vsearch.backends.registry import ModelRegistry  # noqa: E402
from vsearch.licensing import Verdict, audit  # noqa: E402

_LABEL = {
    Verdict.OK: "ок",
    Verdict.REJECTED: "ВІДХИЛЕНО",
    Verdict.UNVERIFIED: "НЕ ПІДТВЕРДЖЕНО",
    Verdict.UNKNOWN: "НЕВІДОМА",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="режим релізу: валити і на UNVERIFIED, і на незакріплених ревізіях",
    )
    args = parser.parse_args()

    report = audit(ModelRegistry())
    width = max(len(f.model) for f in report.findings)

    print(f"{'МОДЕЛЬ'.ljust(width)}  {'ЛІЦЕНЗІЯ'.ljust(14)}  СТАН")
    print("-" * (width + 60))
    for finding in report.findings:
        state = _LABEL[finding.verdict]
        if finding.reason:
            state += f" — {finding.reason}"
        elif finding.optional:
            state += " (опційна)"
        print(f"{finding.model.ljust(width)}  {finding.license.ljust(14)}  {state}")
        if finding.notes:
            print(f"{' ' * (width + 16)}  ↳ {finding.notes}")

    print()
    if report.unpinned_revisions:
        print(f"Не закріплені ревізії ({len(report.unpinned_revisions)}): "
              f"{', '.join(report.unpinned_revisions)}")
        print("  Відтворюваний образ потребує commit SHA:")
        print("    python scripts/fetch_models.py --update-lock")

    if not report.is_clean(strict=args.strict):
        print(f"\nПРОВАЛЕНО: блокувальних знахідок — {len(report.blocking)}", file=sys.stderr)
        return 1

    print(f"Придатно для комерційного використання "
          f"(непідтверджених: {len(report.unverified)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
