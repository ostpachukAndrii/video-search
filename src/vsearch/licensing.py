"""Політика ліцензій (п.6).

Живе в пакеті, а не в скрипті: те саме правило перевіряють і CLI, і тести.
Дві копії списку розійшлися б при першому ж додаванні моделі.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

#: Ліцензії, під якими комерційне використання дозволене без застережень.
ALLOWED: frozenset[str] = frozenset(
    {"Apache-2.0", "MIT", "BSD-3-Clause", "BSD-2-Clause", "PostgreSQL", "CC-BY-4.0"}
)

#: Свідомо відхилені — явний список тримається, щоб вони не повернулися
#: непоміченими разом із «зручною» новою моделлю.
REJECTED: dict[str, str] = {
    "AGPL-3.0": "мережевий копілефт — непридатна для постачання замовнику",
    "AGPL-3.0-only": "мережевий копілефт — непридатна для постачання замовнику",
    "CC-BY-NC-4.0": "заборонено комерційне використання",
    "CC-BY-NC-SA-4.0": "заборонено комерційне використання",
    "ELv2": "обмежує надання як сервісу",
    "Qwen-Research": "лише дослідницьке використання",
    "non-commercial": "заборонено комерційне використання",
}

#: Стан, коли ліцензія формально відкрита, але походження ваг потребує
#: висновку юриста. Модель у такому стані має бути опційною й за фіче-флагом.
UNVERIFIED = "UNVERIFIED"


class Verdict(str, Enum):
    OK = "ok"
    REJECTED = "rejected"
    UNVERIFIED = "unverified"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class LicenseFinding:
    model: str
    license: str
    verdict: Verdict
    reason: str = ""
    optional: bool = False
    notes: str = ""

    @property
    def blocks_release(self) -> bool:
        return self.verdict in (Verdict.REJECTED, Verdict.UNKNOWN)


@dataclass
class AuditReport:
    findings: list[LicenseFinding] = field(default_factory=list)
    unpinned_revisions: list[str] = field(default_factory=list)

    @property
    def blocking(self) -> list[LicenseFinding]:
        return [f for f in self.findings if f.blocks_release]

    @property
    def unverified(self) -> list[LicenseFinding]:
        return [f for f in self.findings if f.verdict is Verdict.UNVERIFIED]

    def is_clean(self, *, strict: bool = False) -> bool:
        if self.blocking:
            return False
        if strict and (self.unverified or self.unpinned_revisions):
            return False
        return True


def classify(license_id: str) -> tuple[Verdict, str]:
    """Один рядок ліцензії → вердикт і причина."""
    if license_id in REJECTED:
        return Verdict.REJECTED, REJECTED[license_id]
    if license_id == UNVERIFIED:
        return Verdict.UNVERIFIED, "потрібен висновок юриста щодо походження ваг"
    if license_id in ALLOWED:
        return Verdict.OK, ""
    return Verdict.UNKNOWN, "невідома ліцензія — додайте до ALLOWED або відхиліть"


def audit(registry) -> AuditReport:
    """Перевірити весь маніфест."""
    report = AuditReport()
    for entry in registry:
        verdict, reason = classify(entry.license)
        report.findings.append(
            LicenseFinding(
                model=entry.name,
                license=entry.license,
                verdict=verdict,
                reason=reason,
                optional=entry.optional,
                notes=entry.notes,
            )
        )
        if not entry.revision_is_pinned:
            report.unpinned_revisions.append(entry.name)
    return report
