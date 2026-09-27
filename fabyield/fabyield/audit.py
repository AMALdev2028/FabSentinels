"""Audit trail: every calculation step and every modelling assumption, in order."""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

log = logging.getLogger("fabyield")


@dataclass(frozen=True)
class AuditEntry:
    step: str
    formula: str
    inputs: dict[str, Any]
    result: Any
    note: str = ""


@dataclass(frozen=True)
class Assumption:
    text: str
    source: str


@dataclass
class AuditTrail:
    entries: list[AuditEntry] = field(default_factory=list)
    assumptions: list[Assumption] = field(default_factory=list)

    def record(self, step: str, formula: str, inputs: dict[str, Any], result: Any, note: str = "") -> Any:
        self.entries.append(AuditEntry(step, formula, inputs, result, note))
        log.debug("%s: %s | inputs=%s -> %s", step, formula, inputs, result)
        return result

    def assume(self, text: str, source: str) -> None:
        if any(a.text == text for a in self.assumptions):
            return
        self.assumptions.append(Assumption(text, source))
        log.info("ASSUMPTION: %s [source: %s]", text, source)

    def to_dict(self) -> dict[str, Any]:
        return {"assumptions": [asdict(a) for a in self.assumptions],
                "calculation_steps": [asdict(e) for e in self.entries]}
