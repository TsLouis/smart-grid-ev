from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Mapping


@dataclass(frozen=True)
class TwinAction:
    """Validated command candidate for the twin layer."""

    action_type: str
    payload: Mapping[str, Any]
    proposed_by: str


@dataclass(frozen=True)
class ConstraintViolation:
    field: str
    message: str
    severity: str = "error"


@dataclass(frozen=True)
class TwinResult:
    accepted: bool
    payload: Mapping[str, Any]
    violations: List[ConstraintViolation] = field(default_factory=list)


class TwinStateValidator:
    """Shared guardrail: agents may propose, twins decide what is executable."""

    def validate_action(self, action: TwinAction) -> TwinResult:
        violations: List[ConstraintViolation] = []
        if not action.action_type:
            violations.append(ConstraintViolation("action_type", "action type is required"))
        return TwinResult(accepted=len(violations) == 0, payload=dict(action.payload), violations=violations)
