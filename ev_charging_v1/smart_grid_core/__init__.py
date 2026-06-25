"""
Refactor foundation for the EV charging smart-grid project.

This package is intentionally additive. It defines the clean interfaces that
new code should use while the legacy scripts continue to run unchanged.
"""

from .assets import AssetRegistry, AssetSpec, default_asset_registry
from .charging import ChargeCurve, build_constant_power_curve
from .schemas import (
    ChargeDecision,
    ChargeEvent,
    DecisionRecord,
    ForecastResult,
    ScenarioContext,
    StationCandidate,
    VehicleState,
)

__all__ = [
    "AssetRegistry",
    "AssetSpec",
    "ChargeCurve",
    "ChargeDecision",
    "ChargeEvent",
    "DecisionRecord",
    "ForecastResult",
    "ScenarioContext",
    "StationCandidate",
    "VehicleState",
    "build_constant_power_curve",
    "default_asset_registry",
]
