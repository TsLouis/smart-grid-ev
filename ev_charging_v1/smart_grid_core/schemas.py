from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional


@dataclass(frozen=True)
class ScenarioContext:
    day: int
    label: str
    time_step_minutes: int = 5
    tags: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VehicleState:
    vehicle_id: int
    current_node: int
    current_time: float
    soc: float
    battery_capacity_kwh: float
    target_soc: float
    population: str = "unknown"
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StationCandidate:
    station_index: int
    node: int
    travel_time_h: float
    travel_energy_kwh: float
    queue_time_h: float
    price: float
    grid_risk: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ForecastResult:
    source: str
    horizon_steps: int
    values: Any
    confidence: Optional[float] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChargeDecision:
    should_charge: bool
    target_soc: float
    mode_preference: str
    candidate_station_indices: List[int]
    weights: Mapping[str, float]
    reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DecisionRecord:
    agent: str
    observation: Mapping[str, Any]
    decision: Mapping[str, Any]
    accepted: bool
    validator_notes: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ChargeEvent:
    vehicle_id: int
    station_index: int
    node: int
    pile_type: str
    arrival_time: float
    start_time: float
    end_time: float
    start_step: int
    end_step: int
    arrival_soc: float
    final_soc: float
    energy_kwh: float
    power_sequence_kw: List[float]
    queue_time_h: float
    decision: Optional[DecisionRecord] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
