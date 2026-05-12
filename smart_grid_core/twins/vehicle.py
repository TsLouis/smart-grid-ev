from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping, Optional

from ..schemas import ChargeEvent, VehicleState
from .base import ConstraintViolation, TwinResult

VehicleMode = Literal["idle", "travelling", "dwelling", "queued", "charging", "inactive"]


@dataclass(frozen=True)
class VehicleConfig:
    vehicle_id: int
    initial_node: int
    initial_time: float
    initial_soc: float
    battery_capacity_kwh: float = 61.4
    target_soc: float = 0.95
    population: str = "unknown"
    metadata: Mapping = field(default_factory=dict)


@dataclass(frozen=True)
class TripInstruction:
    destination_node: int
    departure_time: float
    travel_time_h: float
    travel_energy_kwh: float
    dwell_time_h: float = 0.0
    metadata: Mapping = field(default_factory=dict)


@dataclass(frozen=True)
class DwellInstruction:
    until_time: float
    metadata: Mapping = field(default_factory=dict)


@dataclass(frozen=True)
class VehicleSnapshot:
    vehicle_id: int
    current_node: int
    current_time: float
    soc: float
    battery_capacity_kwh: float
    target_soc: float
    mode: VehicleMode
    population: str
    destination_node: Optional[int] = None
    dwell_until: Optional[float] = None
    metadata: Mapping = field(default_factory=dict)


class VehicleTwin:
    """White-box vehicle state twin.

    The twin applies already-decided trip, dwell, queue, and charge-completion
    events. It does not sample destinations, predict dwell time, decide charge
    demand, choose target SOC, or tune user preference parameters.
    """

    def __init__(self, config: VehicleConfig):
        self.vehicle_id = config.vehicle_id
        self.current_node = config.initial_node
        self.current_time = float(config.initial_time)
        self.soc = self._clamp_soc(config.initial_soc)
        self.battery_capacity_kwh = float(config.battery_capacity_kwh)
        self.target_soc = self._clamp_soc(config.target_soc)
        self.population = config.population
        self.mode: VehicleMode = "idle"
        self.destination_node: Optional[int] = None
        self.dwell_until: Optional[float] = None
        self.metadata = dict(config.metadata)
        self.history: list[dict] = []
        self._validate_initial()

    def state(self) -> VehicleState:
        return VehicleState(
            vehicle_id=self.vehicle_id,
            current_node=self.current_node,
            current_time=self.current_time,
            soc=self.soc,
            battery_capacity_kwh=self.battery_capacity_kwh,
            target_soc=self.target_soc,
            population=self.population,
            metadata={"mode": self.mode, **self.metadata},
        )

    def snapshot(self) -> VehicleSnapshot:
        return VehicleSnapshot(
            vehicle_id=self.vehicle_id,
            current_node=self.current_node,
            current_time=self.current_time,
            soc=self.soc,
            battery_capacity_kwh=self.battery_capacity_kwh,
            target_soc=self.target_soc,
            mode=self.mode,
            population=self.population,
            destination_node=self.destination_node,
            dwell_until=self.dwell_until,
            metadata=dict(self.metadata),
        )

    def apply_trip(self, instruction: TripInstruction) -> TwinResult:
        validation = self.validate_trip(instruction)
        if not validation.accepted:
            return validation

        arrival_time = instruction.departure_time + instruction.travel_time_h
        energy_soc = instruction.travel_energy_kwh / self.battery_capacity_kwh
        self.current_time = arrival_time
        self.current_node = instruction.destination_node
        self.soc = self._clamp_soc(self.soc - energy_soc)
        self.destination_node = instruction.destination_node
        self.mode = "dwelling" if instruction.dwell_time_h > 0 else "idle"
        self.dwell_until = arrival_time + instruction.dwell_time_h if instruction.dwell_time_h > 0 else None
        self._record(
            "trip",
            {
                "destination_node": instruction.destination_node,
                "departure_time": instruction.departure_time,
                "arrival_time": arrival_time,
                "travel_time_h": instruction.travel_time_h,
                "travel_energy_kwh": instruction.travel_energy_kwh,
                "dwell_until": self.dwell_until,
            },
        )
        return TwinResult(accepted=True, payload={"state": self.snapshot()}, violations=[])

    def begin_dwell(self, instruction: DwellInstruction) -> TwinResult:
        if instruction.until_time < self.current_time:
            return TwinResult(
                accepted=False,
                payload={"vehicle_id": self.vehicle_id},
                violations=[ConstraintViolation("until_time", "dwell end time cannot be before current time")],
            )
        self.mode = "dwelling"
        self.dwell_until = float(instruction.until_time)
        self._record("begin_dwell", {"until_time": self.dwell_until})
        return TwinResult(accepted=True, payload={"state": self.snapshot()}, violations=[])

    def end_dwell(self, current_time: Optional[float] = None) -> TwinResult:
        if current_time is None:
            current_time = self.dwell_until if self.dwell_until is not None else self.current_time
        if current_time < self.current_time:
            return TwinResult(
                accepted=False,
                payload={"vehicle_id": self.vehicle_id},
                violations=[ConstraintViolation("current_time", "cannot move vehicle time backwards")],
            )
        self.current_time = float(current_time)
        self.dwell_until = None
        self.mode = "idle"
        self._record("end_dwell", {"current_time": self.current_time})
        return TwinResult(accepted=True, payload={"state": self.snapshot()}, violations=[])

    def mark_queued(self, *, station_index: int, current_time: float) -> TwinResult:
        if current_time < self.current_time:
            return TwinResult(
                accepted=False,
                payload={"vehicle_id": self.vehicle_id},
                violations=[ConstraintViolation("current_time", "cannot move vehicle time backwards")],
            )
        self.current_time = float(current_time)
        self.mode = "queued"
        self._record("queued", {"station_index": station_index, "current_time": current_time})
        return TwinResult(accepted=True, payload={"state": self.snapshot()}, violations=[])

    def mark_charging(self, *, station_node: int, start_time: float) -> TwinResult:
        if start_time < self.current_time:
            return TwinResult(
                accepted=False,
                payload={"vehicle_id": self.vehicle_id},
                violations=[ConstraintViolation("start_time", "cannot move vehicle time backwards")],
            )
        self.current_node = station_node
        self.current_time = float(start_time)
        self.mode = "charging"
        self._record("charging", {"station_node": station_node, "start_time": start_time})
        return TwinResult(accepted=True, payload={"state": self.snapshot()}, violations=[])

    def apply_charge_event(self, event: ChargeEvent) -> TwinResult:
        if event.vehicle_id != self.vehicle_id:
            return TwinResult(
                accepted=False,
                payload={"vehicle_id": self.vehicle_id, "event_vehicle_id": event.vehicle_id},
                violations=[ConstraintViolation("vehicle_id", "charge event belongs to another vehicle")],
            )
        if event.end_time < self.current_time:
            return TwinResult(
                accepted=False,
                payload={"vehicle_id": self.vehicle_id},
                violations=[ConstraintViolation("end_time", "cannot move vehicle time backwards")],
            )
        self.current_node = event.node
        self.current_time = float(event.end_time)
        self.soc = self._clamp_soc(event.final_soc)
        self.mode = "idle"
        self.dwell_until = None
        self._record(
            "charge_complete",
            {
                "station_index": event.station_index,
                "node": event.node,
                "end_time": event.end_time,
                "final_soc": event.final_soc,
                "energy_kwh": event.energy_kwh,
            },
        )
        return TwinResult(accepted=True, payload={"state": self.snapshot()}, violations=[])

    def deactivate(self, *, current_time: Optional[float] = None, reason: str = "") -> TwinResult:
        if current_time is not None:
            if current_time < self.current_time:
                return TwinResult(
                    accepted=False,
                    payload={"vehicle_id": self.vehicle_id},
                    violations=[ConstraintViolation("current_time", "cannot move vehicle time backwards")],
                )
            self.current_time = float(current_time)
        self.mode = "inactive"
        self._record("inactive", {"current_time": self.current_time, "reason": reason})
        return TwinResult(accepted=True, payload={"state": self.snapshot()}, violations=[])

    def validate_trip(self, instruction: TripInstruction) -> TwinResult:
        violations: list[ConstraintViolation] = []
        if instruction.departure_time < self.current_time:
            violations.append(ConstraintViolation("departure_time", "departure cannot be before current time"))
        if instruction.travel_time_h < 0:
            violations.append(ConstraintViolation("travel_time_h", "travel time cannot be negative"))
        if instruction.travel_energy_kwh < 0:
            violations.append(ConstraintViolation("travel_energy_kwh", "travel energy cannot be negative"))
        if instruction.dwell_time_h < 0:
            violations.append(ConstraintViolation("dwell_time_h", "dwell time cannot be negative"))
        if self.battery_capacity_kwh <= 0:
            violations.append(ConstraintViolation("battery_capacity_kwh", "battery capacity must be positive"))
        return TwinResult(
            accepted=len(violations) == 0,
            payload={"vehicle_id": self.vehicle_id},
            violations=violations,
        )

    def _validate_initial(self) -> None:
        if self.battery_capacity_kwh <= 0:
            raise ValueError("battery_capacity_kwh must be positive")
        if self.current_time < 0:
            raise ValueError("initial_time cannot be negative")

    def _record(self, event_type: str, payload: Mapping) -> None:
        self.history.append({"event_type": event_type, "payload": dict(payload)})

    @staticmethod
    def _clamp_soc(soc: float) -> float:
        return min(1.0, max(0.0, float(soc)))
