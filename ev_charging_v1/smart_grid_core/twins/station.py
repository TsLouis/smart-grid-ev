from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Literal, Mapping, Optional

from ..charging import ChargeCurve, build_constant_power_curve
from ..schemas import ChargeEvent, DecisionRecord
from .base import ConstraintViolation, TwinResult

PileType = Literal["fast", "slow"]


@dataclass(frozen=True)
class StationConfig:
    station_index: int
    node: int
    fast_piles: int = 120
    slow_piles: int = 120
    delta_tau_h: float = 5 / 60


@dataclass(frozen=True)
class ChargeRequest:
    vehicle_id: int
    arrival_time: float
    arrival_soc: float
    target_soc: float
    battery_capacity_kwh: float
    available_time_h: float
    efficiency: float
    pile_type: PileType
    power_kw: float
    decision: Optional[DecisionRecord] = None
    metadata: Mapping = field(default_factory=dict)


@dataclass
class QueuedRequest:
    request: ChargeRequest
    queued_at: float


class StationTwin:
    """White-box charging-station resource twin.

    The twin only applies already-decided requests. It does not forecast queue
    time, choose pile type, choose target SOC, or rank stations.
    """

    def __init__(self, config: StationConfig):
        self.config = config
        self.fast_occupied = 0
        self.slow_occupied = 0
        self.fast_queue: Deque[QueuedRequest] = deque()
        self.slow_queue: Deque[QueuedRequest] = deque()
        self.active: Dict[int, ChargeEvent] = {}
        self.completed: List[ChargeEvent] = []

    def snapshot(self) -> dict:
        return {
            "station_index": self.config.station_index,
            "node": self.config.node,
            "fast_piles": self.config.fast_piles,
            "slow_piles": self.config.slow_piles,
            "fast_occupied": self.fast_occupied,
            "slow_occupied": self.slow_occupied,
            "fast_queue_len": len(self.fast_queue),
            "slow_queue_len": len(self.slow_queue),
            "active_vehicle_ids": sorted(self.active),
            "completed_count": len(self.completed),
        }

    def validate_request(self, request: ChargeRequest) -> TwinResult:
        violations: List[ConstraintViolation] = []
        if request.pile_type not in ("fast", "slow"):
            violations.append(ConstraintViolation("pile_type", "pile_type must be 'fast' or 'slow'"))
        if request.battery_capacity_kwh <= 0:
            violations.append(ConstraintViolation("battery_capacity_kwh", "battery capacity must be positive"))
        if request.available_time_h < 0:
            violations.append(ConstraintViolation("available_time_h", "available time cannot be negative"))
        if request.power_kw < 0:
            violations.append(ConstraintViolation("power_kw", "power cannot be negative"))
        if not 0 <= request.arrival_soc <= 1:
            violations.append(ConstraintViolation("arrival_soc", "arrival SOC must be in [0, 1]"))
        if not 0 <= request.target_soc <= 1:
            violations.append(ConstraintViolation("target_soc", "target SOC must be in [0, 1]"))
        return TwinResult(
            accepted=len(violations) == 0,
            payload={"vehicle_id": request.vehicle_id},
            violations=violations,
        )

    def has_free_pile(self, pile_type: PileType) -> bool:
        if pile_type == "fast":
            return self.fast_occupied < self.config.fast_piles
        return self.slow_occupied < self.config.slow_piles

    def submit_request(self, request: ChargeRequest) -> TwinResult:
        validation = self.validate_request(request)
        if not validation.accepted:
            return validation

        if self.has_free_pile(request.pile_type):
            event = self._start_charge(request, start_time=request.arrival_time)
            return TwinResult(accepted=True, payload={"status": "started", "event": event}, violations=[])

        queue = self.fast_queue if request.pile_type == "fast" else self.slow_queue
        queue.append(QueuedRequest(request=request, queued_at=request.arrival_time))
        return TwinResult(
            accepted=True,
            payload={"status": "queued", "queue_position": len(queue), "vehicle_id": request.vehicle_id},
            violations=[],
        )

    def finish_vehicle(self, vehicle_id: int, finish_time: Optional[float] = None) -> TwinResult:
        event = self.active.pop(vehicle_id, None)
        if event is None:
            return TwinResult(
                accepted=False,
                payload={"vehicle_id": vehicle_id},
                violations=[ConstraintViolation("vehicle_id", "vehicle is not actively charging")],
            )

        self._release_pile(event.pile_type)
        self.completed.append(event)
        next_event = self._start_next_if_waiting(event.pile_type, finish_time or event.end_time)
        return TwinResult(
            accepted=True,
            payload={"status": "finished", "event": event, "next_event": next_event},
            violations=[],
        )

    def replay(self, events: Iterable[ChargeEvent]) -> None:
        """Load completed events into station history for deterministic replay."""

        for event in events:
            if event.station_index == self.config.station_index:
                self.completed.append(event)

    def _start_charge(self, request: ChargeRequest, start_time: float) -> ChargeEvent:
        self._occupy_pile(request.pile_type)
        curve = self._build_curve(request)
        start_step = int(start_time / self.config.delta_tau_h)
        end_time = start_time + curve.duration_h
        end_step = start_step + len(curve.power_sequence_kw)
        event = ChargeEvent(
            vehicle_id=request.vehicle_id,
            station_index=self.config.station_index,
            node=self.config.node,
            pile_type=request.pile_type,
            arrival_time=request.arrival_time,
            start_time=start_time,
            end_time=end_time,
            start_step=start_step,
            end_step=end_step,
            arrival_soc=request.arrival_soc,
            final_soc=curve.final_soc,
            energy_kwh=curve.energy_kwh,
            power_sequence_kw=curve.power_sequence_kw,
            queue_time_h=max(0.0, start_time - request.arrival_time),
            decision=request.decision,
            metadata=dict(request.metadata),
        )
        self.active[request.vehicle_id] = event
        return event

    def _build_curve(self, request: ChargeRequest) -> ChargeCurve:
        return build_constant_power_curve(
            initial_soc=request.arrival_soc,
            target_soc=request.target_soc,
            battery_capacity_kwh=request.battery_capacity_kwh,
            available_time_h=request.available_time_h,
            efficiency=request.efficiency,
            power_kw=request.power_kw,
            delta_tau_h=self.config.delta_tau_h,
        )

    def _start_next_if_waiting(self, pile_type: str, start_time: float) -> ChargeEvent | None:
        queue = self.fast_queue if pile_type == "fast" else self.slow_queue
        if not queue:
            return None
        queued = queue.popleft()
        return self._start_charge(queued.request, start_time=start_time)

    def _occupy_pile(self, pile_type: str) -> None:
        if pile_type == "fast":
            self.fast_occupied += 1
        else:
            self.slow_occupied += 1

    def _release_pile(self, pile_type: str) -> None:
        if pile_type == "fast":
            self.fast_occupied = max(0, self.fast_occupied - 1)
        else:
            self.slow_occupied = max(0, self.slow_occupied - 1)
