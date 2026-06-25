from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any, Iterable, Mapping, Sequence

from ..schemas import ChargeEvent


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _infer_power_sequence(event: Mapping[str, Any], delta_tau_h: float) -> list[float]:
    if "power_sequence" in event and event["power_sequence"] is not None:
        return [float(v) for v in event["power_sequence"]]

    power = _as_float(event.get("power", event.get("const_power", 0.0)))
    if power <= 0:
        return []

    capacity = _as_float(event.get("capacity", event.get("battery_capacity", 61.4)), 61.4)
    arrival_soc = _as_float(event.get("arrival_soc", 0.0))
    final_soc = _as_float(event.get("target_soc", event.get("final_soc", arrival_soc)), arrival_soc)
    energy_kwh = max(0.0, capacity * (final_soc - arrival_soc))
    if energy_kwh <= 0:
        return []

    duration_h = energy_kwh / power
    steps = max(1, int(math.ceil(duration_h / delta_tau_h)))
    return [power for _ in range(steps)]


def _infer_steps(event: Mapping[str, Any], power_sequence: Sequence[float], delta_tau_h: float) -> tuple[int, int]:
    if "start_step" in event:
        start_step = int(event["start_step"])
        end_step = int(event.get("end_step", start_step + len(power_sequence)))
        return start_step, end_step

    end_step = int(event.get("end_step", event.get("time_step", 0)))
    if power_sequence:
        start_step = max(0, end_step - len(power_sequence))
    else:
        duration_h = _as_float(event.get("duration_h", 0.0))
        start_step = max(0, end_step - int(math.ceil(duration_h / delta_tau_h)))
    return start_step, end_step


def normalize_charge_event(
    event: Mapping[str, Any],
    *,
    delta_tau_h: float = 5 / 60,
    node_to_station: Mapping[int, int] | None = None,
    default_vehicle_id: int = -1,
) -> ChargeEvent:
    """Convert a legacy charging event dict into the canonical schema.

    The legacy project has at least two event shapes:
    - scalar-power events: `time_step`, `node`, `arrival_soc`, `target_soc`,
      `capacity`, `power`;
    - sequence events: `start_step`, `end_step`, `power_sequence`.

    This adapter preserves what exists and infers only the fields required for
    canonical replay. It should be used for migration/parity checks, not as a
    substitute for the future event generator.
    """

    power_sequence = _infer_power_sequence(event, delta_tau_h)
    start_step, end_step = _infer_steps(event, power_sequence, delta_tau_h)

    node = int(event.get("node", -1))
    station_index = int(event.get("station_index", event.get("best_cs", -1)))
    if station_index < 0 and node_to_station is not None:
        station_index = int(node_to_station.get(node, -1))

    arrival_soc = _as_float(event.get("arrival_soc", 0.0))
    final_soc = _as_float(event.get("target_soc", event.get("final_soc", arrival_soc)), arrival_soc)
    capacity = _as_float(event.get("capacity", event.get("battery_capacity", 61.4)), 61.4)
    energy_kwh = _as_float(event.get("energy_kwh", event.get("charged_energy", 0.0)))
    if energy_kwh <= 0:
        energy_kwh = max(0.0, capacity * (final_soc - arrival_soc))

    start_time = _as_float(event.get("start_time", start_step * delta_tau_h))
    end_time = _as_float(event.get("end_time", end_step * delta_tau_h))
    arrival_time = _as_float(event.get("arrival_time", start_time))
    queue_time = _as_float(event.get("queue_time_h", max(0.0, start_time - arrival_time)))

    return ChargeEvent(
        vehicle_id=int(event.get("vehicle_id", event.get("veh_id", default_vehicle_id))),
        station_index=station_index,
        node=node,
        pile_type=str(event.get("pile_type", "unknown")),
        arrival_time=arrival_time,
        start_time=start_time,
        end_time=end_time,
        start_step=start_step,
        end_step=end_step,
        arrival_soc=arrival_soc,
        final_soc=final_soc,
        energy_kwh=energy_kwh,
        power_sequence_kw=list(power_sequence),
        queue_time_h=queue_time,
        metadata={"legacy_event": dict(event)},
    )


def normalize_charge_events(
    events: Iterable[Mapping[str, Any]],
    *,
    delta_tau_h: float = 5 / 60,
    node_to_station: Mapping[int, int] | None = None,
) -> list[ChargeEvent]:
    return [
        normalize_charge_event(
            event,
            delta_tau_h=delta_tau_h,
            node_to_station=node_to_station,
            default_vehicle_id=index,
        )
        for index, event in enumerate(events)
    ]


class LegacyEventAdapter:
    def __init__(self, delta_tau_h: float = 5 / 60, node_to_station: Mapping[int, int] | None = None):
        self.delta_tau_h = delta_tau_h
        self.node_to_station = node_to_station

    def normalize(self, events: Iterable[Mapping[str, Any]]) -> list[ChargeEvent]:
        return normalize_charge_events(
            events,
            delta_tau_h=self.delta_tau_h,
            node_to_station=self.node_to_station,
        )

    def normalize_as_dicts(self, events: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [asdict(event) for event in self.normalize(events)]
