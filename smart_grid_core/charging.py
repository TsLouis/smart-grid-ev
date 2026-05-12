from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ChargeCurve:
    final_soc: float
    energy_kwh: float
    duration_h: float
    power_sequence_kw: List[float]


def clamp_soc(soc: float) -> float:
    return min(1.0, max(0.0, float(soc)))


def build_constant_power_curve(
    *,
    initial_soc: float,
    target_soc: float,
    battery_capacity_kwh: float,
    available_time_h: float,
    efficiency: float,
    power_kw: float,
    delta_tau_h: float = 5 / 60,
) -> ChargeCurve:
    """Build a deterministic constant-power charging curve.

    This function does not decide when to charge, which station to use, or what
    target SOC should be. Those are agent responsibilities. The function only
    converts already-approved parameters into a white-box power sequence.
    """

    initial_soc = clamp_soc(initial_soc)
    target_soc = clamp_soc(target_soc)
    if target_soc <= initial_soc or battery_capacity_kwh <= 0 or available_time_h <= 0 or power_kw <= 0:
        return ChargeCurve(initial_soc, 0.0, 0.0, [])

    efficiency = max(0.0, float(efficiency))
    if efficiency <= 0:
        return ChargeCurve(initial_soc, 0.0, 0.0, [])

    needed_kwh = (target_soc - initial_soc) * battery_capacity_kwh
    deliverable_kwh = power_kw * efficiency * available_time_h
    energy_kwh = min(needed_kwh, deliverable_kwh)
    duration_h = energy_kwh / (power_kw * efficiency)
    final_soc = clamp_soc(initial_soc + energy_kwh / battery_capacity_kwh)

    sequence: List[float] = []
    remaining_h = duration_h
    while remaining_h > 1e-12:
        step_h = min(delta_tau_h, remaining_h)
        sequence.append(float(power_kw))
        remaining_h -= step_h

    return ChargeCurve(
        final_soc=final_soc,
        energy_kwh=energy_kwh,
        duration_h=duration_h,
        power_sequence_kw=sequence,
    )


def build_power_sequence_from_legacy(
    power_sequence_kw: list[float] | None,
    *,
    fallback_power_kw: float,
    steps: int,
) -> list[float]:
    if power_sequence_kw is not None:
        return [float(v) for v in power_sequence_kw]
    if steps <= 0 or fallback_power_kw <= 0:
        return []
    return [float(fallback_power_kw) for _ in range(steps)]
