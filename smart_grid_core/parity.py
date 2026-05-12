from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .legacy_adapters import normalize_charge_events
from .schemas import ChargeEvent


# Legacy node→station mapping baked into recorded event files. Tied to
# the legacy dataset, NOT the new region/alignment table — do not consume
# from outside this module. New code should source station mapping from
# `runtime.default_regions.DEFAULT_STATION_ALIGNMENTS`.
_LEGACY_STATION_NODES: tuple[int, ...] = (7, 10, 22, 27, 12, 20, 14)
LEGACY_NODE_TO_STATION: Mapping[int, int] = {
    node: idx for idx, node in enumerate(_LEGACY_STATION_NODES)
}


@dataclass(frozen=True)
class LoadAggregationConfig:
    n_stations: int = 7
    t_simulation: int = 312
    trim_start: int = 12
    trim_end: int = 300
    delta_tau_h: float = 5 / 60


@dataclass(frozen=True)
class ParityReport:
    event_path: str
    baseline_path: str
    event_count: int
    converted_event_count: int
    baseline_shape: tuple[int, ...]
    replay_shape: tuple[int, ...]
    max_abs_error: float
    mean_abs_error: float
    total_baseline_energy_step_kw: float
    total_replay_energy_step_kw: float
    notes: list[str]


def load_legacy_events(path: str | Path) -> list[Mapping]:
    with Path(path).open("rb") as handle:
        events = pickle.load(handle)
    if not isinstance(events, list):
        raise TypeError(f"Expected a list of legacy events, got {type(events)!r}")
    return events


def load_baseline_array(path: str | Path):
    import numpy as np

    return np.load(path)


def aggregate_charge_events(
    events: Iterable[ChargeEvent],
    config: LoadAggregationConfig = LoadAggregationConfig(),
):
    import numpy as np

    full = np.zeros((config.n_stations, config.t_simulation), dtype=float)
    skipped = 0
    for event in events:
        station = event.station_index
        if station < 0 or station >= config.n_stations:
            skipped += 1
            continue
        for offset, power_kw in enumerate(event.power_sequence_kw):
            step = event.start_step + offset
            if 0 <= step < config.t_simulation:
                full[station, step] += float(power_kw)

    trimmed = full[:, config.trim_start : config.trim_end]
    return trimmed, {"skipped_events": skipped}


def replay_legacy_events(
    events: Sequence[Mapping],
    *,
    node_to_station: Mapping[int, int] | None = None,
    config: LoadAggregationConfig = LoadAggregationConfig(),
):
    canonical = normalize_charge_events(
        events,
        delta_tau_h=config.delta_tau_h,
        node_to_station=node_to_station or LEGACY_NODE_TO_STATION,
    )
    return aggregate_charge_events(canonical, config=config)


def compare_replay_to_baseline(
    *,
    event_path: str | Path,
    baseline_path: str | Path,
    node_to_station: Mapping[int, int] | None = None,
    config: LoadAggregationConfig = LoadAggregationConfig(),
) -> ParityReport:
    import numpy as np

    legacy_events = load_legacy_events(event_path)
    replay_load, meta = replay_legacy_events(
        legacy_events,
        node_to_station=node_to_station,
        config=config,
    )
    baseline = load_baseline_array(baseline_path)

    notes: list[str] = []
    if tuple(replay_load.shape) != tuple(baseline.shape):
        notes.append(f"shape mismatch: replay={replay_load.shape}, baseline={baseline.shape}")
        min_rows = min(replay_load.shape[0], baseline.shape[0])
        min_cols = min(replay_load.shape[1], baseline.shape[1])
        replay_cmp = replay_load[:min_rows, :min_cols]
        baseline_cmp = baseline[:min_rows, :min_cols]
    else:
        replay_cmp = replay_load
        baseline_cmp = baseline

    diff = replay_cmp - baseline_cmp
    if meta.get("skipped_events"):
        notes.append(f"skipped events without station mapping: {meta['skipped_events']}")

    return ParityReport(
        event_path=str(event_path),
        baseline_path=str(baseline_path),
        event_count=len(legacy_events),
        converted_event_count=len(legacy_events) - int(meta.get("skipped_events", 0)),
        baseline_shape=tuple(int(v) for v in baseline.shape),
        replay_shape=tuple(int(v) for v in replay_load.shape),
        max_abs_error=float(np.max(np.abs(diff))) if diff.size else 0.0,
        mean_abs_error=float(np.mean(np.abs(diff))) if diff.size else 0.0,
        total_baseline_energy_step_kw=float(np.sum(baseline_cmp)),
        total_replay_energy_step_kw=float(np.sum(replay_cmp)),
        notes=notes,
    )


def report_as_dict(report: ParityReport) -> dict:
    return asdict(report)
