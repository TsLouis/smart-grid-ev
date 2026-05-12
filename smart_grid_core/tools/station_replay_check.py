from __future__ import annotations

import argparse

from ..legacy_adapters import normalize_charge_events
from ..parity import LEGACY_NODE_TO_STATION, load_legacy_events  # legacy data context
from ..twins.station import StationConfig, StationTwin


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay canonical events into StationTwin histories.")
    parser.add_argument("event_path", help="Path to a legacy charge_events_*.pkl file")
    parser.add_argument("--station", type=int, default=0, help="station index to inspect")
    parser.add_argument("--node", type=int, default=None, help="station node; defaults from legacy mapping")
    args = parser.parse_args()

    events = load_legacy_events(args.event_path)
    canonical = normalize_charge_events(events, node_to_station=LEGACY_NODE_TO_STATION)
    node = args.node
    if node is None:
        inverse = {station: station_node for station_node, station in LEGACY_NODE_TO_STATION.items()}
        node = inverse.get(args.station, -1)

    station = StationTwin(StationConfig(station_index=args.station, node=node))
    station.replay(canonical)
    snapshot = station.snapshot()
    print(f"station_index: {snapshot['station_index']}")
    print(f"node: {snapshot['node']}")
    print(f"completed_count: {snapshot['completed_count']}")
    print(f"active_vehicle_ids: {snapshot['active_vehicle_ids']}")
    print(f"fast_queue_len: {snapshot['fast_queue_len']}")
    print(f"slow_queue_len: {snapshot['slow_queue_len']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
