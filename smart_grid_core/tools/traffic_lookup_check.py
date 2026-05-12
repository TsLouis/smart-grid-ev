from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from ..assets import default_asset_registry
from ..runtime.default_regions import DEFAULT_STATION_ALIGNMENTS
from ..twins.traffic import PrecomputedTrafficTwin


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect deterministic TrafficTwin route-table values.")
    parser.add_argument("--root", default=".", help="EV_charging project root")
    parser.add_argument("--day", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--origin", type=int, default=0)
    parser.add_argument("--time", type=float, default=8.0, help="query time in hours")
    parser.add_argument(
        "--region",
        default=None,
        help="Region id to bind region-scoped assets (e.g. district_a). "
        "Default reads legacy paths under the project root.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    registry = default_asset_registry(Path(args.root))
    if args.region:
        registry = registry.for_region(args.region)
    station_nodes = [
        a.traffic_node
        for a in sorted(DEFAULT_STATION_ALIGNMENTS, key=lambda x: x.station_index)
    ]
    twin = PrecomputedTrafficTwin.from_registry(
        registry,
        day=args.day,
        station_nodes=station_nodes,
    )
    states = twin.station_route_states(origin_node=args.origin, query_time_h=args.time)

    if args.json:
        print(json.dumps([asdict(state) for state in states], ensure_ascii=False, indent=2))
    else:
        print(f"day: {args.day}")
        print(f"origin_node: {args.origin}")
        print(f"query_time_h: {args.time}")
        print(f"time_step: {states[0].time_step if states else 'n/a'}")
        for state in states:
            status = "reachable" if state.reachable else "unreachable"
            print(
                f"station={state.station_index} node={state.station_node} "
                f"time={state.travel_time_h:.6f}h energy={state.travel_energy_kwh:.6f}kWh {status}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
