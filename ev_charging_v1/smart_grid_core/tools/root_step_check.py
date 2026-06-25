from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ..assets import default_asset_registry
from ..orchestrator import InteractionOrchestrator
from ..runtime.default_regions import build_default_city
from ..runtime.default_root import build_default_root_agent


_DEFAULT_OBSERVATION = {
    "current_node": 0,
    "current_time": 8.0,
    "soc": 0.35,
    "battery_capacity_kwh": 61.4,
    "next_trip_energy_kwh": 12.0,
    "reachable_stations": [
        {"station_index": 0, "travel_time_h": 0.10, "travel_energy_kwh": 1.5},
        {"station_index": 2, "travel_time_h": 0.18, "travel_energy_kwh": 2.4},
        {"station_index": 4, "travel_time_h": 0.22, "travel_energy_kwh": 3.0},
    ],
    "station_prices": {0: 0.85, 2: 0.92, 4: 0.78},
    "station_resource_state": {"fast_queue_len": 2, "slow_queue_len": 1},
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one InteractionOrchestrator.step() and report the result.")
    parser.add_argument("--root", default=".", help="EV_charging project root")
    parser.add_argument("--day", type=int, default=1, choices=(1, 2, 3))
    parser.add_argument("--observation", default=None, help="Path to a JSON file with the observation payload.")
    parser.add_argument("--json", action="store_true", help="Print the full OrchestrationStep as JSON.")
    args = parser.parse_args()

    registry = default_asset_registry(Path(args.root))
    city = build_default_city(registry, day=args.day)
    root_agent = build_default_root_agent(registry)
    orch = InteractionOrchestrator(root_agent=root_agent, city=city)

    if args.observation:
        observation = json.loads(Path(args.observation).read_text(encoding="utf-8"))
    else:
        observation = _DEFAULT_OBSERVATION

    step = orch.step(observation)

    if args.json:
        payload = {
            "execution_order": list(step.execution_order),
            "records": [asdict(record) for record in step.records],
            "control_flow_errors": list(step.control_flow_errors),
            "geo_errors": list(step.geo_errors),
            "control_flow_graph": (
                step.control_flow_graph.to_dict() if step.control_flow_graph is not None else None
            ),
            "geo_graphs": [graph.to_dict() for graph in step.geo_graphs],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        return 0

    print(f"day: {args.day}  city: {city.city_id}  regions: {[r.region_id for r in city.regions]}")
    print(f"execution_order: {' -> '.join(step.execution_order)}")
    accepted = sum(1 for r in step.records if r.validation["accepted"])
    print(f"proposals: {accepted}/{len(step.records)} accepted")
    for record in step.records:
        verdict = "ok " if record.validation["accepted"] else "REJ"
        violations = record.validation["violations"]
        confidence = record.proposal.get("confidence")
        conf_str = f"conf={confidence:.2f}" if confidence is not None else "conf=n/a"
        suffix = f" violations={len(violations)}" if violations else ""
        print(
            f"  [{verdict}] {record.agent_id:24s} {record.proposal['proposal_type']:24s} {conf_str}{suffix}"
        )
    print(f"control_flow errors: {len(step.control_flow_errors)}  geo errors: {len(step.geo_errors)}")
    for err in step.control_flow_errors:
        print(f"  control_flow: {err}")
    for err in step.geo_errors:
        print(f"  geo: {err}")
    return 0 if not step.control_flow_errors and not step.geo_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
