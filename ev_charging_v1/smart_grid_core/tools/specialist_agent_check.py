from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from ..agents import (
    ChargeNeedAgent,
    ChargingModeAgent,
    GridFriendlyAgent,
    PricePolicyAgent,
    QueueForecastAgent,
    StationDecisionAgent,
    TargetSocAgent,
)
from ..assets import default_asset_registry


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic baseline specialist agents.")
    parser.add_argument("--root", default=".", help="EV_charging project root")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    registry = default_asset_registry(Path(args.root))
    vehicle_state = {
        "vehicle_id": 1,
        "current_node": 0,
        "current_time": 8.0,
        "soc": 0.28,
        "battery_capacity_kwh": 61.4,
    }
    route_observation = {
        "reachable_stations": [
            {"station_index": 0, "travel_time_h": 0.24, "travel_energy_kwh": 4.3},
            {"station_index": 1, "travel_time_h": 0.28, "travel_energy_kwh": 5.2},
            {"station_index": 6, "travel_time_h": 0.16, "travel_energy_kwh": 2.6},
        ],
        "station_prices": {0: 0.69, 1: 0.70, 6: 0.68},
    }
    grid_observation = {
        "renewable_state": {"pv": 0.19, "wind": 0.22},
        "base_load_state": {"residential": 0.65, "commercial": 0.71, "work": 0.89},
    }
    station_observation = {"station_resource_state": {"fast_queue_len": 3, "slow_queue_len": 1}}

    charge_need = ChargeNeedAgent().propose({**vehicle_state, "next_trip_energy_kwh": 8.0})
    proposals = [
        charge_need,
        TargetSocAgent().propose({**vehicle_state, "charge_need": charge_need.payload}),
        ChargingModeAgent().propose({**vehicle_state, "dwell_time_h": 2.0}),
        StationDecisionAgent().propose(route_observation),
        GridFriendlyAgent().propose(grid_observation),
        PricePolicyAgent(registry).propose({}),
        QueueForecastAgent().propose(station_observation),
    ]

    if args.json:
        print(json.dumps([asdict(proposal) for proposal in proposals], ensure_ascii=False, indent=2))
    else:
        for proposal in proposals:
            print(f"{proposal.agent_name}: {proposal.proposal_type} confidence={proposal.confidence}")
            print(f"  {proposal.payload}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
