from __future__ import annotations

import argparse
from dataclasses import asdict
import json

from ..twins.vehicle import DwellInstruction, TripInstruction, VehicleConfig, VehicleTwin


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a deterministic VehicleTwin state-transition check.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    vehicle = VehicleTwin(
        VehicleConfig(
            vehicle_id=1,
            initial_node=0,
            initial_time=8.0,
            initial_soc=0.8,
            battery_capacity_kwh=60.0,
            target_soc=0.95,
            population="demo",
        )
    )
    records = [asdict(vehicle.snapshot())]
    vehicle.apply_trip(
        TripInstruction(
            destination_node=7,
            departure_time=8.0,
            travel_time_h=0.25,
            travel_energy_kwh=6.0,
            dwell_time_h=1.0,
        )
    )
    records.append(asdict(vehicle.snapshot()))
    vehicle.end_dwell()
    records.append(asdict(vehicle.snapshot()))

    if args.json:
        print(json.dumps(records, ensure_ascii=False, indent=2))
    else:
        for record in records:
            print(
                f"vehicle={record['vehicle_id']} node={record['current_node']} "
                f"time={record['current_time']:.3f} soc={record['soc']:.3f} mode={record['mode']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
