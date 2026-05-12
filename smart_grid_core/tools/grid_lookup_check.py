from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from ..assets import default_asset_registry
from ..twins.grid import GridTwin


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect deterministic GridTwin states.")
    parser.add_argument("--root", default=".", help="EV_charging project root")
    parser.add_argument("--day", type=int, default=1, choices=[1, 2, 3])
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
    twin = GridTwin.from_registry(registry, day=args.day)
    prices = twin.all_station_prices(query_time_h=args.time)
    renewable = twin.renewable_state(query_time_h=args.time)
    base_load = twin.base_load_state(query_time_h=args.time)

    payload = {
        "snapshot": twin.snapshot(),
        "prices": [asdict(price) for price in prices],
        "renewable": asdict(renewable),
        "base_load": asdict(base_load),
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"day: {args.day}")
        print(f"query_time_h: {args.time}")
        print(f"station_price_step: {prices[0].time_step if prices else 'n/a'}")
        for price in prices:
            print(f"station={price.station_index} price={price.price:.6f}")
        print(
            f"renewable step={renewable.time_step} pv={renewable.pv:.6f} wind={renewable.wind:.6f}"
        )
        print(
            f"base_load step={base_load.time_step} residential={base_load.residential:.6f} "
            f"commercial={base_load.commercial:.6f} work={base_load.work:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
