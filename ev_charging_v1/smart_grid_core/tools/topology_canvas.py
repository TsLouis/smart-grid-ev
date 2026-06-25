from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..visualization.topology_dashboard import build_topology_snapshot, write_topology_dashboard


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a twin-owned topology canvas with runtime information flow.")
    parser.add_argument("--root", default=".", help="EV_charging project root")
    parser.add_argument("--output-dir", default="output_topology_canvas")
    parser.add_argument("--day", type=int, default=1, choices=(1, 2, 3))
    parser.add_argument("--max-traffic-edges", type=int, default=80)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    snapshot = build_topology_snapshot(
        Path(args.root),
        day=args.day,
        max_traffic_edges=args.max_traffic_edges,
    )
    outputs = write_topology_dashboard(snapshot, args.output_dir)
    if args.json:
        print(json.dumps(outputs, ensure_ascii=False, indent=2))
    else:
        print(f"topology_json: {outputs['json_path']}")
        print(f"topology_html: {outputs['html_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
