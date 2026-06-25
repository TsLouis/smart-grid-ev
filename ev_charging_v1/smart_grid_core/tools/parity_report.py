from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..assets import default_asset_registry
from ..parity import compare_replay_to_baseline, report_as_dict


SCENARIOS = {
    "jun30": ("legacy_jun30_events", "legacy_jun30_load"),
    "jul1": ("legacy_jul1_events", "legacy_jul1_load"),
    "jul2": ("legacy_jul2_events", "legacy_jul2_load"),
    "jun30_no_queue": ("legacy_jun30_no_queue_events", "legacy_jun30_no_queue_load"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay legacy charge events and compare load baselines.")
    parser.add_argument("--root", default=".", help="EV_charging project root")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="jun30")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    args = parser.parse_args()

    registry = default_asset_registry(Path(args.root))
    event_key, load_key = SCENARIOS[args.scenario]
    report = compare_replay_to_baseline(
        event_path=registry.path(event_key),
        baseline_path=registry.path(load_key),
    )

    if args.json:
        print(json.dumps(report_as_dict(report), ensure_ascii=False, indent=2))
    else:
        print(f"scenario: {args.scenario}")
        print(f"events: {report.event_count}")
        print(f"baseline_shape: {report.baseline_shape}")
        print(f"replay_shape: {report.replay_shape}")
        print(f"max_abs_error: {report.max_abs_error:.6f}")
        print(f"mean_abs_error: {report.mean_abs_error:.6f}")
        print(f"baseline_sum: {report.total_baseline_energy_step_kw:.6f}")
        print(f"replay_sum: {report.total_replay_energy_step_kw:.6f}")
        if report.notes:
            print("notes:")
            for note in report.notes:
                print(f"- {note}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
