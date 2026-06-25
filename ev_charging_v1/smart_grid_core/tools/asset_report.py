from __future__ import annotations

import argparse
from pathlib import Path

from ..assets import default_asset_registry


def main() -> int:
    parser = argparse.ArgumentParser(description="Report registered smart-grid assets.")
    parser.add_argument("--root", default=".", help="EV_charging project root")
    args = parser.parse_args()

    registry = default_asset_registry(Path(args.root))
    for asset in registry.list():
        status = "ok" if registry.exists(asset.key) else "missing"
        required = "required" if asset.required else "optional"
        print(f"{status}\t{required}\t{asset.category}\t{asset.owner}\t{asset.key}\t{asset.path}")

    missing = registry.missing_required()
    if missing:
        print("\nMissing required assets:")
        for asset in missing:
            print(f"- {asset.key}: {asset.path}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
