from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from ..assets import AssetRegistry, AssetSpec, default_asset_registry
from ..runtime.default_regions import (
    DEFAULT_STATION_ALIGNMENTS,
    RegionSpec,
    default_city_regions,
)
from ..twins import StationAlignment


_NUMPY_LOADERS = {"numpy"}


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _copy_asset(spec: AssetSpec, source: Path, target: Path) -> None:
    _ensure_dir(target.parent)
    shutil.copy2(source, target)


def _perturb_numpy_asset(
    source: Path,
    target: Path,
    *,
    seed: int,
    scale: float,
) -> None:
    import numpy as np

    _ensure_dir(target.parent)
    array = np.load(source, allow_pickle=True)
    # Only perturb floats. Integer arrays (e.g. line_index) encode discrete
    # node IDs; casting float noise back to int truncates 0.95→0 and breaks
    # topology. Object/int arrays are copied verbatim.
    if array.dtype.kind == "f" and scale > 0:
        rng = np.random.default_rng(seed)
        noise = rng.normal(loc=1.0, scale=scale, size=array.shape)
        perturbed = (array * noise).astype(array.dtype, copy=False)
        np.save(target, perturbed, allow_pickle=False)
    else:
        np.save(target, array, allow_pickle=array.dtype == object)


def clone_region(
    *,
    source_registry: AssetRegistry,
    target_root: Path,
    region_id: str,
    perturbation_seed: int | None,
    perturbation_scale: float,
) -> dict:
    """Materialize one region's data directory.

    Returns a manifest dict listing every asset that was copied or
    perturbed and where it now lives. Assets whose source is missing are
    reported but not fatal — many days are optional.
    """
    target_dir = target_root / "data" / "regions" / region_id
    _ensure_dir(target_dir)
    summary = {"region_id": region_id, "copied": [], "perturbed": [], "skipped_missing": []}
    for spec in source_registry.region_scoped_specs():
        source_path = source_registry.path(spec.key)
        target_path = target_dir / Path(spec.path).name
        if not source_path.exists():
            summary["skipped_missing"].append({"key": spec.key, "expected_at": str(source_path)})
            continue
        if (
            perturbation_seed is not None
            and perturbation_scale > 0
            and spec.loader in _NUMPY_LOADERS
        ):
            _perturb_numpy_asset(
                source_path,
                target_path,
                seed=perturbation_seed + hash(spec.key) % 10_000,
                scale=perturbation_scale,
            )
            summary["perturbed"].append({"key": spec.key, "path": str(target_path)})
        else:
            _copy_asset(spec, source_path, target_path)
            summary["copied"].append({"key": spec.key, "path": str(target_path)})
    return summary


def write_alignment_index(
    target_root: Path,
    *,
    regions: Iterable[RegionSpec],
    alignments: Iterable[StationAlignment],
) -> Path:
    payload = {
        "stations": [asdict(alignment) for alignment in alignments],
        "regions": [
            {
                "region_id": region.region_id,
                "parent_region_id": region.parent_region_id,
                "coord_offset": list(region.coord_offset),
                "perturbation_seed": region.perturbation_seed,
                "perturbation_scale": region.perturbation_scale,
                "label": region.label,
            }
            for region in regions
        ],
    }
    target_path = target_root / "data" / "regions" / "region_alignment.json"
    _ensure_dir(target_path.parent)
    target_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return target_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Project root containing the legacy `Load Forecasting/` directory.",
    )
    args = parser.parse_args(argv)

    project_root = Path(args.project_root).resolve()
    source_registry = default_asset_registry(project_root)
    region_specs = default_city_regions()

    summaries = []
    for region in region_specs:
        summary = clone_region(
            source_registry=source_registry,
            target_root=project_root,
            region_id=region.region_id,
            perturbation_seed=region.perturbation_seed,
            perturbation_scale=region.perturbation_scale,
        )
        summaries.append(summary)
        print(
            f"[{region.region_id}] copied={len(summary['copied'])} "
            f"perturbed={len(summary['perturbed'])} "
            f"skipped_missing={len(summary['skipped_missing'])}"
        )
        for entry in summary["skipped_missing"]:
            print(f"  missing: {entry['key']} (expected at {entry['expected_at']})")

    alignment_path = write_alignment_index(
        project_root, regions=region_specs, alignments=DEFAULT_STATION_ALIGNMENTS
    )
    print(f"alignment index -> {alignment_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
