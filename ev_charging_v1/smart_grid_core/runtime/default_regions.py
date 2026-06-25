from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..assets import AssetRegistry
from ..twins import (
    CityTwin,
    GridTwin,
    PrecomputedTrafficTwin,
    RegionTwin,
    StationAlignment,
)


DEFAULT_STATION_ALIGNMENTS: tuple[StationAlignment, ...] = (
    StationAlignment(station_index=0, grid_node=3, traffic_node=7, label="S1"),
    StationAlignment(station_index=1, grid_node=8, traffic_node=8, label="S2"),
    StationAlignment(station_index=2, grid_node=14, traffic_node=10, label="S3"),
    StationAlignment(station_index=3, grid_node=15, traffic_node=17, label="S4"),
    StationAlignment(station_index=4, grid_node=18, traffic_node=20, label="S5"),
    StationAlignment(station_index=5, grid_node=9, traffic_node=12, label="S6"),
    StationAlignment(station_index=6, grid_node=21, traffic_node=22, label="S7"),
)


@dataclass(frozen=True)
class RegionSpec:
    region_id: str
    parent_region_id: str
    coord_offset: tuple[float, float]
    perturbation_seed: int | None
    perturbation_scale: float
    label: str


def default_city_regions() -> tuple[RegionSpec, ...]:
    """Two-district reference city.

    district_a is the verbatim copy of the legacy single-region data;
    district_b is the same shape with seeded gaussian perturbation and a
    fixed coordinate offset so it renders next to district_a in the
    topology canvas.
    """
    return (
        RegionSpec(
            region_id="district_a",
            parent_region_id="city",
            coord_offset=(0.0, 0.0),
            perturbation_seed=None,
            perturbation_scale=0.0,
            label="District A",
        ),
        RegionSpec(
            region_id="district_b",
            parent_region_id="city",
            coord_offset=(20.0, 0.0),
            perturbation_seed=42,
            perturbation_scale=0.05,
            label="District B",
        ),
    )


def build_region_twin(
    *,
    region: RegionSpec,
    base_registry: AssetRegistry,
    day: int,
    alignments: Sequence[StationAlignment] = DEFAULT_STATION_ALIGNMENTS,
) -> RegionTwin:
    region_registry = base_registry.for_region(region.region_id)
    grid = GridTwin.from_registry(region_registry, day=day)
    traffic = PrecomputedTrafficTwin.from_registry(
        region_registry,
        day=day,
        station_nodes=[a.traffic_node for a in alignments],
    )
    return RegionTwin(
        region_id=region.region_id,
        parent_region_id=region.parent_region_id,
        grid=grid,
        traffic=traffic,
        station_alignments=list(alignments),
        coord_offset=region.coord_offset,
        label=region.label,
    )


def build_default_city(
    base_registry: AssetRegistry,
    *,
    day: int,
    city_id: str = "city",
    regions: Sequence[RegionSpec] | None = None,
    alignments: Sequence[StationAlignment] = DEFAULT_STATION_ALIGNMENTS,
) -> CityTwin:
    region_specs = list(regions) if regions is not None else list(default_city_regions())
    region_twins = [
        build_region_twin(
            region=spec, base_registry=base_registry, day=day, alignments=alignments
        )
        for spec in region_specs
    ]
    return CityTwin(city_id=city_id, regions=region_twins)
