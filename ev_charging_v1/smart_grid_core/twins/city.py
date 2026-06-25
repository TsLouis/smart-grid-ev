from __future__ import annotations

from typing import Sequence

from .region import RegionTwin
from .topology import TopologyGraph, TopologyRegistry


class CityTwin:
    """Top-level container holding multiple RegionTwins.

    The city is a registry coordinator: it builds a TopologyRegistry that
    indexes every region's geo graphs, validates them, and exposes a flat
    snapshot for dashboards. Region-level state stays inside RegionTwin.
    """

    def __init__(self, *, city_id: str = "city", regions: Sequence[RegionTwin] = ()) -> None:
        self.city_id = city_id
        self.regions: list[RegionTwin] = []
        for region in regions:
            self.add_region(region)

    def add_region(self, region: RegionTwin) -> None:
        if region.region_id in {r.region_id for r in self.regions}:
            raise ValueError(f"region {region.region_id} already registered")
        if region.parent_region_id not in (None, self.city_id):
            raise ValueError(
                f"region {region.region_id} parent_region_id {region.parent_region_id!r} "
                f"does not match city_id {self.city_id!r}"
            )
        self.regions.append(region)

    def build_registry(
        self,
        *,
        flow_tensors: dict[str, object] | None = None,
        max_edges: int = 80,
        query_time_h: float | None = None,
    ) -> tuple[TopologyRegistry, list[str]]:
        registry = TopologyRegistry()
        errors: list[str] = []
        flow_tensors = flow_tensors or {}
        for region in self.regions:
            region_errors = region.register_into(
                registry,
                flow_tensor=flow_tensors.get(region.region_id),
                max_edges=max_edges,
                query_time_h=query_time_h,
            )
            errors.extend(f"[{region.region_id}] {msg}" for msg in region_errors)
        return registry, errors

    def topology_graphs(
        self,
        *,
        flow_tensors: dict[str, object] | None = None,
        max_edges: int = 80,
        query_time_h: float | None = None,
    ) -> list[TopologyGraph]:
        registry, _ = self.build_registry(
            flow_tensors=flow_tensors, max_edges=max_edges, query_time_h=query_time_h
        )
        return registry.all()
