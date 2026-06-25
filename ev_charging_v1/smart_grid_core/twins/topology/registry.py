from __future__ import annotations

from typing import Iterator

from .core import Domain, TopologyGraph


class TopologyRegistry:
    """Indexed store for topology graphs across domains and regions.

    Dashboards and orchestration consume graphs only through this registry,
    so adding a new domain or region does not require touching consumers.
    """

    def __init__(self) -> None:
        self._graphs: dict[tuple[Domain, str | None, str], TopologyGraph] = {}

    def register(self, graph: TopologyGraph) -> None:
        key = (graph.domain, graph.region_id, f"{graph.twin_id}:{graph.layer}")
        self._graphs[key] = graph

    def get(self, *, domain: Domain, region_id: str | None, twin_layer: str) -> TopologyGraph | None:
        return self._graphs.get((domain, region_id, twin_layer))

    def by_region(self, region_id: str | None) -> list[TopologyGraph]:
        return [graph for (_, region, _), graph in self._graphs.items() if region == region_id]

    def by_domain(self, domain: Domain) -> list[TopologyGraph]:
        return [graph for (graph_domain, _, _), graph in self._graphs.items() if graph_domain is domain]

    def all(self) -> list[TopologyGraph]:
        return list(self._graphs.values())

    def __iter__(self) -> Iterator[TopologyGraph]:
        return iter(self._graphs.values())

    def __len__(self) -> int:
        return len(self._graphs)
