from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

from .grid import DEFAULT_GRID_LAYOUT, GridTwin
from .topology import (
    COUPLING_EDGE_KIND,
    Domain,
    TopologyEdge,
    TopologyGraph,
    TopologyNode,
    TopologyRegistry,
    validate_coupling_graph,
    validate_geo_graph,
    validate_station_pairing,
)
from .traffic import DEFAULT_TRAFFIC_LAYOUT, PrecomputedTrafficTwin


@dataclass(frozen=True)
class StationAlignment:
    """Maps a single charging station's identity across grid and traffic layers.

    A station is the only physical entity that lives in both layers; the
    alignment is the source of truth that ties grid_node <-> traffic_node
    via station_index.
    """

    station_index: int
    grid_node: int
    traffic_node: int
    label: str | None = None


class RegionTwin:
    """One district: aggregates grid + traffic + station-alignment.

    The region owns spatial placement (coord_offset) and the alignment
    table. Inner twins stay region-agnostic; the region wraps their
    topology graphs with region_id, prefixed IDs, and the offset.
    """

    def __init__(
        self,
        *,
        region_id: str,
        parent_region_id: str | None,
        grid: GridTwin,
        traffic: PrecomputedTrafficTwin,
        station_alignments: Sequence[StationAlignment],
        coord_offset: tuple[float, float] = (0.0, 0.0),
        label: str | None = None,
    ) -> None:
        self.region_id = region_id
        self.parent_region_id = parent_region_id
        self.grid = grid
        self.traffic = traffic
        self.station_alignments = list(station_alignments)
        self.coord_offset = coord_offset
        self.label = label or region_id
        self._validate_alignments()

    def _validate_alignments(self) -> None:
        n_stations = self.grid.config.n_stations
        if len(self.station_alignments) != n_stations:
            raise ValueError(
                f"region {self.region_id}: expected {n_stations} alignments, got {len(self.station_alignments)}"
            )
        seen = {a.station_index for a in self.station_alignments}
        if seen != set(range(n_stations)):
            raise ValueError(
                f"region {self.region_id}: station_index coverage incomplete: {sorted(seen)}"
            )
        traffic_count = self.traffic.road_time_table.shape[2]
        if traffic_count != n_stations:
            raise ValueError(
                f"region {self.region_id}: traffic table has {traffic_count} stations, grid has {n_stations}"
            )

    @property
    def station_grid_nodes(self) -> list[int]:
        ordered = sorted(self.station_alignments, key=lambda a: a.station_index)
        return [a.grid_node for a in ordered]

    @property
    def station_traffic_nodes(self) -> list[int]:
        ordered = sorted(self.station_alignments, key=lambda a: a.station_index)
        return [a.traffic_node for a in ordered]

    def grid_topology(self, *, query_time_h: float | None = None) -> TopologyGraph:
        graph = self.grid.topology_graph(
            station_grid_nodes=self.station_grid_nodes,
            query_time_h=query_time_h,
        )
        return self._wrap(graph)

    def traffic_topology(
        self, *, flow_tensor=None, max_edges: int = 80, query_time_h: float | None = None
    ) -> TopologyGraph:
        graph = self.traffic.topology_graph(
            flow_tensor=flow_tensor,
            station_nodes=self.station_traffic_nodes,
            max_edges=max_edges,
            query_time_h=query_time_h,
        )
        return self._wrap(graph)

    def station_topology(self) -> TopologyGraph:
        ordered = sorted(self.station_alignments, key=lambda a: a.station_index)
        ox, oy = self.coord_offset
        nodes: list[TopologyNode] = []
        for alignment in ordered:
            gx, gy = DEFAULT_GRID_LAYOUT.get(alignment.grid_node, (0.0, 0.0))
            tx, ty = DEFAULT_TRAFFIC_LAYOUT.get(alignment.traffic_node, (0.0, 0.0))
            nodes.append(
                TopologyNode(
                    id=f"{self.region_id}:station:{alignment.station_index}",
                    label=alignment.label or f"S{alignment.station_index}",
                    twin_id="station",
                    layer="station",
                    kind="station",
                    x=(gx + tx) / 2 + ox,
                    y=(gy + ty) / 2 + oy,
                    state={},
                    metadata={
                        "station_index": alignment.station_index,
                        "grid_node": alignment.grid_node,
                        "traffic_node": alignment.traffic_node,
                    },
                )
            )
        return TopologyGraph(
            twin_id="station",
            layer="station",
            nodes=nodes,
            edges=[],
            metadata={"source": "region_twin", "purpose": "station_alignment"},
            domain=Domain.GEO,
            region_id=self.region_id,
            parent_region_id=self.parent_region_id,
        )

    def coupling_topology(self) -> TopologyGraph:
        """Interlayer edges connecting grid station nodes to their traffic peers.

        Edges live in a coupling graph with empty `nodes` — the source/target
        refer to IDs already present in `grid_topology()` and `traffic_topology()`.
        Coupling strength can later be set via `weight`.
        """
        prefix = f"{self.region_id}:"
        edges: list[TopologyEdge] = []
        for alignment in sorted(self.station_alignments, key=lambda a: a.station_index):
            grid_id = f"{prefix}grid:{alignment.grid_node}"
            traffic_id = f"{prefix}traffic:{alignment.traffic_node}"
            edges.append(
                TopologyEdge(
                    id=f"{prefix}coupling:station:{alignment.station_index}",
                    source=grid_id,
                    target=traffic_id,
                    twin_id="region",
                    layer="coupling",
                    kind=COUPLING_EDGE_KIND,
                    directed=False,
                    weight=None,
                    metadata={
                        "station_index": alignment.station_index,
                        "label": alignment.label or f"S{alignment.station_index}",
                        "grid_node": alignment.grid_node,
                        "traffic_node": alignment.traffic_node,
                    },
                )
            )
        return TopologyGraph(
            twin_id="region",
            layer="coupling",
            nodes=[],
            edges=edges,
            metadata={"source": "region_twin", "purpose": "station_interlayer"},
            domain=Domain.GEO,
            region_id=self.region_id,
            parent_region_id=self.parent_region_id,
        )

    def topology_graphs(
        self,
        *,
        flow_tensor=None,
        max_edges: int = 80,
        query_time_h: float | None = None,
    ) -> list[TopologyGraph]:
        return [
            self.grid_topology(query_time_h=query_time_h),
            self.traffic_topology(
                flow_tensor=flow_tensor, max_edges=max_edges, query_time_h=query_time_h
            ),
            self.station_topology(),
            self.coupling_topology(),
        ]

    def register_into(
        self,
        registry: TopologyRegistry,
        *,
        flow_tensor=None,
        max_edges: int = 80,
        query_time_h: float | None = None,
    ) -> list[str]:
        errors: list[str] = []
        graphs = self.topology_graphs(
            flow_tensor=flow_tensor, max_edges=max_edges, query_time_h=query_time_h
        )
        coupling_graph = next((g for g in graphs if g.layer == "coupling"), None)
        for graph in graphs:
            if graph is coupling_graph:
                continue
            errors.extend(validate_geo_graph(graph))
            registry.register(graph)
        if coupling_graph is not None:
            neighbor_graphs = [g for g in graphs if g is not coupling_graph]
            errors.extend(validate_coupling_graph(coupling_graph, neighbor_graphs=neighbor_graphs))
            registry.register(coupling_graph)
        grid_graph = next((g for g in graphs if g.layer == "grid"), None)
        traffic_graph = next((g for g in graphs if g.layer == "traffic"), None)
        if grid_graph is not None and traffic_graph is not None:
            station_indices = [a.station_index for a in self.station_alignments]
            errors.extend(
                validate_station_pairing(
                    grid_graph, traffic_graph, station_indices=station_indices
                )
            )
        return errors

    def _wrap(self, graph: TopologyGraph) -> TopologyGraph:
        ox, oy = self.coord_offset
        prefix = f"{self.region_id}:"
        new_nodes = [
            replace(node, id=f"{prefix}{node.id}", x=node.x + ox, y=node.y + oy)
            for node in graph.nodes
        ]
        new_edges = [
            replace(
                edge,
                id=f"{prefix}{edge.id}",
                source=f"{prefix}{edge.source}",
                target=f"{prefix}{edge.target}",
            )
            for edge in graph.edges
        ]
        return TopologyGraph(
            twin_id=graph.twin_id,
            layer=graph.layer,
            nodes=new_nodes,
            edges=new_edges,
            metadata=graph.metadata,
            domain=graph.domain,
            region_id=self.region_id,
            parent_region_id=self.parent_region_id,
        )
