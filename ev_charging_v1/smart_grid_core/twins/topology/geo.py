from __future__ import annotations

from typing import Iterable

from .core import Domain, TopologyGraph


GEO_LAYERS = frozenset({"grid", "traffic", "station", "vehicle", "coupling"})


COUPLING_EDGE_KIND = "interlayer"


def validate_geo_graph(graph: TopologyGraph) -> list[str]:
    """Return invariant violations for a GEO-domain graph.

    Empty list means the graph is well-formed. Coupling graphs are
    interlayer-only and validated separately (their edges reference IDs
    that live in *other* graphs).
    """
    if graph.domain is not Domain.GEO:
        return [f"graph {graph.twin_id}:{graph.layer} domain is {graph.domain}, not GEO"]
    if graph.layer == "coupling":
        return []

    errors: list[str] = []
    node_ids = {node.id for node in graph.nodes}
    for edge in graph.edges:
        if edge.source not in node_ids:
            errors.append(f"edge {edge.id}: source {edge.source} not in nodes")
        if edge.target not in node_ids:
            errors.append(f"edge {edge.id}: target {edge.target} not in nodes")
    return errors


def validate_coupling_graph(
    coupling_graph: TopologyGraph,
    *,
    neighbor_graphs: Iterable[TopologyGraph],
) -> list[str]:
    """Verify a coupling graph's edges reference real nodes in adjacent layers."""
    if coupling_graph.layer != "coupling":
        return [f"graph {coupling_graph.twin_id}:{coupling_graph.layer} is not a coupling graph"]
    known_ids: set[str] = set()
    for graph in neighbor_graphs:
        if graph is coupling_graph:
            continue
        known_ids.update(node.id for node in graph.nodes)
    errors: list[str] = []
    for edge in coupling_graph.edges:
        if edge.kind != COUPLING_EDGE_KIND:
            errors.append(f"edge {edge.id}: kind {edge.kind!r} must be {COUPLING_EDGE_KIND!r}")
        if edge.source not in known_ids:
            errors.append(f"edge {edge.id}: source {edge.source} not in any neighbor graph")
        if edge.target not in known_ids:
            errors.append(f"edge {edge.id}: target {edge.target} not in any neighbor graph")
    return errors


def validate_station_pairing(
    grid_graph: TopologyGraph,
    traffic_graph: TopologyGraph,
    *,
    station_indices: Iterable[int],
) -> list[str]:
    """Verify each station_index is represented in both grid and traffic graphs.

    Stations are the only physical entity that crosses grid <-> traffic;
    pairing must be explicit by station_index in node metadata.
    """
    errors: list[str] = []
    grid_indices = {n.metadata.get("station_index") for n in grid_graph.nodes if n.kind == "station"}
    traffic_indices = {n.metadata.get("station_index") for n in traffic_graph.nodes if n.kind == "station"}
    for index in station_indices:
        if index not in grid_indices:
            errors.append(f"station_index {index} missing from grid graph")
        if index not in traffic_indices:
            errors.append(f"station_index {index} missing from traffic graph")
    return errors
