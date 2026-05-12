from __future__ import annotations

from collections import defaultdict, deque
from typing import Iterable

from .core import TopologyGraph
from .registry import TopologyRegistry


def graph_metrics(graph: TopologyGraph) -> dict:
    """Compute headline complex-network metrics for one graph.

    Returns degree distribution, average degree, density, and (for
    connected components) the largest component's BFS diameter. Coupling
    graphs typically reference external nodes — we report what we can.
    """
    n = len(graph.nodes)
    m = len(graph.edges)
    if n == 0:
        return {
            "node_count": 0,
            "edge_count": m,
            "avg_degree": 0.0,
            "density": 0.0,
            "diameter": None,
            "components": 0,
            "degree_histogram": {},
        }

    adjacency: dict[str, set[str]] = defaultdict(set)
    node_ids = {node.id for node in graph.nodes}
    for edge in graph.edges:
        if edge.source in node_ids and edge.target in node_ids:
            adjacency[edge.source].add(edge.target)
            adjacency[edge.target].add(edge.source)

    degrees = [len(adjacency.get(node.id, set())) for node in graph.nodes]
    histogram: dict[int, int] = defaultdict(int)
    for degree in degrees:
        histogram[degree] += 1

    components = _connected_components(node_ids, adjacency)
    largest = max(components, key=len) if components else set()
    diameter = _bfs_diameter(largest, adjacency) if largest else None

    return {
        "node_count": n,
        "edge_count": m,
        "avg_degree": (sum(degrees) / n) if n else 0.0,
        "density": (2 * m) / (n * (n - 1)) if n > 1 else 0.0,
        "diameter": diameter,
        "components": len(components),
        "degree_histogram": dict(sorted(histogram.items())),
    }


def registry_metrics(registry: TopologyRegistry) -> dict:
    """Aggregate metrics for an entire TopologyRegistry."""
    per_graph: list[dict] = []
    node_total = 0
    edge_total = 0
    interlayer_edges = 0
    for graph in registry.all():
        m = graph_metrics(graph)
        per_graph.append(
            {
                "twin_id": graph.twin_id,
                "layer": graph.layer,
                "region_id": graph.region_id,
                "domain": graph.domain.value,
                **m,
            }
        )
        node_total += m["node_count"]
        edge_total += m["edge_count"]
        if graph.layer == "coupling":
            interlayer_edges += m["edge_count"]

    return {
        "graphs": len(per_graph),
        "node_total": node_total,
        "edge_total": edge_total,
        "interlayer_edges": interlayer_edges,
        "per_graph": per_graph,
    }


def _connected_components(
    node_ids: Iterable[str],
    adjacency: dict[str, set[str]],
) -> list[set[str]]:
    seen: set[str] = set()
    components: list[set[str]] = []
    for start in node_ids:
        if start in seen:
            continue
        component: set[str] = set()
        queue = deque([start])
        while queue:
            current = queue.popleft()
            if current in component:
                continue
            component.add(current)
            queue.extend(adjacency.get(current, ()))
        seen |= component
        components.append(component)
    return components


def _bfs_diameter(component: set[str], adjacency: dict[str, set[str]]) -> int:
    """Approximate the diameter via two-pass BFS (exact on trees, bound elsewhere)."""
    if not component:
        return 0
    start = next(iter(component))
    far_node, _ = _bfs_farthest(start, component, adjacency)
    _, depth = _bfs_farthest(far_node, component, adjacency)
    return depth


def _bfs_farthest(
    start: str, component: set[str], adjacency: dict[str, set[str]]
) -> tuple[str, int]:
    distances = {start: 0}
    queue = deque([start])
    farthest = start
    max_depth = 0
    while queue:
        current = queue.popleft()
        depth = distances[current]
        if depth > max_depth:
            max_depth = depth
            farthest = current
        for neighbor in adjacency.get(current, ()):
            if neighbor in component and neighbor not in distances:
                distances[neighbor] = depth + 1
                queue.append(neighbor)
    return farthest, max_depth
