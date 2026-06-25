from __future__ import annotations

from collections import defaultdict, deque

from .core import Domain, TopologyGraph


CONTROL_FLOW_KINDS = frozenset({"agent", "subagent", "tool"})


def validate_control_flow_graph(graph: TopologyGraph) -> list[str]:
    """Return invariant violations for a CONTROL_FLOW-domain graph.

    Rules:
      - graph.domain must be CONTROL_FLOW
      - every node.kind must be in CONTROL_FLOW_KINDS
      - tool nodes must be leaves (no outgoing edges)
      - subagent nodes must have at least one incoming edge
      - the graph must be acyclic
      - every edge endpoint must exist
    """
    if graph.domain is not Domain.CONTROL_FLOW:
        return [f"graph {graph.twin_id}:{graph.layer} domain is {graph.domain}, not CONTROL_FLOW"]

    errors: list[str] = []
    nodes_by_id = {n.id: n for n in graph.nodes}
    for node in graph.nodes:
        if node.kind not in CONTROL_FLOW_KINDS:
            errors.append(f"node {node.id}: kind {node.kind!r} not in {sorted(CONTROL_FLOW_KINDS)}")

    indegree: dict[str, int] = defaultdict(int)
    outgoing: dict[str, list[str]] = defaultdict(list)
    for edge in graph.edges:
        if edge.source not in nodes_by_id:
            errors.append(f"edge {edge.id}: source {edge.source} not in nodes")
            continue
        if edge.target not in nodes_by_id:
            errors.append(f"edge {edge.id}: target {edge.target} not in nodes")
            continue
        if nodes_by_id[edge.source].kind == "tool":
            errors.append(f"edge {edge.id}: tool {edge.source} cannot be a source")
        outgoing[edge.source].append(edge.target)
        indegree[edge.target] += 1

    for node in graph.nodes:
        if node.kind == "subagent" and indegree[node.id] == 0:
            errors.append(f"subagent {node.id} has no parent")

    visited = 0
    queue = deque(n.id for n in graph.nodes if indegree[n.id] == 0)
    local_indegree = dict(indegree)
    while queue:
        current = queue.popleft()
        visited += 1
        for successor in outgoing.get(current, ()):
            local_indegree[successor] -= 1
            if local_indegree[successor] == 0:
                queue.append(successor)
    if visited != len(graph.nodes):
        errors.append("control flow graph contains a cycle")

    return errors
