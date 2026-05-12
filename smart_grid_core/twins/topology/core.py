from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class Domain(str, Enum):
    """Top-level partition of topology graphs.

    GEO graphs describe physical/spatial reality (grid, traffic, station,
    vehicle layers placed at coordinates). CONTROL_FLOW graphs describe
    runtime structure (root agent -> subagents -> tools, agent invocation
    DAGs). The two domains share the same Node/Edge/Graph shape but obey
    different invariants and are validated separately.
    """

    GEO = "geo"
    CONTROL_FLOW = "control_flow"


@dataclass(frozen=True)
class TopologyNode:
    id: str
    label: str
    twin_id: str
    layer: str
    kind: str
    x: float
    y: float
    state: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TopologyEdge:
    id: str
    source: str
    target: str
    twin_id: str
    layer: str
    kind: str
    directed: bool = False
    weight: float | None = None
    state: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TopologyGraph:
    twin_id: str
    layer: str
    nodes: list[TopologyNode]
    edges: list[TopologyEdge]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    domain: Domain = Domain.GEO
    region_id: str | None = None
    parent_region_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "twin_id": self.twin_id,
            "layer": self.layer,
            "nodes": [node.__dict__ for node in self.nodes],
            "edges": [edge.__dict__ for edge in self.edges],
            "metadata": dict(self.metadata),
            "domain": self.domain.value,
            "region_id": self.region_id,
            "parent_region_id": self.parent_region_id,
        }
