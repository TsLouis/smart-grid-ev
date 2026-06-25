from __future__ import annotations

import heapq
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..runtime.manifest import AgentManifest
from ..twins.topology import (
    Domain,
    TopologyEdge,
    TopologyGraph,
    TopologyNode,
    validate_control_flow_graph,
)
from .base import AgentPolicy, AgentProposal


# Within a single topological level, run earlier-stage groups first.
# Advisor runs last so it observes proposals from all other groups.
_GROUP_RANK: dict[str, int] = {
    "perception": 0,
    "decision": 1,
    "policy": 2,
    "advisor": 3,
}


@dataclass(frozen=True)
class SubagentBinding:
    """Pairs a runnable AgentPolicy with its declared manifest.

    The manifest's `consumes`/`produces` capabilities drive the
    execution order; the policy's `propose()` does the actual work.
    """

    policy: AgentPolicy
    manifest: AgentManifest
    group: str

    @property
    def agent_id(self) -> str:
        return self.manifest.agent_id


@dataclass(frozen=True)
class ToolBinding:
    """A stateless lookup callable owned by the root agent.

    Tools are registered for visibility and validation only; subagents
    invoke them directly through their constructors. Tools always
    appear as leaves in the control-flow graph.
    """

    tool_id: str
    description: str = ""


@dataclass
class RootAgentRun:
    """Output of one RootAgent.step().

    Holds the proposals collected this turn together with the
    control-flow graph that captures which subagents/tools ran and
    how they fed each other.
    """

    proposals: dict[str, AgentProposal] = field(default_factory=dict)
    execution_order: list[str] = field(default_factory=list)
    graph: TopologyGraph | None = None
    graph_errors: list[str] = field(default_factory=list)


class RootAgent:
    """Single root agent that owns subagents (grouped) and tools.

    The root is the project-level Agent contract: callers hand it an
    observation; it schedules its subagents in topological order based
    on each subagent's manifest, hands each subagent a merged view of
    the original observation plus upstream proposals, and returns a
    control-flow graph describing the run.
    """

    def __init__(
        self,
        *,
        agent_id: str = "ev_charging_root_agent",
        subagents: Sequence[SubagentBinding] = (),
        tools: Sequence[ToolBinding] = (),
    ) -> None:
        self.agent_id = agent_id
        self._subagents: dict[str, SubagentBinding] = {}
        self._tools: dict[str, ToolBinding] = {}
        for subagent in subagents:
            self.register_subagent(subagent)
        for tool in tools:
            self.register_tool(tool)

    def register_subagent(self, binding: SubagentBinding) -> None:
        if binding.agent_id in self._subagents:
            raise ValueError(f"subagent {binding.agent_id!r} already registered")
        self._subagents[binding.agent_id] = binding

    def register_tool(self, tool: ToolBinding) -> None:
        if tool.tool_id in self._tools:
            raise ValueError(f"tool {tool.tool_id!r} already registered")
        self._tools[tool.tool_id] = tool

    @property
    def subagents(self) -> list[SubagentBinding]:
        return list(self._subagents.values())

    @property
    def tools(self) -> list[ToolBinding]:
        return list(self._tools.values())

    def execution_order(self) -> list[str]:
        """Topological sort of subagent ids by produces -> consumes edges."""
        producer_index: dict[str, str] = {}
        for binding in self._subagents.values():
            for capability in binding.manifest.produces:
                producer_index[capability] = binding.agent_id

        indegree: dict[str, int] = {agent_id: 0 for agent_id in self._subagents}
        successors: dict[str, list[str]] = defaultdict(list)
        for binding in self._subagents.values():
            for capability in binding.manifest.consumes:
                producer = producer_index.get(capability)
                if producer is None or producer == binding.agent_id:
                    continue
                successors[producer].append(binding.agent_id)
                indegree[binding.agent_id] += 1

        heap: list[tuple[int, str]] = []
        for agent_id, deg in indegree.items():
            if deg == 0:
                heapq.heappush(heap, self._priority_key(agent_id))
        order: list[str] = []
        local_indegree = dict(indegree)
        while heap:
            _, current = heapq.heappop(heap)
            order.append(current)
            for successor in successors.get(current, ()):
                local_indegree[successor] -= 1
                if local_indegree[successor] == 0:
                    heapq.heappush(heap, self._priority_key(successor))
        if len(order) != len(self._subagents):
            cyclic = sorted(set(self._subagents) - set(order))
            raise ValueError(
                f"subagent dependency cycle involving: {cyclic}"
            )
        return order

    def _priority_key(self, agent_id: str) -> tuple[int, str]:
        binding = self._subagents[agent_id]
        return (_GROUP_RANK.get(binding.group, 99), agent_id)

    def control_flow_graph(self) -> TopologyGraph:
        """Static control-flow topology for this root.

        Independent of any single observation. RootAgent.step() builds
        the same shape but tags each subagent with the run's confidence.
        """
        return self._build_graph(proposals={})

    def step(self, observation: Mapping[str, Any]) -> RootAgentRun:
        run = RootAgentRun()
        merged: dict[str, Any] = dict(observation)
        for agent_id in self.execution_order():
            binding = self._subagents[agent_id]
            proposal = binding.policy.propose(merged)
            run.proposals[agent_id] = proposal
            run.execution_order.append(agent_id)
            for capability in binding.manifest.produces:
                merged[capability] = dict(proposal.payload)
            merged.setdefault("specialist_proposals", []).append(
                {
                    "agent_id": agent_id,
                    "proposal_type": proposal.proposal_type,
                    "confidence": proposal.confidence,
                    "payload": dict(proposal.payload),
                }
            )
        run.graph = self._build_graph(run.proposals)
        run.graph_errors = validate_control_flow_graph(run.graph)
        return run

    def _build_graph(self, proposals: Mapping[str, AgentProposal]) -> TopologyGraph:
        nodes: list[TopologyNode] = []
        edges: list[TopologyEdge] = []

        nodes.append(
            TopologyNode(
                id=self.agent_id,
                label=self.agent_id,
                twin_id=self.agent_id,
                layer="control",
                kind="agent",
                x=0.0,
                y=0.0,
                state={},
                metadata={"role": "root_agent"},
            )
        )

        producer_index: dict[str, str] = {}
        for binding in self._subagents.values():
            for capability in binding.manifest.produces:
                producer_index[capability] = binding.agent_id

        groups = sorted({binding.group for binding in self._subagents.values()})
        group_x = {group: float(index) for index, group in enumerate(groups)}
        group_counters: dict[str, int] = defaultdict(int)
        for binding in self._subagents.values():
            agent_id = binding.agent_id
            x = group_x.get(binding.group, 0.0)
            y = float(group_counters[binding.group])
            group_counters[binding.group] += 1
            metadata: dict[str, Any] = {
                "group": binding.group,
                "agent_type": binding.manifest.agent_type,
                "consumes": list(binding.manifest.consumes),
                "produces": list(binding.manifest.produces),
            }
            proposal = proposals.get(agent_id)
            if proposal is not None:
                metadata["confidence"] = proposal.confidence
                metadata["proposal_type"] = proposal.proposal_type
            nodes.append(
                TopologyNode(
                    id=agent_id,
                    label=agent_id,
                    twin_id=self.agent_id,
                    layer="control",
                    kind="subagent",
                    x=x,
                    y=y + 1.0,
                    state={},
                    metadata=metadata,
                )
            )
            edges.append(
                TopologyEdge(
                    id=f"{self.agent_id}->{agent_id}",
                    source=self.agent_id,
                    target=agent_id,
                    twin_id=self.agent_id,
                    layer="control",
                    kind="delegates",
                    directed=True,
                    metadata={"group": binding.group},
                )
            )
            for capability in binding.manifest.consumes:
                producer = producer_index.get(capability)
                if producer is None or producer == agent_id:
                    continue
                edges.append(
                    TopologyEdge(
                        id=f"{producer}->{agent_id}:{capability}",
                        source=producer,
                        target=agent_id,
                        twin_id=self.agent_id,
                        layer="control",
                        kind="data_flow",
                        directed=True,
                        metadata={"capability": capability},
                    )
                )

        for index, tool in enumerate(self._tools.values()):
            nodes.append(
                TopologyNode(
                    id=tool.tool_id,
                    label=tool.tool_id,
                    twin_id=self.agent_id,
                    layer="control",
                    kind="tool",
                    x=float(len(groups)),
                    y=float(index),
                    state={},
                    metadata={"description": tool.description},
                )
            )
            edges.append(
                TopologyEdge(
                    id=f"{self.agent_id}->{tool.tool_id}",
                    source=self.agent_id,
                    target=tool.tool_id,
                    twin_id=self.agent_id,
                    layer="control",
                    kind="invokes",
                    directed=True,
                )
            )

        return TopologyGraph(
            twin_id=self.agent_id,
            layer="control",
            nodes=nodes,
            edges=edges,
            metadata={"source": "root_agent", "groups": groups},
            domain=Domain.CONTROL_FLOW,
            region_id=None,
            parent_region_id=None,
        )
