from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from .agents import AgentProposal, RootAgent, RootAgentRun
from .twins import CityTwin
from .twins.base import TwinAction, TwinResult, TwinStateValidator
from .twins.topology import TopologyGraph, TopologyRegistry


@dataclass
class ProposalRecord:
    """One proposal plus the twin layer's validation verdict."""

    agent_id: str
    proposal: dict[str, Any]
    validation: dict[str, Any]


@dataclass
class OrchestrationStep:
    """Output of one InteractionOrchestrator.step().

    Aggregates the agent run (proposals + control-flow graph) and the
    twin-side topology snapshot (per-region geo graphs) into a single
    serializable record.
    """

    records: list[ProposalRecord] = field(default_factory=list)
    control_flow_graph: TopologyGraph | None = None
    control_flow_errors: list[str] = field(default_factory=list)
    geo_graphs: list[TopologyGraph] = field(default_factory=list)
    geo_errors: list[str] = field(default_factory=list)
    execution_order: list[str] = field(default_factory=list)


class InteractionOrchestrator:
    """Coordination layer between the root agent and the city twin.

    The root agent owns subagents and tools; the city twin owns regions
    and their geo topology. The orchestrator runs one observation
    through both sides and returns a single step record that the
    dashboard, parity tooling, or downstream RL loop can consume.
    """

    def __init__(
        self,
        *,
        root_agent: RootAgent,
        city: CityTwin | None = None,
        validator: TwinStateValidator | None = None,
    ) -> None:
        self.root_agent = root_agent
        self.city = city
        self.validator = validator or TwinStateValidator()

    def step(
        self,
        observation: Mapping[str, Any],
        *,
        flow_tensors: dict[str, object] | None = None,
        max_traffic_edges: int = 80,
        query_time_h: float | None = None,
    ) -> OrchestrationStep:
        run = self.root_agent.step(observation)
        step = OrchestrationStep(
            control_flow_graph=run.graph,
            control_flow_errors=list(run.graph_errors),
            execution_order=list(run.execution_order),
        )
        for agent_id in run.execution_order:
            proposal = run.proposals[agent_id]
            validation = self._validate(proposal)
            step.records.append(
                ProposalRecord(
                    agent_id=agent_id,
                    proposal=asdict(proposal),
                    validation={
                        "accepted": validation.accepted,
                        "payload": dict(validation.payload),
                        "violations": [asdict(v) for v in validation.violations],
                    },
                )
            )
        if self.city is not None:
            inferred_time = query_time_h
            if inferred_time is None:
                value = observation.get("current_time")
                if isinstance(value, (int, float)):
                    inferred_time = float(value)
            registry, errors = self.city.build_registry(
                flow_tensors=flow_tensors,
                max_edges=max_traffic_edges,
                query_time_h=inferred_time,
            )
            step.geo_graphs = registry.all()
            step.geo_errors = list(errors)
        return step

    def topology_registry(
        self,
        *,
        flow_tensors: dict[str, object] | None = None,
        max_edges: int = 80,
        query_time_h: float | None = None,
    ) -> TopologyRegistry:
        """Standalone registry build (no agent run) for visualization."""
        registry = TopologyRegistry()
        if self.city is not None:
            built, _ = self.city.build_registry(
                flow_tensors=flow_tensors, max_edges=max_edges, query_time_h=query_time_h
            )
            for graph in built.all():
                registry.register(graph)
        registry.register(self.root_agent.control_flow_graph())
        return registry

    def _validate(self, proposal: AgentProposal) -> TwinResult:
        action = TwinAction(
            action_type=proposal.proposal_type,
            payload=proposal.payload,
            proposed_by=proposal.agent_name,
        )
        return self.validator.validate_action(action)
