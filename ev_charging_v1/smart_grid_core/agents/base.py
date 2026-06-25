from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class AgentProposal:
    """Structured suggestion emitted by an intelligent agent.

    A proposal is not state. The twin layer must validate it before execution.
    """

    agent_name: str
    proposal_type: str
    payload: Mapping[str, Any]
    confidence: float | None = None
    rationale: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


class AgentPolicy(Protocol):
    name: str

    def propose(self, observation: Mapping[str, Any]) -> AgentProposal:
        """Return a structured proposal without mutating twin state."""


class Tool(Protocol):
    """Stateless lookup or calculation invoked by a subagent.

    Tools never produce AgentProposals. They return raw values that
    subagents incorporate into their own proposals. Distinguishing tools
    from agents lets RootAgent build a control-flow graph where tools
    are always leaves.
    """

    name: str

    def call(self, observation: Mapping[str, Any]) -> Any:
        """Return a value derived from the observation; no side effects."""


class RuleBasedAgentPolicy:
    """Temporary base for non-LLM agents.

    The first refactor should wrap existing matrices, cached outputs, and
    classical optimization results here. LLM implementations can later satisfy
    the same protocol.
    """

    name = "rule_based_agent"

    def propose(self, observation: Mapping[str, Any]) -> AgentProposal:
        return AgentProposal(
            agent_name=self.name,
            proposal_type="noop",
            payload={},
            confidence=1.0,
            rationale="placeholder rule-based proposal",
        )


def get_observation_value(source: Mapping[str, Any], key: str, default: Any = None) -> Any:
    """Look up `key` in observation, falling back through nested state/vehicle_state.

    Subagents share this lookup so that they accept observations whether the
    caller flattened state into the top dict or left it nested.
    """
    if key in source:
        return source[key]
    state = source.get("state")
    if isinstance(state, Mapping) and key in state:
        return state[key]
    vehicle = source.get("vehicle_state")
    if isinstance(vehicle, Mapping) and key in vehicle:
        return vehicle[key]
    return default


@dataclass(frozen=True)
class SpecialistAgent:
    """Base for rule-baseline subagents.

    Provides a uniform `proposal()` factory; concrete subagents implement
    `propose(observation) -> AgentProposal`.
    """

    name: str
    proposal_type: str

    def proposal(
        self, payload: Mapping[str, Any], *, confidence: float | None = None, rationale: str = ""
    ) -> AgentProposal:
        return AgentProposal(
            agent_name=self.name,
            proposal_type=self.proposal_type,
            payload=dict(payload),
            confidence=confidence,
            rationale=rationale,
        )
