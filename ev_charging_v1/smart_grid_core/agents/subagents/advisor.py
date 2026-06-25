from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..base import AgentProposal


def _normalize_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


@dataclass(frozen=True)
class LLMAdvisorAgent:
    """Standalone LLM-style advisor agent.

    This class is intentionally a normal agent implementation rather than a
    dependency hidden inside another specialist. A future API-backed LLM can
    replace the response provider while keeping the same proposal contract.
    """

    name: str = "llm_advisor_agent"
    provider: str = "placeholder"

    def propose(self, observation: Mapping[str, Any]) -> AgentProposal:
        context_keys = tuple(sorted(str(key) for key in observation.keys()))
        specialist_proposals = observation.get("specialist_proposals", [])
        if isinstance(specialist_proposals, Sequence) and not isinstance(specialist_proposals, (str, bytes)):
            proposal_count = len(specialist_proposals)
        else:
            proposal_count = 0

        summary = _normalize_text(
            observation.get("summary"),
            "LLM advisor slot is wired; no external LLM provider has been attached yet.",
        )
        return AgentProposal(
            agent_name=self.name,
            proposal_type="llm_advice",
            payload={
                "source": self.provider,
                "summary": summary,
                "strategy_review": _normalize_text(observation.get("strategy_review"), "pending"),
                "explanation": _normalize_text(observation.get("explanation"), "pending"),
                "observed_context_keys": context_keys,
                "specialist_proposal_count": proposal_count,
            },
            confidence=None,
            rationale="standalone LLM agent proposal; it does not mutate twin or runtime state",
            metadata={"agent_family": "llm", "interactive": False},
        )


@dataclass(frozen=True)
class HumanLLMAgent:
    """Human-in-the-loop implementation of the same LLM advisor slot."""

    advice: str
    name: str = "human_llm_agent"

    def propose(self, observation: Mapping[str, Any]) -> AgentProposal:
        return AgentProposal(
            agent_name=self.name,
            proposal_type="llm_advice",
            payload={
                "source": "human",
                "summary": self.advice,
                "strategy_review": _normalize_text(observation.get("strategy_review"), "provided_by_human"),
                "explanation": _normalize_text(observation.get("explanation"), self.advice),
                "observed_context_keys": tuple(sorted(str(key) for key in observation.keys())),
            },
            confidence=None,
            rationale="human response wrapped as a standalone LLM-agent proposal",
            metadata={"agent_family": "llm", "interactive": True},
        )
