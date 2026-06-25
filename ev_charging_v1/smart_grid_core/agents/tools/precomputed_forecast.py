from __future__ import annotations

from typing import Any, Mapping

from ..base import AgentProposal
from ...assets import AssetRegistry


class PrecomputedForecastAgent:
    """Reuse an existing forecast artifact instead of retraining.

    Wrapped as an AgentPolicy so it can sit in the same orchestration
    pipeline, but it is structurally a tool: pure asset lookup, no
    learned state.
    """

    def __init__(self, name: str, registry: AssetRegistry, asset_key: str):
        self.name = name
        self.registry = registry
        self.asset_key = asset_key

    def propose(self, observation: Mapping[str, Any]) -> AgentProposal:
        asset = self.registry.get(self.asset_key)
        return AgentProposal(
            agent_name=self.name,
            proposal_type="precomputed_forecast",
            payload={
                "asset_key": asset.key,
                "path": str(self.registry.path(asset.key)),
                "category": asset.category,
                "role": asset.role,
                "loader": asset.loader,
                "exists": self.registry.exists(asset.key),
                "observation": dict(observation),
            },
            confidence=1.0 if self.registry.exists(asset.key) else 0.0,
            rationale="reuse existing forecast artifact; do not retrain by default",
        )
