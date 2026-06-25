from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ..base import AgentProposal
from ...assets import AssetRegistry


class StaticPolicyAgent:
    """Reuse an optimized policy array instead of recomputing.

    Wrapped as an AgentPolicy for pipeline uniformity, but structurally
    a tool: a registered asset path is the entire output.
    """

    def __init__(self, name: str, registry: AssetRegistry, asset_key: str):
        self.name = name
        self.registry = registry
        self.asset_key = asset_key

    def propose(self, observation: Mapping[str, Any]) -> AgentProposal:
        asset_path: Path = self.registry.path(self.asset_key)
        return AgentProposal(
            agent_name=self.name,
            proposal_type="static_policy",
            payload={
                "asset_key": self.asset_key,
                "path": str(asset_path),
                "exists": asset_path.exists(),
                "observation": dict(observation),
            },
            confidence=1.0 if asset_path.exists() else 0.0,
            rationale="reuse existing optimized policy before recomputing",
        )
