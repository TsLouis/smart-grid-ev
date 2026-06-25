from __future__ import annotations

from typing import Any, Mapping

from ..base import AgentProposal, SpecialistAgent, get_observation_value
from ...assets import AssetRegistry


class ChargingModeAgent(SpecialistAgent):
    def __init__(self, slow_dwell_threshold_h: float = 6.0, night_start_h: float = 22.0):
        super().__init__("charging_mode_agent", "charging_mode")
        self.slow_dwell_threshold_h = slow_dwell_threshold_h
        self.night_start_h = night_start_h

    def propose(self, observation: Mapping[str, Any]) -> AgentProposal:
        dwell_time_h = float(observation.get("dwell_time_h", 0.0))
        current_time = float(get_observation_value(observation, "current_time", 0.0))
        mode = "slow" if dwell_time_h >= self.slow_dwell_threshold_h or current_time >= self.night_start_h else "fast"
        return self.proposal(
            {
                "mode_preference": mode,
                "dwell_time_h": dwell_time_h,
                "current_time": current_time,
            },
            confidence=0.75,
            rationale="rule baseline charging mode proposal",
        )


class GridFriendlyAgent(SpecialistAgent):
    def __init__(self):
        super().__init__("grid_friendly_agent", "grid_friendly_strategy")

    def propose(self, observation: Mapping[str, Any]) -> AgentProposal:
        renewable = observation.get("renewable_state", {})
        base_load = observation.get("base_load_state", {})
        pv = float(renewable.get("pv", 0.0))
        wind = float(renewable.get("wind", 0.0))
        load = sum(float(base_load.get(key, 0.0)) for key in ("residential", "commercial", "work"))
        renewable_ratio = (pv + wind) / load if load > 0 else 0.0
        grid_weight = 0.35 if renewable_ratio < 0.5 else 0.15
        return self.proposal(
            {
                "grid_weight": grid_weight,
                "renewable_ratio": renewable_ratio,
                "recommended_bias": "avoid_peak" if grid_weight >= 0.35 else "use_renewable_window",
            },
            confidence=0.7,
            rationale="rule baseline grid-friendly strategy proposal",
        )


class PricePolicyAgent(SpecialistAgent):
    def __init__(self, registry: AssetRegistry | None = None, asset_key: str = "dynamic_price"):
        super().__init__("price_policy_agent", "price_policy")
        self.registry = registry
        self.asset_key = asset_key

    def propose(self, observation: Mapping[str, Any]) -> AgentProposal:
        if self.registry is None:
            return self.proposal(
                {"source": "none", "asset_key": self.asset_key, "exists": False},
                confidence=0.0,
                rationale="no asset registry attached",
            )
        path = self.registry.path(self.asset_key)
        return self.proposal(
            {
                "source": "registered_asset",
                "asset_key": self.asset_key,
                "path": str(path),
                "exists": path.exists(),
            },
            confidence=1.0 if path.exists() else 0.0,
            rationale="reuse existing dynamic price or optimized policy asset",
        )
