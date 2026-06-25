from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..base import AgentProposal, SpecialistAgent, get_observation_value
from ...assets import AssetRegistry


class TripForecastAgent(SpecialistAgent):
    """Forecasts destination distribution from existing transition matrices.

    This agent owns uncertainty. The TrafficTwin only applies route mechanics.
    """

    def __init__(self, transition_matrices: Sequence[Any] | None = None):
        super().__init__("trip_forecast_agent", "trip_forecast")
        self.transition_matrices = transition_matrices

    @classmethod
    def from_registry(cls, registry: AssetRegistry, asset_key: str = "state_transition_default") -> "TripForecastAgent":
        import pandas as pd

        path = registry.path(asset_key)
        matrices = []
        for hour in range(24):
            frame = pd.read_excel(path, sheet_name=f"period_{hour + 1}", header=None)
            matrices.append(frame.values.astype(float))
        return cls(matrices)

    def propose(self, observation: Mapping[str, Any]) -> AgentProposal:
        current_node = int(get_observation_value(observation, "current_node", 0))
        current_time = float(get_observation_value(observation, "current_time", 0.0))
        hour = int(current_time) % 24
        if self.transition_matrices is None:
            return self.proposal(
                {
                    "source": "missing_transition_matrix",
                    "current_node": current_node,
                    "hour": hour,
                    "destination_probabilities": {},
                },
                confidence=0.0,
                rationale="transition matrices were not loaded",
            )
        row = self.transition_matrices[hour][current_node]
        total = float(row.sum())
        if total <= 0:
            probabilities = {}
        else:
            probabilities = {int(index): float(value / total) for index, value in enumerate(row) if value > 0}
        return self.proposal(
            {
                "source": "state_transition_matrix",
                "current_node": current_node,
                "hour": hour,
                "destination_probabilities": probabilities,
            },
            confidence=1.0,
            rationale="reused existing hourly transition matrix",
        )


class QueueForecastAgent(SpecialistAgent):
    def __init__(self):
        super().__init__("queue_forecast_agent", "queue_forecast")

    def propose(self, observation: Mapping[str, Any]) -> AgentProposal:
        station_state = observation.get("station_resource_state", {})
        fast_queue = int(station_state.get("fast_queue_len", 0))
        slow_queue = int(station_state.get("slow_queue_len", 0))
        return self.proposal(
            {
                "fast_queue_len": fast_queue,
                "slow_queue_len": slow_queue,
                "forecast_horizon": "current_state_only",
            },
            confidence=0.6,
            rationale="placeholder queue expert using current station state only",
        )
