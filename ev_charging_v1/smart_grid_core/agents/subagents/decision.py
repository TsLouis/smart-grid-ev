from __future__ import annotations

from typing import Any, Mapping

from ..base import AgentProposal, SpecialistAgent, get_observation_value


class ChargeNeedAgent(SpecialistAgent):
    def __init__(self, threshold_soc: float = 0.3):
        super().__init__("charge_need_agent", "charge_need")
        self.threshold_soc = threshold_soc

    def propose(self, observation: Mapping[str, Any]) -> AgentProposal:
        soc = float(get_observation_value(observation, "soc", 1.0))
        next_trip_energy_kwh = float(observation.get("next_trip_energy_kwh", 0.0))
        capacity = float(get_observation_value(observation, "battery_capacity_kwh", 61.4))
        projected_soc = soc - (next_trip_energy_kwh / capacity if capacity > 0 else 0.0)
        should_charge = projected_soc <= self.threshold_soc
        return self.proposal(
            {
                "should_charge": should_charge,
                "current_soc": soc,
                "projected_soc": projected_soc,
                "threshold_soc": self.threshold_soc,
            },
            confidence=0.8,
            rationale="rule baseline; replaceable by learned or LLM-backed expert later",
        )


class TargetSocAgent(SpecialistAgent):
    def __init__(self, default_target_soc: float = 0.9, high_risk_target_soc: float = 0.95):
        super().__init__("target_soc_agent", "target_soc")
        self.default_target_soc = default_target_soc
        self.high_risk_target_soc = high_risk_target_soc

    def propose(self, observation: Mapping[str, Any]) -> AgentProposal:
        charge_need = observation.get("charge_need", {})
        projected_soc = float(charge_need.get("projected_soc", get_observation_value(observation, "soc", 1.0)))
        target = self.high_risk_target_soc if projected_soc < 0.2 else self.default_target_soc
        return self.proposal(
            {"target_soc": target, "projected_soc": projected_soc},
            confidence=0.75,
            rationale="rule baseline target SOC proposal",
        )


class StationDecisionAgent(SpecialistAgent):
    def __init__(self, weights: Mapping[str, float] | None = None):
        super().__init__("station_decision_agent", "station_decision")
        self.weights = dict(weights or {"time": 0.55, "energy": 0.15, "price": 0.30})

    def propose(self, observation: Mapping[str, Any]) -> AgentProposal:
        routes = observation.get("reachable_stations", [])
        prices = observation.get("station_prices", {})
        ranked = []
        for route in routes:
            station_index = int(route["station_index"])
            price = float(prices.get(station_index, prices.get(str(station_index), 0.0)))
            score = (
                self.weights.get("time", 0.0) * float(route.get("travel_time_h", 0.0))
                + self.weights.get("energy", 0.0) * float(route.get("travel_energy_kwh", 0.0))
                + self.weights.get("price", 0.0) * price
            )
            ranked.append({"station_index": station_index, "score": score, "price": price, **dict(route)})
        ranked.sort(key=lambda item: item["score"])
        return self.proposal(
            {"ranked_stations": ranked, "weights": self.weights},
            confidence=0.8 if ranked else 0.0,
            rationale="rule baseline station ranking from twin observations",
        )
