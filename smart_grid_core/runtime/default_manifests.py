from __future__ import annotations

from .manifest import AgentManifest


def default_agent_manifests() -> list[AgentManifest]:
    return [
        AgentManifest(
            agent_id="trip_forecast_agent",
            agent_type="TripForecastAgent",
            consumes=("vehicle_state", "route_state"),
            produces=("trip_forecast",),
        ),
        AgentManifest(
            agent_id="station_decision_agent",
            agent_type="StationDecisionAgent",
            consumes=("vehicle_state", "reachable_stations", "station_price", "station_resource_state"),
            produces=("station_decision",),
        ),
        AgentManifest(
            agent_id="grid_friendly_agent",
            agent_type="GridFriendlyAgent",
            consumes=("station_price", "renewable_state", "base_load_state"),
            produces=("grid_friendly_strategy",),
        ),
        AgentManifest(
            agent_id="price_policy_agent",
            agent_type="PricePolicyAgent",
            consumes=("station_price", "base_load_state"),
            produces=("price_policy",),
        ),
        AgentManifest(
            agent_id="queue_forecast_agent",
            agent_type="QueueForecastAgent",
            consumes=("station_resource_state", "charge_event"),
            produces=("queue_forecast",),
        ),
        AgentManifest(
            agent_id="charging_mode_agent",
            agent_type="ChargingModeAgent",
            consumes=("vehicle_state", "station_resource_state", "station_price"),
            produces=("charging_mode",),
        ),
        AgentManifest(
            agent_id="charge_need_agent",
            agent_type="ChargeNeedAgent",
            consumes=("vehicle_state", "trip_forecast"),
            produces=("charge_need",),
        ),
        AgentManifest(
            agent_id="target_soc_agent",
            agent_type="TargetSocAgent",
            consumes=("vehicle_state", "trip_forecast", "station_price"),
            produces=("target_soc",),
        ),
        AgentManifest(
            agent_id="llm_advisor_agent",
            agent_type="LLMAdvisorAgent",
            consumes=("runtime_context", "specialist_proposals"),
            produces=("llm_advice",),
            optional_inputs=("vehicle_state", "reachable_stations", "station_price", "station_resource_state"),
            metadata={
                "agent_family": "llm",
                "implementation": "placeholder_or_human_until_api_provider_is_attached",
            },
        ),
    ]
