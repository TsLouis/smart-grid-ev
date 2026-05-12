from .advisor import HumanLLMAgent, LLMAdvisorAgent
from .decision import ChargeNeedAgent, StationDecisionAgent, TargetSocAgent
from .perception import QueueForecastAgent, TripForecastAgent
from .policy import ChargingModeAgent, GridFriendlyAgent, PricePolicyAgent

__all__ = [
    "ChargeNeedAgent",
    "ChargingModeAgent",
    "GridFriendlyAgent",
    "HumanLLMAgent",
    "LLMAdvisorAgent",
    "PricePolicyAgent",
    "QueueForecastAgent",
    "StationDecisionAgent",
    "TargetSocAgent",
    "TripForecastAgent",
]


SUBAGENT_GROUPS: dict[str, tuple[str, ...]] = {
    "perception": ("TripForecastAgent", "QueueForecastAgent"),
    "decision": ("ChargeNeedAgent", "TargetSocAgent", "StationDecisionAgent"),
    "policy": ("ChargingModeAgent", "GridFriendlyAgent", "PricePolicyAgent"),
    "advisor": ("LLMAdvisorAgent", "HumanLLMAgent"),
}
