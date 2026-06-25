from __future__ import annotations

from typing import Iterable

from ..agents import (
    ChargeNeedAgent,
    ChargingModeAgent,
    GridFriendlyAgent,
    LLMAdvisorAgent,
    PricePolicyAgent,
    QueueForecastAgent,
    RootAgent,
    StationDecisionAgent,
    SubagentBinding,
    TargetSocAgent,
    ToolBinding,
    TripForecastAgent,
)
from ..agents.subagents import SUBAGENT_GROUPS
from ..assets import AssetRegistry
from .default_manifests import default_agent_manifests
from .manifest import AgentManifest


_AGENT_TYPE_TO_GROUP: dict[str, str] = {
    agent_type: group
    for group, agent_types in SUBAGENT_GROUPS.items()
    for agent_type in agent_types
}


def _instantiate_subagent(manifest: AgentManifest, registry: AssetRegistry):
    agent_type = manifest.agent_type
    if agent_type == "TripForecastAgent":
        try:
            return TripForecastAgent.from_registry(registry)
        except Exception:
            return TripForecastAgent()
    if agent_type == "ChargeNeedAgent":
        return ChargeNeedAgent()
    if agent_type == "TargetSocAgent":
        return TargetSocAgent()
    if agent_type == "StationDecisionAgent":
        return StationDecisionAgent()
    if agent_type == "ChargingModeAgent":
        return ChargingModeAgent()
    if agent_type == "GridFriendlyAgent":
        return GridFriendlyAgent()
    if agent_type == "PricePolicyAgent":
        return PricePolicyAgent(registry=registry)
    if agent_type == "QueueForecastAgent":
        return QueueForecastAgent()
    if agent_type == "LLMAdvisorAgent":
        return LLMAdvisorAgent()
    raise ValueError(f"unknown agent type: {agent_type!r}")


def build_default_root_agent(
    registry: AssetRegistry,
    *,
    manifests: Iterable[AgentManifest] | None = None,
    agent_id: str = "ev_charging_root_agent",
) -> RootAgent:
    """Wire the standard EV-charging root agent from the manifest list.

    Subagent grouping comes from `SUBAGENT_GROUPS`; tools are registered
    purely for visibility in the control-flow graph and do not run on
    their own — subagents call them via their constructors.
    """
    manifests = list(manifests) if manifests is not None else list(default_agent_manifests())
    bindings: list[SubagentBinding] = []
    for manifest in manifests:
        policy = _instantiate_subagent(manifest, registry)
        group = _AGENT_TYPE_TO_GROUP.get(manifest.agent_type, "advisor")
        bindings.append(SubagentBinding(policy=policy, manifest=manifest, group=group))

    tools = [
        ToolBinding(
            tool_id="precomputed_forecast_tool",
            description="Lookup of registered forecast assets (no retraining).",
        ),
        ToolBinding(
            tool_id="static_policy_tool",
            description="Lookup of registered optimized policy arrays.",
        ),
    ]
    return RootAgent(agent_id=agent_id, subagents=bindings, tools=tools)
