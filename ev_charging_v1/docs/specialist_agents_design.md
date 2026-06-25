# Specialist Agents Design

## Purpose

Specialist agents own uncertainty. They produce structured proposals but never
mutate twin state directly.

The first implementation is intentionally non-LLM and non-training:

- reuse existing matrices and assets where available;
- use simple rules as placeholders;
- keep the proposal schema stable so stronger models can be swapped in later.

## Current Agents

- `TripForecastAgent`
  - Uses existing transition matrices when loaded.
  - Produces `trip_forecast`.

- `ChargeNeedAgent`
  - Rule baseline using SOC and projected next-trip energy.
  - Produces `charge_need`.

- `TargetSocAgent`
  - Rule baseline target SOC proposal.
  - Produces `target_soc`.

- `ChargingModeAgent`
  - Rule baseline fast/slow preference from dwell time and time of day.
  - Produces `charging_mode`.

- `StationDecisionAgent`
  - Rule baseline station ranking from TrafficTwin/GridTwin observations.
  - Produces `station_decision`.

- `GridFriendlyAgent`
  - Rule baseline grid-friendly bias from renewable and base-load state.
  - Produces `grid_friendly_strategy`.

- `PricePolicyAgent`
  - Reuses registered price/optimization assets.
  - Produces `price_policy`.

- `QueueForecastAgent`
  - Placeholder using current station state only.
  - Produces `queue_forecast`.

- `LLMAdvisorAgent`
  - Standalone LLM-style advisor at the same agent layer as the specialists.
  - Produces `llm_advice` from runtime context and other expert proposals.
  - Current implementation is placeholder/API-ready; `HumanLLMAgent` can play
    the same role interactively before a provider is attached.

## Boundary

Agents can propose:

- destination probabilities;
- dwell estimates;
- charge need;
- target SOC;
- charging mode;
- station ranking;
- grid-friendly weights;
- price policy references;
- queue forecasts.
- LLM or human review/advice.

Agents cannot directly change:

- vehicle SOC;
- vehicle location;
- station queue;
- pile occupation;
- road flow;
- grid voltage or load state.

All proposals must go through runtime/orchestrator and then twin validation.
