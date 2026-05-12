# StationTwin Design

## Purpose

`StationTwin` is the white-box digital twin for charging-station resources. It
tracks deterministic resource state only:

- fast and slow pile counts;
- occupied piles;
- waiting queues;
- active charging events;
- completed charging events;
- deterministic power sequence generation from already-approved parameters.

## Hard Boundary

`StationTwin` must not contain uncertainty or policy logic.

It does not:

- forecast future queue time;
- choose fast versus slow charging;
- choose target SOC;
- decide whether a vehicle should charge;
- rank stations;
- tune user preference weights;
- interpret weather, activities, or policy text.

Those responsibilities belong to specialist agent nodes such as
`QueueForecastAgent`, `ChargingModeAgent`, `TargetSocAgent`,
`StationDecisionAgent`, and `GridFriendlyAgent`.

## Interaction Contract

Agents produce structured proposals. The orchestrator validates and converts an
accepted proposal into a `ChargeRequest`. `StationTwin` then applies the request
deterministically and emits a `ChargeEvent`.

```text
Observation
  -> AgentProposal
  -> validated ChargeRequest
  -> StationTwin.submit_request()
  -> ChargeEvent
```

## Current Implementation

Implemented files:

- `smart_grid_core/charging.py`
- `smart_grid_core/twins/station.py`
- `smart_grid_core/tools/station_replay_check.py`

The current charge curve is constant-power and deterministic. Existing legacy
events can be replayed into station histories for parity inspection.
