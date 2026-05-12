# Runtime Layer Design

## Purpose

The runtime layer is the T-shaped middle layer between twins, agents, data
assets, events, and visualization.

It must support future multi-level and hot-pluggable twin/agent systems without
learning the internal topology logic of each scenario.

## Core Boundary

Topology belongs to twins.

The runtime layer does not know:

- road topology semantics;
- distribution-grid topology semantics;
- station-grid connection logic;
- path calculation;
- power-flow calculation;
- queue mechanics;
- vehicle behavior mechanics.

The runtime only knows manifests:

- what observations a twin provides;
- what actions a twin accepts;
- what events a twin emits;
- what visualization views it exposes;
- what agent slots it wants;
- what capabilities an agent produces.

## Principle

```text
Topology -> Twin
Orchestration -> Runtime Layer
Uncertainty -> Agent
```

## Hot-Plug Contract

A twin exposes a `TwinManifest`:

```text
twin_id
twin_type
level
parent_twin_id
provided_observations
accepted_actions
emitted_events
visualization_views
agent_slots
```

An agent exposes an `AgentManifest`:

```text
agent_id
agent_type
consumes
produces
optional_inputs
level
```

The runtime pairs them by capability names only. It does not inspect internal
graphs or model code.

## Current Implementation

Implemented files:

- `smart_grid_core/runtime/manifest.py`
- `smart_grid_core/runtime/registry.py`
- `smart_grid_core/runtime/default_manifests.py`
- `smart_grid_core/tools/runtime_manifest_check.py`
- `smart_grid_core/visualization/dashboard.py`
- `smart_grid_core/tools/visualization_snapshot.py`
- `smart_grid_core/twins/topology.py`
- `smart_grid_core/visualization/topology_dashboard.py`
- `smart_grid_core/tools/topology_canvas.py`

The current default manifest set declares four topology/mechanism twin groups,
one runtime-level T-layer manifest, and nine agent slots:

- `TrafficTwin`
- `GridTwin`
- `StationTwinGroup`
- `VehicleTwinGroup`
- `RuntimeLayer`
- `TripForecastAgent`
- `StationDecisionAgent`
- `GridFriendlyAgent`
- `PricePolicyAgent`
- `QueueForecastAgent`
- `ChargingModeAgent`
- `ChargeNeedAgent`
- `TargetSocAgent`
- `LLMAdvisorAgent`

This is only the wiring contract. The actual topology and mechanism logic stay
inside each twin.

`RuntimeLayer` is deliberately not a topology owner. Its `llm_advisor` slot is a
bridge for explanation, strategy review, and human/LLM interaction over runtime
context plus specialist proposals.

The first report dashboard is only a state bundle. The topology canvas is the
primary inspection surface: traffic and grid twins export topology graphs, the
runtime carries those graphs plus information-flow edges, and the browser
renders layers without owning topology semantics.

The topology canvas also includes a Human-as-LLM panel. The browser-side input
is wrapped as the same `llm_advice` proposal shape used by the standalone LLM
agent, so a future API provider can replace the human input without changing
the twin or runtime boundary.
