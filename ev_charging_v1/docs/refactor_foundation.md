# Smart Grid Refactor Foundation

## Goal

The refactor keeps all existing functions and data while separating the system
into:

- twin core: state mapping, mechanism calculation, constraint validation,
  deterministic replay;
- agent layer: forecast information, policy proposals, parameter suggestions,
  result explanation;
- interaction orchestrator: structured proposal, validation, execution log.

Prediction should not live inside the twin core. It should be provided by
specialized agents, preferably by reusing existing matrices, cached results, and
optimization outputs before any retraining.

## Non-Negotiable Rules

1. Existing legacy scripts remain runnable during the refactor.
2. Existing data, precomputed tables, optimized policies, and event outputs are
   treated as reusable assets.
3. Agents cannot directly mutate SOC, queues, road flows, voltages, or load
   curves.
4. Agents emit structured proposals only.
5. Twins validate proposals before execution.
6. LLM integration is a standalone agent behind the same proposal interface, not
   an internal dependency of another specialist and not a replacement for
   deterministic mechanism code.

## First Additive Package

`smart_grid_core` is the new additive package. It currently contains:

- `assets.py`: registry for reusable data/model artifacts;
- `schemas.py`: common event, forecast, vehicle, station, and decision records;
- `agents/base.py`: generic agent proposal interface;
- `agents/precomputed.py`: wrappers for existing precomputed forecasts/policies;
- `twins/base.py`: validation primitives for twin execution;
- `legacy_adapters/events.py`: conversion from legacy charge events to the
  canonical `ChargeEvent` schema;
- `tools/asset_report.py`: asset existence report for the current project;
- `parity.py` and `tools/parity_report.py`: legacy event replay and baseline
  load comparison;
- `charging.py`, `twins/station.py`, and `tools/station_replay_check.py`:
  deterministic station-resource twin and replay inspection;
- `twins/traffic.py` and `tools/traffic_lookup_check.py`: deterministic
  precomputed route-time and route-energy lookup;
- `twins/grid.py` and `tools/grid_lookup_check.py`: deterministic grid-side
  price, renewable, and base-load state lookup;
- `twins/vehicle.py` and `tools/vehicle_state_check.py`: deterministic vehicle
  state transitions for trips, dwell, queue, and charge completion;
- `runtime/`: T-shaped hot-plug runtime layer that matches twins and agents by
  manifest capabilities without understanding twin topology internals;
- `agents/specialists.py` and `tools/specialist_agent_check.py`: baseline
  non-LLM expert agents for uncertainty-bearing decisions;
- `agents/llm.py` and `tools/llm_agent_check.py`: standalone LLM-style advisor
  plus a human-in-the-loop stand-in for testing the same interaction path;
- `visualization/dashboard.py` and `tools/visualization_snapshot.py`: runtime
  inspection snapshot for checking twin state, agent proposals, LLM/human
  review, asset status, and parity results;
- `twins/topology.py`, `visualization/topology_dashboard.py`, and
  `tools/topology_canvas.py`: twin-owned topology export, runtime information
  flow rendering, and Human-as-LLM interaction on top of the graph;
- `orchestrator.py`: minimal agent-to-twin interaction loop.

Current parity results are recorded in `docs/parity_results.md`.

No legacy code is changed by this first step.

## Existing Assets To Reuse First

- `N_N.npy`: road flow tensor.
- `people.npy`: demand scaling profile.
- `state_transition_matrices_*.xlsx`: trip transition forecasts.
- `od_matrices_*.xlsx`: OD estimates.
- `dynamic_price.npy`: existing dynamic price strategy.
- `road_time_table_day*.npy`: travel time caches.
- `delta_Q_table_day*.npy`: driving energy caches.
- `eps_table_day*.npy`: station price caches.
- `charge_*_table.npy`: charge lookup tables.
- `result/Curve_positons_iter*.npy`: optimization-derived policy candidates.
- `output_*/charge_events_*.pkl`: event replay baselines.
- `output_*/predicted_load_*.npy`: station load baselines.

## Migration Order

1. Keep legacy entrypoints untouched.
2. Register all reusable assets in `AssetRegistry`.
3. Wrap OD/transition matrices as forecast agents.
4. Wrap dynamic price and optimization outputs as strategy agents.
5. Define canonical `ChargeEvent` and migrate `load_simulator.py` to read it.
6. Split event simulation into vehicle, traffic, station, and grid twins.
7. Replace copied `run_jun30.py`, `run_jul1.py`, and `run_jul2.py` with a
   scenario runner after parity checks.
8. Keep LLM/human review as a standalone advisory agent while deterministic
   parity remains the foundation.
