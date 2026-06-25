# TrafficTwin Design

## Purpose

`TrafficTwin` is the white-box traffic and route-state twin. It exposes
deterministic route information already produced by the mechanism layer:

- node-to-station travel time;
- node-to-station driving energy;
- station reachability;
- route-table metadata.

## Hard Boundary

`TrafficTwin` must not contain uncertainty or policy logic.

It does not:

- predict OD demand;
- predict destination;
- predict future congestion beyond loaded input tables;
- rank charging stations;
- decide where a vehicle should charge;
- decide whether a vehicle should start a trip.

Those responsibilities belong to specialist agent nodes such as
`DemandForecastAgent`, `TripForecastAgent`, and `StationDecisionAgent`.

## Current Implementation

Implemented files:

- `smart_grid_core/twins/traffic.py`
- `smart_grid_core/tools/traffic_lookup_check.py`

The first implementation is `PrecomputedTrafficTwin`, backed by existing
`road_time_table_day*.npy` and `delta_Q_table_day*.npy` assets. This preserves
the current model outputs and avoids recomputing routes during early refactor
steps.

## Later Adapter

A later `LegacyDijkstraTrafficTwin` can wrap `dijkstra_numba.py` directly for
on-demand route computation. It should still remain white-box and deterministic.
