# VehicleTwin Design

## Purpose

`VehicleTwin` is the white-box vehicle-state twin. It applies deterministic
state transitions after other components have already provided the required
inputs.

It tracks:

- current node;
- current simulation time;
- SOC;
- battery capacity;
- target SOC value carried as state;
- current mode (`idle`, `travelling`, `dwelling`, `queued`, `charging`,
  `inactive`);
- simple transition history.

## Hard Boundary

`VehicleTwin` must not contain uncertainty or behavioral policy logic.

It does not:

- sample departure time;
- predict destination;
- predict dwell duration;
- decide whether charging is needed;
- choose target SOC;
- choose charging station;
- choose fast or slow charging;
- tune user preference weights.

Those responsibilities belong to specialist agent nodes such as
`TripForecastAgent`, `DwellForecastAgent`, `ChargeNeedAgent`,
`TargetSocAgent`, and `StationDecisionAgent`.

## Interaction Contract

Agents and other twins provide already-decided values:

```text
TripInstruction(destination, departure time, travel time, travel energy, dwell time)
  -> VehicleTwin.apply_trip()
  -> deterministic node/time/SOC update

ChargeEvent(final SOC, station node, end time)
  -> VehicleTwin.apply_charge_event()
  -> deterministic post-charge state update
```

## Current Implementation

Implemented files:

- `smart_grid_core/twins/vehicle.py`
- `smart_grid_core/tools/vehicle_state_check.py`

The current implementation is deliberately small and does not replace
`online_event_sim.py`; it gives the refactor a white-box vehicle state primitive
that can be wired into the scenario runner after parity checks.
