# GridTwin Design

## Purpose

`GridTwin` is the white-box grid-side state twin. It exposes deterministic,
time-indexed grid data loaded from existing project assets:

- station dynamic price table;
- photovoltaic profile;
- wind profile;
- residential, commercial, and work-area base-load profiles;
- distribution-line data as registered assets.

## Hard Boundary

`GridTwin` must not contain uncertainty or strategy logic.

It does not:

- decide grid-friendly charging weights;
- optimize prices;
- decide demand response actions;
- recommend charging stations;
- infer grid risk from text or policy context;
- forecast future renewable output beyond loaded profiles.

Those responsibilities belong to specialist agent nodes such as
`GridFriendlyAgent`, `PricePolicyAgent`, `RenewableForecastAgent`, and
`ScenarioAgent`.

## Current Implementation

Implemented files:

- `smart_grid_core/twins/grid.py`
- `smart_grid_core/tools/grid_lookup_check.py`

The first implementation wraps existing arrays and preserves current dynamic
price and profile data. Later versions can add a deterministic power-flow
adapter around `powerflow.py`, while keeping price optimization and uncertainty
outside the twin.
