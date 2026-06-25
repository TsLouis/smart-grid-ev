# Legacy Parity Results

These checks replay legacy `charge_events_*.pkl` files through the new
canonical `ChargeEvent` adapter and aggregate station load again. The replayed
load is compared against the legacy `predicted_load_*.npy` baseline.

Run from `~/Smart_Grid_Project/EV_charging`:

```bash
../.venv/bin/python -m smart_grid_core.tools.parity_report --root . --scenario jun30
../.venv/bin/python -m smart_grid_core.tools.parity_report --root . --scenario jul1
../.venv/bin/python -m smart_grid_core.tools.parity_report --root . --scenario jul2
../.venv/bin/python -m smart_grid_core.tools.parity_report --root . --scenario jun30_no_queue
```

## Current Results

| Scenario | Events | Shape | Max Abs Error | Mean Abs Error | Baseline Sum | Replay Sum |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `jun30` | 6343 | `(7, 288)` | 0.000000 | 0.000000 | 3518636.974738 | 3518636.974738 |
| `jul1` | 6856 | `(7, 288)` | 0.000000 | 0.000000 | 3868143.052811 | 3868143.052811 |
| `jul2` | 6727 | `(7, 288)` | 0.000000 | 0.000000 | 3672696.777269 | 3672696.777269 |
| `jun30_no_queue` | 7689 | `(7, 288)` | 0.000000 | 0.000000 | 3799999.435400 | 3799999.435400 |

## Meaning

The new event schema and replay aggregation can exactly reproduce existing
legacy station-load outputs for the checked scenarios. This gives us a safety
check before splitting the legacy simulation into vehicle, traffic, charging
station, and grid twins.
