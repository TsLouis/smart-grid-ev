# Asset Reuse Policy

The refactor should avoid retraining or recomputing unless an asset is missing,
incompatible with the scenario, or explicitly marked stale.

## Asset Classes

| Class | Examples | Refactor Role |
| --- | --- | --- |
| Raw data | `N_N.npy`, `people.npy`, `grid_RX.xlsx` | loaded by twins and forecast agents |
| Forecast products | `od_matrices_*.xlsx`, `state_transition_matrices_*.xlsx` | served by forecast agents |
| Precompute caches | `road_time_table_day*.npy`, `delta_Q_table_day*.npy`, `eps_table_day*.npy` | reused by traffic/grid twins |
| Charge lookup tables | `charge_soc_grid.npy`, `charge_eta_grid.npy`, `charge_time_table.npy`, `charge_energy_table.npy` | reused by station twin |
| Optimization outputs | `Curve_positons_iter*.npy`, `CS_Load_inter*.npy` | served by strategy agents |
| Legacy baselines | `charge_events_*.pkl`, `predicted_load_*.npy` | replay and parity checks |

## Default Behavior

1. Look up asset in `AssetRegistry`.
2. If present, reuse it.
3. If missing, return a structured missing-asset proposal or warning.
4. Only recompute through an explicit pipeline command.
5. Never silently retrain from inside a twin.

## LLM Placement

LLM adapters should initially be limited to:

- scenario text understanding;
- parameter adjustment proposals;
- policy explanation;
- anomaly attribution;
- report generation.

Numerical forecasts should first reuse the existing OD matrices, transition
matrices, dynamic price arrays, and optimization outputs.
