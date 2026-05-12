# Target Module Tree

```text
EV_charging/
  smart_grid_core/
    assets.py
    charging.py
    schemas.py
    orchestrator.py
    parity.py
    runtime/
      manifest.py
      registry.py
      default_manifests.py
    visualization/
      dashboard.py
      topology_dashboard.py
    agents/
      base.py
      llm.py
      precomputed.py
      specialists.py
      demand_forecast.py        # later
      trip_forecast.py          # later
      charge_decision.py        # later
      station_decision.py       # later
      grid_strategy.py          # later
    twins/
      base.py
      grid.py
      station.py
      topology.py
      traffic.py                # later
      vehicle.py
      station.py                # later
      grid.py                   # later
      scenario.py               # later
    legacy_adapters/
      events.py
      dijkstra_adapter.py       # later
      load_adapter.py           # later
    tools/
      asset_report.py
      grid_lookup_check.py
      parity_report.py
      runtime_manifest_check.py
      topology_canvas.py
      visualization_snapshot.py
      llm_agent_check.py
      specialist_agent_check.py
      station_replay_check.py
      traffic_lookup_check.py
      vehicle_state_check.py
  docs/
    refactor_foundation.md
    asset_reuse_policy.md
    llm_agent_design.md
    refactor_tree.md
```

Legacy directories remain in place until parity checks pass.
