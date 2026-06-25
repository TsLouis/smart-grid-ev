from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class AssetSpec:
    """A reusable data/model artifact from the legacy project.

    `region_scoped=True` marks assets whose values differ per district
    (grid profiles, traffic tables, station price caches). Region-scoped
    assets are resolved against `data/regions/{region_id}/` when the
    registry is bound to a region; non-scoped assets always resolve
    against the project root.
    """

    key: str
    path: str
    category: str
    role: str
    owner: str
    loader: str
    required: bool = False
    notes: str = ""
    region_scoped: bool = False


class AssetRegistry:
    """Central index for existing data, model outputs, and caches.

    The registry stores paths and metadata only. Loading happens in
    adapters so the twin core never silently retrains or recomputes.
    Bind a region with `for_region(region_id)` to redirect region-scoped
    asset paths into `data/regions/{region_id}/`.
    """

    def __init__(self, root: Path, assets: Iterable[AssetSpec], region_id: Optional[str] = None):
        self.root = Path(root)
        self.region_id = region_id
        self._assets: Dict[str, AssetSpec] = {asset.key: asset for asset in assets}

    def get(self, key: str) -> AssetSpec:
        return self._assets[key]

    def path(self, key: str) -> Path:
        spec = self.get(key)
        if self.region_id is not None and spec.region_scoped:
            return self.root / "data" / "regions" / self.region_id / Path(spec.path).name
        return self.root / spec.path

    def exists(self, key: str) -> bool:
        return self.path(key).exists()

    def list(self, category: Optional[str] = None, owner: Optional[str] = None) -> List[AssetSpec]:
        assets = list(self._assets.values())
        if category is not None:
            assets = [asset for asset in assets if asset.category == category]
        if owner is not None:
            assets = [asset for asset in assets if asset.owner == owner]
        return assets

    def missing_required(self) -> List[AssetSpec]:
        return [asset for asset in self._assets.values() if asset.required and not self.exists(asset.key)]

    def for_region(self, region_id: str) -> "AssetRegistry":
        return AssetRegistry(self.root, self._assets.values(), region_id=region_id)

    def region_scoped_specs(self) -> List[AssetSpec]:
        return [spec for spec in self._assets.values() if spec.region_scoped]


def default_asset_specs() -> List[AssetSpec]:
    """Known assets discovered in the current legacy project.

    These are treated as reusable model/data products. The refactor should wrap
    them first, not retrain or regenerate them by default.
    """

    return [
        AssetSpec(
            key="traffic_flow_tensor",
            path="Load Forecasting/N_N.npy",
            category="traffic",
            role="time-indexed road flow tensor",
            owner="traffic_twin",
            loader="numpy",
            required=True,
            region_scoped=True,
        ),
        AssetSpec(
            key="people_demand_profile",
            path="Load Forecasting/people.npy",
            category="demand",
            role="5-minute demand scaling profile",
            owner="demand_forecast_agent",
            loader="numpy",
            required=True,
        ),
        AssetSpec(
            key="state_transition_default",
            path="Load Forecasting/state_transition_matrices_20260328_214839.xlsx",
            category="forecast",
            role="hourly node transition matrices",
            owner="trip_forecast_agent",
            loader="excel",
            required=True,
        ),
        AssetSpec(
            key="state_transition_day1",
            path="Load Forecasting/state_transition_matrices_day1.xlsx",
            category="forecast",
            role="June 30 transition matrices",
            owner="trip_forecast_agent",
            loader="excel",
        ),
        AssetSpec(
            key="state_transition_day2",
            path="Load Forecasting/state_transition_matrices_day2.xlsx",
            category="forecast",
            role="July 1 transition matrices",
            owner="trip_forecast_agent",
            loader="excel",
        ),
        AssetSpec(
            key="state_transition_day3",
            path="Load Forecasting/state_transition_matrices_day3.xlsx",
            category="forecast",
            role="July 2 transition matrices",
            owner="trip_forecast_agent",
            loader="excel",
        ),
        AssetSpec(
            key="od_day1",
            path="Load Forecasting/od_matrices_6.30.xlsx",
            category="forecast",
            role="June 30 OD estimate",
            owner="demand_forecast_agent",
            loader="excel",
        ),
        AssetSpec(
            key="od_day2",
            path="Load Forecasting/od_matrices_7.1.xlsx",
            category="forecast",
            role="July 1 OD estimate",
            owner="demand_forecast_agent",
            loader="excel",
        ),
        AssetSpec(
            key="od_day3",
            path="Load Forecasting/od_matrices_7.2.xlsx",
            category="forecast",
            role="July 2 OD estimate",
            owner="demand_forecast_agent",
            loader="excel",
        ),
        AssetSpec(
            key="dynamic_price",
            path="Load Forecasting/dynamic_price.npy",
            category="grid",
            role="dynamic charging price strategy",
            owner="grid_strategy_agent",
            loader="numpy",
            required=True,
            region_scoped=True,
        ),
        AssetSpec(
            key="pv_profile",
            path="Load Forecasting/pv_data.npy",
            category="grid",
            role="15-minute photovoltaic profile",
            owner="grid_twin",
            loader="numpy",
            required=True,
            region_scoped=True,
        ),
        AssetSpec(
            key="wind_profile",
            path="Load Forecasting/wind_data.npy",
            category="grid",
            role="15-minute wind profile",
            owner="grid_twin",
            loader="numpy",
            required=True,
            region_scoped=True,
        ),
        AssetSpec(
            key="residential_load_profile",
            path="Load Forecasting/j_load_data.npy",
            category="grid",
            role="15-minute residential base-load profile",
            owner="grid_twin",
            loader="numpy",
            required=True,
            region_scoped=True,
        ),
        AssetSpec(
            key="commercial_load_profile",
            path="Load Forecasting/s_load_data.npy",
            category="grid",
            role="15-minute commercial base-load profile",
            owner="grid_twin",
            loader="numpy",
            required=True,
            region_scoped=True,
        ),
        AssetSpec(
            key="work_load_profile",
            path="Load Forecasting/b_load_data.npy",
            category="grid",
            role="15-minute work-area base-load profile",
            owner="grid_twin",
            loader="numpy",
            required=True,
            region_scoped=True,
        ),
        AssetSpec(
            key="road_time_day1",
            path="Load Forecasting/road_time_table_day1.npy",
            category="precompute",
            role="node-to-station travel time cache",
            owner="traffic_twin",
            loader="numpy",
            region_scoped=True,
        ),
        AssetSpec(
            key="road_time_day2",
            path="Load Forecasting/road_time_table_day2.npy",
            category="precompute",
            role="node-to-station travel time cache",
            owner="traffic_twin",
            loader="numpy",
            region_scoped=True,
        ),
        AssetSpec(
            key="road_time_day3",
            path="Load Forecasting/road_time_table_day3.npy",
            category="precompute",
            role="node-to-station travel time cache",
            owner="traffic_twin",
            loader="numpy",
            region_scoped=True,
        ),
        AssetSpec(
            key="delta_q_day1",
            path="Load Forecasting/delta_Q_table_day1.npy",
            category="precompute",
            role="node-to-station driving energy cache",
            owner="traffic_twin",
            loader="numpy",
            region_scoped=True,
        ),
        AssetSpec(
            key="delta_q_day2",
            path="Load Forecasting/delta_Q_table_day2.npy",
            category="precompute",
            role="node-to-station driving energy cache",
            owner="traffic_twin",
            loader="numpy",
            region_scoped=True,
        ),
        AssetSpec(
            key="delta_q_day3",
            path="Load Forecasting/delta_Q_table_day3.npy",
            category="precompute",
            role="node-to-station driving energy cache",
            owner="traffic_twin",
            loader="numpy",
            region_scoped=True,
        ),
        AssetSpec(
            key="eps_day1",
            path="Load Forecasting/eps_table_day1.npy",
            category="precompute",
            role="5-minute station price cache",
            owner="grid_twin",
            loader="numpy",
            region_scoped=True,
        ),
        AssetSpec(
            key="eps_day2",
            path="Load Forecasting/eps_table_day2.npy",
            category="precompute",
            role="5-minute station price cache",
            owner="grid_twin",
            loader="numpy",
            region_scoped=True,
        ),
        AssetSpec(
            key="eps_day3",
            path="Load Forecasting/eps_table_day3.npy",
            category="precompute",
            role="5-minute station price cache",
            owner="grid_twin",
            loader="numpy",
            region_scoped=True,
        ),
        AssetSpec(
            key="charge_time_table",
            path="Load Forecasting/charge_time_table.npy",
            category="charging",
            role="fast-charge time lookup table",
            owner="station_twin",
            loader="numpy",
            required=True,
        ),
        AssetSpec(
            key="line_resistance",
            path="Load Forecasting/Rl_data.npy",
            category="grid",
            role="distribution line resistance array",
            owner="grid_twin",
            loader="numpy",
            region_scoped=True,
        ),
        AssetSpec(
            key="line_reactance",
            path="Load Forecasting/Xl_data.npy",
            category="grid",
            role="distribution line reactance array",
            owner="grid_twin",
            loader="numpy",
            region_scoped=True,
        ),
        AssetSpec(
            key="line_index",
            path="Load Forecasting/Li_data.npy",
            category="grid",
            role="distribution line endpoint index array",
            owner="grid_twin",
            loader="numpy",
            region_scoped=True,
        ),
        AssetSpec(
            key="charge_energy_table",
            path="Load Forecasting/charge_energy_table.npy",
            category="charging",
            role="fast-charge energy lookup table",
            owner="station_twin",
            loader="numpy",
            required=True,
        ),
        AssetSpec(
            key="optimized_price_positions_iter10",
            path="Load Forecasting/result/Curve_positons_iter10.npy",
            category="optimization",
            role="WOA/Gurobi-derived price policy candidate",
            owner="grid_strategy_agent",
            loader="numpy",
            notes="Reuse as an existing policy candidate before recomputing optimization.",
        ),
        AssetSpec(
            key="legacy_jun30_events",
            path="Load Forecasting/output_jun30/charge_events_all_day1.pkl",
            category="baseline",
            role="legacy event simulation output",
            owner="replay_agent",
            loader="pickle",
        ),
        AssetSpec(
            key="legacy_jun30_load",
            path="Load Forecasting/output_jun30/predicted_load_ac_jun30.npy",
            category="baseline",
            role="legacy station load baseline",
            owner="replay_agent",
            loader="numpy",
        ),
        AssetSpec(
            key="legacy_jul1_events",
            path="Load Forecasting/output_jul1/charge_events_all_day2.pkl",
            category="baseline",
            role="legacy event simulation output",
            owner="replay_agent",
            loader="pickle",
        ),
        AssetSpec(
            key="legacy_jul1_load",
            path="Load Forecasting/output_jul1/predicted_load_ac_jul1.npy",
            category="baseline",
            role="legacy station load baseline",
            owner="replay_agent",
            loader="numpy",
        ),
        AssetSpec(
            key="legacy_jul2_events",
            path="Load Forecasting/output_jul2/charge_events_all_day3.pkl",
            category="baseline",
            role="legacy event simulation output",
            owner="replay_agent",
            loader="pickle",
        ),
        AssetSpec(
            key="legacy_jul2_load",
            path="Load Forecasting/output_jul2/predicted_load_ac_jul2.npy",
            category="baseline",
            role="legacy station load baseline",
            owner="replay_agent",
            loader="numpy",
        ),
        AssetSpec(
            key="legacy_jun30_no_queue_events",
            path="Load Forecasting/output_jun30_no_queue/charge_events_all_day1.pkl",
            category="baseline",
            role="legacy no-queue event simulation output",
            owner="replay_agent",
            loader="pickle",
        ),
        AssetSpec(
            key="legacy_jun30_no_queue_load",
            path="Load Forecasting/output_jun30_no_queue/predicted_load_ac_jun30.npy",
            category="baseline",
            role="legacy no-queue station load baseline",
            owner="replay_agent",
            loader="numpy",
        ),
    ]


def default_asset_registry(root: str | Path) -> AssetRegistry:
    return AssetRegistry(Path(root), default_asset_specs())
