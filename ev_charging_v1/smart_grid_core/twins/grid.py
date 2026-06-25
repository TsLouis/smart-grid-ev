from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from ..assets import AssetRegistry
from .topology import TopologyEdge, TopologyGraph, TopologyNode


DEFAULT_GRID_LAYOUT = {
    0: (0.0, 0.0),
    1: (1.5, 0.0),
    2: (3.0, 0.7),
    3: (1.5, 1.5),
    4: (0.4, 2.2),
    5: (1.5, 3.0),
    6: (3.0, 2.6),
    7: (2.6, 4.0),
    8: (0.6, 1.0),
    9: (4.0, 1.6),
    10: (3.8, 3.2),
    11: (0.6, -0.9),
    12: (1.4, 4.5),
    13: (-1.2, -0.3),
    14: (2.2, 2.2),
    15: (3.1, 1.1),
    16: (1.4, 5.2),
    17: (-1.0, 0.9),
    18: (3.2, 5.0),
    19: (-0.7, 3.6),
    20: (-0.1, -1.5),
    21: (2.1, -1.4),
    22: (3.5, -0.6),
    23: (-1.4, 2.1),
}

DEFAULT_GRID_NODE_KINDS = {
    "substation": {20, 21, 22, 23},
    "residential_load": {0, 1, 2, 6, 7, 9, 10, 13, 17},
    "commercial_load": {3, 4, 8, 14, 15, 18},
    "work_load": {5, 12, 16, 19},
}


@dataclass(frozen=True)
class GridTwinConfig:
    n_stations: int = 7
    n_distribution_nodes: int = 24
    station_price_delta_tau_h: float = 5 / 60
    power_delta_tau_h: float = 15 / 60


@dataclass(frozen=True)
class StationPriceState:
    station_index: int
    query_time_h: float
    time_step: int
    price: float


@dataclass(frozen=True)
class RenewableState:
    query_time_h: float
    time_step: int
    pv: float
    wind: float


@dataclass(frozen=True)
class BaseLoadState:
    query_time_h: float
    time_step: int
    residential: float
    commercial: float
    work: float


class GridTwin:
    """White-box grid-side state twin.

    The twin exposes loaded grid states and deterministic time-indexed values.
    It does not decide price policy, grid-friendly weights, demand response, or
    charging guidance. Those uncertain strategy choices belong to agents.
    """

    def __init__(
        self,
        *,
        dynamic_price,
        station_price_table,
        pv_profile,
        wind_profile,
        residential_load_profile,
        commercial_load_profile,
        work_load_profile,
        line_index=None,
        line_resistance=None,
        line_reactance=None,
        config: GridTwinConfig = GridTwinConfig(),
    ):
        self.dynamic_price = dynamic_price
        self.station_price_table = station_price_table
        self.pv_profile = pv_profile
        self.wind_profile = wind_profile
        self.residential_load_profile = residential_load_profile
        self.commercial_load_profile = commercial_load_profile
        self.work_load_profile = work_load_profile
        self.line_index = line_index
        self.line_resistance = line_resistance
        self.line_reactance = line_reactance
        self.config = config
        self._validate_shapes()

    @classmethod
    def from_registry(cls, registry: AssetRegistry, *, day: int) -> "GridTwin":
        import numpy as np

        return cls(
            dynamic_price=np.load(registry.path("dynamic_price")),
            station_price_table=np.load(registry.path(f"eps_day{day}")),
            pv_profile=np.load(registry.path("pv_profile")),
            wind_profile=np.load(registry.path("wind_profile")),
            residential_load_profile=np.load(registry.path("residential_load_profile")),
            commercial_load_profile=np.load(registry.path("commercial_load_profile")),
            work_load_profile=np.load(registry.path("work_load_profile")),
            line_index=np.load(registry.path("line_index"), allow_pickle=True).astype(int)
            if registry.exists("line_index")
            else None,
            line_resistance=np.load(registry.path("line_resistance"), allow_pickle=True)
            if registry.exists("line_resistance")
            else None,
            line_reactance=np.load(registry.path("line_reactance"), allow_pickle=True)
            if registry.exists("line_reactance")
            else None,
        )

    def station_price(self, *, station_index: int, query_time_h: float) -> StationPriceState:
        self._validate_station(station_index)
        step = self._time_to_step(query_time_h, self.config.station_price_delta_tau_h, self.station_price_table.shape[0])
        return StationPriceState(
            station_index=station_index,
            query_time_h=float(query_time_h),
            time_step=step,
            price=float(self.station_price_table[step, station_index]),
        )

    def all_station_prices(self, *, query_time_h: float) -> list[StationPriceState]:
        return [
            self.station_price(station_index=station_index, query_time_h=query_time_h)
            for station_index in range(self.config.n_stations)
        ]

    def renewable_state(self, *, query_time_h: float) -> RenewableState:
        step = self._time_to_step(query_time_h, self.config.power_delta_tau_h, len(self.pv_profile))
        return RenewableState(
            query_time_h=float(query_time_h),
            time_step=step,
            pv=float(self.pv_profile[step]),
            wind=float(self.wind_profile[step]),
        )

    def base_load_state(self, *, query_time_h: float) -> BaseLoadState:
        step = self._time_to_step(query_time_h, self.config.power_delta_tau_h, len(self.residential_load_profile))
        return BaseLoadState(
            query_time_h=float(query_time_h),
            time_step=step,
            residential=float(self.residential_load_profile[step]),
            commercial=float(self.commercial_load_profile[step]),
            work=float(self.work_load_profile[step]),
        )

    def snapshot(self) -> dict:
        return {
            "n_stations": self.config.n_stations,
            "n_distribution_nodes": self.config.n_distribution_nodes,
            "dynamic_price_shape": tuple(int(v) for v in self.dynamic_price.shape),
            "station_price_table_shape": tuple(int(v) for v in self.station_price_table.shape),
            "pv_profile_len": int(len(self.pv_profile)),
            "wind_profile_len": int(len(self.wind_profile)),
            "residential_load_profile_len": int(len(self.residential_load_profile)),
            "commercial_load_profile_len": int(len(self.commercial_load_profile)),
            "work_load_profile_len": int(len(self.work_load_profile)),
            "line_count": int(len(self.line_index)) if self.line_index is not None else 0,
        }

    def topology_graph(
        self,
        *,
        station_grid_nodes: Sequence[int] | None = None,
        query_time_h: float | None = None,
    ) -> TopologyGraph:
        """Build the grid layer topology.

        When `station_grid_nodes` is provided (length == n_stations), the
        listed nodes are tagged as `kind="station"` and carry their
        `station_index` in metadata. This is the only way the grid layer
        learns where the 7 charging stations sit; without it the grid is
        electrically meaningful but station-blind.

        When `query_time_h` is provided, each node carries a `state` dict
        with the relevant time-indexed value (load for load nodes,
        renewable for substations, price for stations).
        """
        load_by_kind: dict[str, float] = {}
        renewable_kw: float | None = None
        station_prices: dict[int, float] = {}
        if query_time_h is not None:
            base = self.base_load_state(query_time_h=query_time_h)
            load_by_kind = {
                "residential_load": base.residential,
                "commercial_load": base.commercial,
                "work_load": base.work,
            }
            renewable = self.renewable_state(query_time_h=query_time_h)
            renewable_kw = renewable.pv + renewable.wind
            for price in self.all_station_prices(query_time_h=query_time_h):
                station_prices[price.station_index] = price.price
        node_ids = set(range(self.config.n_distribution_nodes))
        if self.line_index is not None:
            for source, target in self.line_index:
                node_ids.add(int(source))
                node_ids.add(int(target))

        station_lookup: dict[int, int] = {}
        if station_grid_nodes is not None:
            if len(station_grid_nodes) != self.config.n_stations:
                raise ValueError(
                    f"station_grid_nodes length {len(station_grid_nodes)} != n_stations {self.config.n_stations}"
                )
            for station_index, grid_node in enumerate(station_grid_nodes):
                station_lookup[int(grid_node)] = station_index
                node_ids.add(int(grid_node))

        nodes = []
        for node_id in sorted(node_ids):
            x, y = DEFAULT_GRID_LAYOUT.get(node_id, (float(node_id % 6), float(node_id // 6)))
            metadata: dict = {"node_index": node_id}
            if node_id in station_lookup:
                kind = "station"
                metadata["station_index"] = station_lookup[node_id]
            else:
                kind = "grid_node"
                for candidate, values in DEFAULT_GRID_NODE_KINDS.items():
                    if node_id in values:
                        kind = candidate
                        break
            node_state: dict = {}
            if query_time_h is not None:
                if kind in load_by_kind:
                    node_state["load_kw"] = load_by_kind[kind]
                if kind == "substation" and renewable_kw is not None:
                    node_state["renewable_kw"] = renewable_kw
                if kind == "station" and node_id in station_lookup:
                    station_index = station_lookup[node_id]
                    if station_index in station_prices:
                        node_state["price"] = station_prices[station_index]
            nodes.append(
                TopologyNode(
                    id=f"grid:{node_id}",
                    label=str(node_id + 1),
                    twin_id="grid",
                    layer="grid",
                    kind=kind,
                    x=x,
                    y=y,
                    state=node_state,
                    metadata=metadata,
                )
            )

        edges = []
        if self.line_index is not None:
            for index, (source, target) in enumerate(self.line_index):
                resistance = None
                reactance = None
                if self.line_resistance is not None and index < len(self.line_resistance):
                    resistance = float(self.line_resistance[index])
                if self.line_reactance is not None and index < len(self.line_reactance):
                    reactance = float(self.line_reactance[index])
                edges.append(
                    TopologyEdge(
                        id=f"grid:{int(source)}-{int(target)}",
                        source=f"grid:{int(source)}",
                        target=f"grid:{int(target)}",
                        twin_id="grid",
                        layer="grid",
                        kind="distribution_line",
                        directed=False,
                        weight=resistance,
                        state={"resistance": resistance, "reactance": reactance},
                        metadata={"line_index": index},
                    )
                )
        return TopologyGraph(
            twin_id="grid",
            layer="grid",
            nodes=nodes,
            edges=edges,
            metadata={"source": "grid_twin", "layout": "distribution grid topology"},
        )

    def _validate_shapes(self) -> None:
        if self.dynamic_price.shape[1] != self.config.n_stations:
            raise ValueError(f"dynamic_price station count mismatch: {self.dynamic_price.shape}")
        if self.station_price_table.shape[1] != self.config.n_stations:
            raise ValueError(f"station_price_table station count mismatch: {self.station_price_table.shape}")
        profiles = [
            self.pv_profile,
            self.wind_profile,
            self.residential_load_profile,
            self.commercial_load_profile,
            self.work_load_profile,
        ]
        lengths = {len(profile) for profile in profiles}
        if len(lengths) != 1:
            raise ValueError(f"grid profile lengths mismatch: {sorted(lengths)}")

    def _validate_station(self, station_index: int) -> None:
        if station_index < 0 or station_index >= self.config.n_stations:
            raise ValueError(f"station_index out of range: {station_index}")

    @staticmethod
    def _time_to_step(query_time_h: float, delta_tau_h: float, n_steps: int) -> int:
        step = int(query_time_h / delta_tau_h)
        return max(0, min(step, n_steps - 1))
