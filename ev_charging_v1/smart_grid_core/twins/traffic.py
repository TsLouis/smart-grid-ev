from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Sequence

from ..assets import AssetRegistry
from .topology import TopologyEdge, TopologyGraph, TopologyNode


DEFAULT_TRAFFIC_LAYOUT = {
    0: (0.0, 0.0),
    1: (1.2, 0.0),
    2: (2.4, 0.0),
    3: (3.6, 0.0),
    4: (4.8, 0.0),
    5: (6.0, 0.0),
    6: (7.2, 0.0),
    7: (2.4, 0.8),
    8: (3.6, 0.8),
    9: (1.2, 1.6),
    10: (3.6, 1.6),
    11: (4.8, 1.6),
    12: (6.0, 1.6),
    13: (7.2, 1.6),
    14: (0.0, 2.4),
    15: (2.4, 2.4),
    16: (3.6, 2.4),
    17: (4.8, 2.4),
    18: (6.0, 2.4),
    19: (7.2, 2.4),
    20: (4.8, 3.2),
    21: (1.2, 4.0),
    22: (2.4, 4.0),
    23: (4.8, 4.0),
    24: (6.0, 4.0),
    25: (2.4, 4.8),
    26: (4.8, 4.8),
    27: (6.0, 4.8),
    28: (7.2, 4.8),
    29: (8.4, 4.8),
    30: (2.4, 5.6),
    31: (4.8, 5.6),
}

DEFAULT_NODE_ZONES = {
    "residential": {0, 1, 4, 5, 6, 12, 13, 18, 19, 23, 24, 26, 27, 28, 29, 31},
    "work": {2, 3, 7, 14, 21, 22, 25, 30},
    "commercial": {8, 9, 10, 11, 15, 16, 17, 20},
}


@dataclass(frozen=True)
class TrafficTwinConfig:
    station_nodes: Sequence[int]
    delta_tau_h: float = 5 / 60
    n_nodes: int = 32


@dataclass(frozen=True)
class StationRouteState:
    origin_node: int
    station_index: int
    station_node: int
    time_step: int
    query_time_h: float
    travel_time_h: float
    travel_energy_kwh: float
    reachable: bool


class PrecomputedTrafficTwin:
    """White-box traffic twin backed by precomputed route tables.

    This twin does not predict destination, demand, congestion, queue time, or
    station choice. It only exposes deterministic values already computed by the
    mechanism layer: node-to-station travel time and driving energy.
    """

    def __init__(
        self,
        *,
        road_time_table,
        delta_q_table,
        config: TrafficTwinConfig,
    ):
        self.road_time_table = road_time_table
        self.delta_q_table = delta_q_table
        self.config = config
        self._validate_shapes()

    @classmethod
    def from_registry(
        cls,
        registry: AssetRegistry,
        *,
        day: int,
        station_nodes: Sequence[int],
        delta_tau_h: float = 5 / 60,
    ) -> "PrecomputedTrafficTwin":
        import numpy as np

        road_key = f"road_time_day{day}"
        delta_key = f"delta_q_day{day}"
        return cls(
            road_time_table=np.load(registry.path(road_key)),
            delta_q_table=np.load(registry.path(delta_key)),
            config=TrafficTwinConfig(
                station_nodes=list(station_nodes),
                delta_tau_h=delta_tau_h,
            ),
        )

    def time_to_step(self, query_time_h: float) -> int:
        step = int(query_time_h / self.config.delta_tau_h)
        return max(0, min(step, self.road_time_table.shape[0] - 1))

    def station_route_state(
        self,
        *,
        origin_node: int,
        station_index: int,
        query_time_h: float,
    ) -> StationRouteState:
        self._validate_origin(origin_node)
        self._validate_station(station_index)
        step = self.time_to_step(query_time_h)
        travel_time = float(self.road_time_table[step, origin_node, station_index])
        travel_energy = float(self.delta_q_table[step, origin_node, station_index])
        reachable = self._is_finite(travel_time) and self._is_finite(travel_energy)
        return StationRouteState(
            origin_node=origin_node,
            station_index=station_index,
            station_node=int(self.config.station_nodes[station_index]),
            time_step=step,
            query_time_h=float(query_time_h),
            travel_time_h=travel_time,
            travel_energy_kwh=travel_energy,
            reachable=reachable,
        )

    def station_route_states(self, *, origin_node: int, query_time_h: float) -> List[StationRouteState]:
        return [
            self.station_route_state(
                origin_node=origin_node,
                station_index=station_index,
                query_time_h=query_time_h,
            )
            for station_index in range(len(self.config.station_nodes))
        ]

    def reachable_station_states(self, *, origin_node: int, query_time_h: float) -> List[StationRouteState]:
        return [state for state in self.station_route_states(origin_node=origin_node, query_time_h=query_time_h) if state.reachable]

    def snapshot(self) -> dict:
        return {
            "n_time_steps": int(self.road_time_table.shape[0]),
            "n_nodes": int(self.road_time_table.shape[1]),
            "n_stations": int(self.road_time_table.shape[2]),
            "station_nodes": [int(node) for node in self.config.station_nodes],
            "delta_tau_h": float(self.config.delta_tau_h),
        }

    def topology_graph(
        self,
        *,
        flow_tensor=None,
        station_nodes: Sequence[int] | None = None,
        max_edges: int = 80,
        required_display_edges: Sequence[tuple[int, int]] = ((7, 14), (14, 7)),
        query_time_h: float | None = None,
    ) -> TopologyGraph:
        active_station_nodes = [int(node) for node in (station_nodes or self.config.station_nodes)]
        station_lookup = {node: index for index, node in enumerate(active_station_nodes)}

        node_inflow: dict[int, float] = {}
        if flow_tensor is not None and query_time_h is not None and len(flow_tensor.shape) == 3:
            steps_per_h = max(1, int(round(1.0 / self.config.delta_tau_h)))
            step = max(0, min(int(query_time_h * steps_per_h), flow_tensor.shape[0] - 1))
            slice_at_t = flow_tensor[step]
            for node_id in range(min(self.config.n_nodes, slice_at_t.shape[1])):
                node_inflow[node_id] = float(slice_at_t[:, node_id].sum())

        nodes = []
        for node_id in range(self.config.n_nodes):
            x, y = DEFAULT_TRAFFIC_LAYOUT.get(node_id, (float(node_id % 8), float(node_id // 8)))
            zone = "unknown"
            for candidate, values in DEFAULT_NODE_ZONES.items():
                if node_id in values:
                    zone = candidate
                    break
            metadata: dict = {"node_index": node_id, "display_node": node_id + 1}
            if node_id in station_lookup:
                metadata["station_index"] = station_lookup[node_id]
            node_state: dict = {"is_charging_station": node_id in station_lookup}
            if node_id in node_inflow:
                node_state["inflow"] = node_inflow[node_id]
            nodes.append(
                TopologyNode(
                    id=f"traffic:{node_id}",
                    label=str(node_id + 1),
                    twin_id="traffic",
                    layer="traffic",
                    kind="station" if node_id in station_lookup else zone,
                    x=x,
                    y=y,
                    state=node_state,
                    metadata=metadata,
                )
            )

        edge_scores: dict[tuple[int, int], float] = {}
        if flow_tensor is not None:
            sample = flow_tensor
            if len(sample.shape) == 3:
                sample = sample[: min(sample.shape[0], 288)]
                mean_flow = sample.mean(axis=0)
            else:
                mean_flow = sample
            for source in range(min(mean_flow.shape[0], self.config.n_nodes)):
                for target in range(min(mean_flow.shape[1], self.config.n_nodes)):
                    value = float(mean_flow[source, target])
                    if source != target and value > 0:
                        edge_scores[(source, target)] = value
        else:
            mean_time = self.road_time_table.mean(axis=0)
            for source in range(mean_time.shape[0]):
                for station_index, target in enumerate(self.config.station_nodes):
                    value = float(mean_time[source, station_index])
                    if source != int(target) and value == value and value < float("inf"):
                        edge_scores[(source, int(target))] = 1.0 / max(value, 0.001)

        selected_edges = dict(sorted(edge_scores.items(), key=lambda item: item[1], reverse=True)[:max_edges])
        for source_display, target_display in required_display_edges:
            source = int(source_display) - 1
            target = int(target_display) - 1
            if (source, target) in edge_scores:
                selected_edges[(source, target)] = edge_scores[(source, target)]

        sorted_edges = sorted(selected_edges.items(), key=lambda item: item[1], reverse=True)
        edges = [
            TopologyEdge(
                id=f"traffic:{source}->{target}",
                source=f"traffic:{source}",
                target=f"traffic:{target}",
                twin_id="traffic",
                layer="traffic",
                kind="road_flow",
                directed=True,
                weight=float(weight),
                state={"mean_flow": float(weight)},
                metadata={
                    "source_node": source,
                    "target_node": target,
                    "source_display_node": source + 1,
                    "target_display_node": target + 1,
                },
            )
            for (source, target), weight in sorted_edges
        ]
        return TopologyGraph(
            twin_id="traffic",
            layer="traffic",
            nodes=nodes,
            edges=edges,
            metadata={"source": "traffic_twin", "layout": "32-node road topology"},
        )

    def _validate_shapes(self) -> None:
        if self.road_time_table.shape != self.delta_q_table.shape:
            raise ValueError(
                f"road_time_table and delta_q_table shape mismatch: "
                f"{self.road_time_table.shape} != {self.delta_q_table.shape}"
            )
        if len(self.road_time_table.shape) != 3:
            raise ValueError("traffic precompute tables must have shape (time, node, station)")
        if self.road_time_table.shape[1] != self.config.n_nodes:
            raise ValueError(f"expected {self.config.n_nodes} nodes, got {self.road_time_table.shape[1]}")
        if self.road_time_table.shape[2] != len(self.config.station_nodes):
            raise ValueError(
                f"station node count mismatch: table has {self.road_time_table.shape[2]}, "
                f"config has {len(self.config.station_nodes)}"
            )

    def _validate_origin(self, origin_node: int) -> None:
        if origin_node < 0 or origin_node >= self.road_time_table.shape[1]:
            raise ValueError(f"origin_node out of range: {origin_node}")

    def _validate_station(self, station_index: int) -> None:
        if station_index < 0 or station_index >= self.road_time_table.shape[2]:
            raise ValueError(f"station_index out of range: {station_index}")

    @staticmethod
    def _is_finite(value: float) -> bool:
        return value != float("inf") and value == value
