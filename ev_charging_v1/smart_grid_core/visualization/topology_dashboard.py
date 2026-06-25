from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..agents import HumanLLMAgent
from ..assets import AssetRegistry, default_asset_registry
from ..orchestrator import InteractionOrchestrator, OrchestrationStep
from ..runtime.default_manifests import default_agent_manifests
from ..runtime.default_regions import build_default_city
from ..runtime.default_root import build_default_root_agent
from ..twins import CityTwin
from ..twins.topology import graph_metrics


DEFAULT_OBSERVATION: dict[str, Any] = {
    "current_node": 0,
    "current_time": 8.0,
    "soc": 0.35,
    "battery_capacity_kwh": 61.4,
    "next_trip_energy_kwh": 12.0,
    "reachable_stations": [
        {"station_index": 0, "travel_time_h": 0.10, "travel_energy_kwh": 1.5},
        {"station_index": 2, "travel_time_h": 0.18, "travel_energy_kwh": 2.4},
        {"station_index": 4, "travel_time_h": 0.22, "travel_energy_kwh": 3.0},
    ],
    "station_prices": {0: 0.85, 2: 0.92, 4: 0.78},
    "station_resource_state": {"fast_queue_len": 2, "slow_queue_len": 1},
}


DEFAULT_TIMELINE_HOURS = 24
DEFAULT_TIMELINE_STEPS = 24


# Which physical twin layer a consumed capability is "read" from, plus a label.
_CAPABILITY_TWIN_MAP: dict[str, tuple[str | None, str]] = {
    "vehicle_state": ("traffic", "车辆 SOC / 电量"),
    "route_state": ("traffic", "路网行驶时间"),
    "reachable_stations": ("traffic", "可达充电站"),
    "station_resource_state": ("grid", "充电站排队状态"),
    "station_price": ("grid", "充电站电价"),
    "renewable_state": ("grid", "可再生出力 PV+风"),
    "base_load_state": ("grid", "电网基础负荷"),
    "charge_event": (None, "历史充电事件"),
}
# Which twin layer a produced capability "writes" back to, plus what it changes.
_PRODUCE_TWIN_MAP: dict[str, tuple[str | None, str]] = {
    "price_policy": ("grid", "调整充电站电价"),
    "grid_friendly_strategy": ("grid", "平移负荷曲线"),
    "station_decision": ("traffic", "选定一个充电站"),
    "charging_mode": ("grid", "设定充电模式 快/慢"),
    "target_soc": ("traffic", "设定目标 SOC"),
}


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _load_flow_tensors(registry: AssetRegistry, city: CityTwin) -> dict[str, np.ndarray]:
    tensors: dict[str, np.ndarray] = {}
    for region in city.regions:
        region_registry = registry.for_region(region.region_id)
        path = region_registry.path("traffic_flow_tensor")
        if path.exists():
            tensors[region.region_id] = np.load(path)
    return tensors


def _region_summary(city: CityTwin) -> list[dict]:
    return [
        {
            "region_id": region.region_id,
            "label": region.label,
            "parent_region_id": region.parent_region_id,
            "coord_offset": list(region.coord_offset),
            "station_count": len(region.station_alignments),
            "station_alignments": [
                {
                    "station_index": a.station_index,
                    "grid_node": a.grid_node,
                    "traffic_node": a.traffic_node,
                    "label": a.label,
                }
                for a in region.station_alignments
            ],
        }
        for region in city.regions
    ]


def _records_payload(step: OrchestrationStep) -> list[dict]:
    return [_plain(asdict(record)) for record in step.records]


def _graphs_with_metrics(step: OrchestrationStep) -> list[dict]:
    payload: list[dict] = []
    for graph in step.geo_graphs:
        data = graph.to_dict()
        data["metrics"] = graph_metrics(graph)
        payload.append(data)
    return payload


def _control_flow_payload(step: OrchestrationStep) -> dict | None:
    if step.control_flow_graph is None:
        return None
    data = step.control_flow_graph.to_dict()
    data["metrics"] = graph_metrics(step.control_flow_graph)
    return data


def _agent_twin_links() -> dict[str, dict]:
    """Per-subagent: which twin layers / quantities it reads and writes."""
    links: dict[str, dict] = {}
    for manifest in default_agent_manifests():
        read_detail = [
            {"capability": cap, "layer": _CAPABILITY_TWIN_MAP[cap][0], "label": _CAPABILITY_TWIN_MAP[cap][1]}
            for cap in manifest.consumes
            if cap in _CAPABILITY_TWIN_MAP
        ]
        write_detail = [
            {"capability": cap, "layer": _PRODUCE_TWIN_MAP[cap][0], "label": _PRODUCE_TWIN_MAP[cap][1]}
            for cap in manifest.produces
            if cap in _PRODUCE_TWIN_MAP
        ]
        links[manifest.agent_id] = {
            "agent_type": manifest.agent_type,
            "reads": sorted({d["layer"] for d in read_detail if d["layer"]}),
            "writes": sorted({d["layer"] for d in write_detail if d["layer"]}),
            "read_detail": read_detail,
            "write_detail": write_detail,
            "consumes": list(manifest.consumes),
            "produces": list(manifest.produces),
        }
    return links


def _city_graph(city: CityTwin, geo_graphs: list[dict]) -> dict:
    nodes: list[dict] = [{"id": city.city_id, "kind": "city", "label": city.city_id, "stats": {}}]
    edges: list[dict] = []
    for region in city.regions:
        region_geo = [
            g for g in geo_graphs if g["region_id"] == region.region_id and g["layer"] in ("grid", "traffic")
        ]
        node_total = sum(g["metrics"]["node_count"] for g in region_geo)
        edge_total = sum(g["metrics"]["edge_count"] for g in region_geo)
        nodes.append(
            {
                "id": region.region_id,
                "kind": "region",
                "label": region.label,
                "parent": region.parent_region_id,
                "stats": {
                    "nodes": node_total,
                    "edges": edge_total,
                    "stations": len(region.station_alignments),
                    "grids": len([g for g in region_geo if g["layer"] == "grid"]),
                },
            }
        )
        edges.append({"source": city.city_id, "target": region.region_id, "kind": "contains"})
    region_ids = [r.region_id for r in city.regions]
    for i in range(len(region_ids)):
        for j in range(i + 1, len(region_ids)):
            edges.append({"source": region_ids[i], "target": region_ids[j], "kind": "shares_schema"})
    return {"nodes": nodes, "edges": edges}


def _build_timeline(
    flow_tensors: dict[str, np.ndarray],
    city: CityTwin,
    *,
    rendered_traffic_edges: set[str],
    n_frames: int = DEFAULT_TIMELINE_STEPS,
) -> list[dict]:
    """Per-frame state along the 24-hour profile (traffic flow + grid daily curve)."""
    if not flow_tensors:
        return []
    sample = next(iter(flow_tensors.values()))
    if sample.ndim != 3 or sample.shape[0] == 0:
        return []
    total_steps = sample.shape[0]
    steps_per_h = total_steps / DEFAULT_TIMELINE_HOURS
    frames: list[dict] = []
    for frame_index in range(n_frames):
        hour = frame_index * DEFAULT_TIMELINE_HOURS / max(n_frames, 1)
        step = max(0, min(int(hour * steps_per_h), total_steps - 1))
        per_region: dict[str, dict] = {}
        for region in city.regions:
            entry: dict[str, Any] = {}
            tensor = flow_tensors.get(region.region_id)
            if tensor is not None:
                slice_at_t = tensor[step]
                n = min(slice_at_t.shape[0], slice_at_t.shape[1])
                edge_values: dict[str, float] = {}
                inflow: dict[str, float] = {}
                for target in range(n):
                    inflow[str(target)] = float(slice_at_t[:, target].sum())
                for source in range(n):
                    for target in range(n):
                        if source == target:
                            continue
                        value = float(slice_at_t[source, target])
                        if value <= 0:
                            continue
                        edge_id = f"{region.region_id}:traffic:{source}->{target}"
                        if edge_id in rendered_traffic_edges:
                            edge_values[edge_id] = value
                entry["traffic_edges"] = edge_values
                entry["traffic_inflow"] = inflow
            try:
                base = region.grid.base_load_state(query_time_h=hour)
                renew = region.grid.renewable_state(query_time_h=hour)
                prices = region.grid.all_station_prices(query_time_h=hour)
                entry["grid_load"] = {
                    "residential_load": float(base.residential),
                    "commercial_load": float(base.commercial),
                    "work_load": float(base.work),
                }
                entry["grid_renewable"] = float(renew.pv + renew.wind)
                entry["grid_prices"] = {str(p.station_index): float(p.price) for p in prices}
            except Exception:  # pragma: no cover - profile assets optional
                pass
            per_region[region.region_id] = entry
        frames.append({"frame": frame_index, "hour": float(hour), "regions": per_region})
    return frames


def build_topology_snapshot(
    root: str | Path = ".",
    *,
    day: int = 1,
    max_traffic_edges: int = 80,
    observation: Mapping[str, Any] | None = None,
    timeline_steps: int = DEFAULT_TIMELINE_STEPS,
) -> dict:
    """One orchestrator step + time-evolving twin state, packaged for the canvas."""
    root_path = Path(root)
    registry = default_asset_registry(root_path)
    city = build_default_city(registry, day=day)
    root_agent = build_default_root_agent(registry)
    orch = InteractionOrchestrator(root_agent=root_agent, city=city)
    flow_tensors = _load_flow_tensors(registry, city)
    obs = dict(observation or DEFAULT_OBSERVATION)
    step = orch.step(obs, flow_tensors=flow_tensors, max_traffic_edges=max_traffic_edges)

    geo_graphs = _graphs_with_metrics(step)
    control_flow = _control_flow_payload(step)
    rendered_traffic_edges = {
        e["id"] for g in geo_graphs if g["layer"] == "traffic" for e in g["edges"]
    }
    timeline = _build_timeline(
        flow_tensors, city, rendered_traffic_edges=rendered_traffic_edges, n_frames=timeline_steps
    )
    human = HumanLLMAgent(advice="(等待你扮演 LLM 顾问输入)").propose(
        {
            "runtime_context": {"purpose": "topology_canvas_review"},
            "topology_layers": ["grid", "traffic", "coupling", "control_flow"],
        }
    )
    return {
        "metadata": {
            "title": "Smart Grid 数字孪生 · 拓扑画布",
            "thesis": "一个 Agent · 在「市 → 区」的数字孪生上做决策 · 物理拓扑与决策流共享同一套孪生语言",
            "root": str(root_path.resolve()),
            "city_id": city.city_id,
            "day": day,
            "regions": _region_summary(city),
        },
        "summary_metrics": _aggregate_metrics(geo_graphs, control_flow),
        "city_graph": _city_graph(city, geo_graphs),
        "geo_graphs": geo_graphs,
        "control_flow_graph": control_flow,
        "agent_twin_links": _agent_twin_links(),
        "execution_order": list(step.execution_order),
        "records": _records_payload(step),
        "errors": {
            "control_flow": list(step.control_flow_errors),
            "geo": list(step.geo_errors),
        },
        "observation": _plain(obs),
        "timeline": timeline,
        "human_llm_proposal": asdict(human),
    }


def _aggregate_metrics(geo_graphs: list[dict], cf: dict | None) -> dict:
    node_total = sum(g["metrics"]["node_count"] for g in geo_graphs)
    edge_total = sum(g["metrics"]["edge_count"] for g in geo_graphs)
    interlayer = sum(g["metrics"]["edge_count"] for g in geo_graphs if g["layer"] == "coupling")
    avg_degrees = [g["metrics"]["avg_degree"] for g in geo_graphs if g["metrics"]["node_count"]]
    diameters = [g["metrics"]["diameter"] for g in geo_graphs if g["metrics"].get("diameter")]
    summary = {
        "geo_graphs": len(geo_graphs),
        "node_total": node_total,
        "edge_total": edge_total,
        "interlayer_edges": interlayer,
        "mean_avg_degree": (sum(avg_degrees) / len(avg_degrees)) if avg_degrees else 0.0,
        "max_diameter": max(diameters) if diameters else None,
    }
    if cf is not None:
        summary["control_flow"] = {
            "node_count": cf["metrics"]["node_count"],
            "edge_count": cf["metrics"]["edge_count"],
            "avg_degree": cf["metrics"]["avg_degree"],
            "diameter": cf["metrics"]["diameter"],
        }
    return summary


def write_topology_dashboard(snapshot: Mapping[str, Any], output_dir: str | Path) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "topology_snapshot.json"
    html_path = output / "topology_canvas.html"
    data = json.dumps(_plain(snapshot), ensure_ascii=False, indent=2)
    json_path.write_text(data + "\n", encoding="utf-8")
    html_path.write_text(_HTML_TEMPLATE.replace("__TOPOLOGY_DATA__", data), encoding="utf-8")
    return {"json_path": str(json_path), "html_path": str(html_path)}


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Smart Grid 数字孪生 · 拓扑画布</title>
<style>
  :root{
    --bg:#f5f6fa; --panel:#fff; --ink:#0f172a; --muted:#64748b; --line:#e2e8f0;
    --grid:#2563eb; --traffic:#0d9488; --station:#dc2626; --coupling:#9333ea;
    --ok:#16a34a; --warn:#d97706; --err:#dc2626;
  }
  *{box-sizing:border-box}
  body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;background:var(--bg);color:var(--ink)}
  .hero{padding:16px 22px 10px;background:linear-gradient(180deg,#fff,#f8fafc);border-bottom:1px solid var(--line)}
  .hero h1{margin:0;font-size:21px}
  .thesis{color:var(--muted);margin-top:3px;font-size:13px}
  .badges{margin-top:9px;display:flex;gap:7px;flex-wrap:wrap}
  .badge{display:inline-flex;gap:5px;align-items:baseline;padding:4px 10px;border:1px solid var(--line);border-radius:999px;background:#fff;font-size:12px}
  .badge b{font-size:14px;font-weight:700}
  .badge.ok{border-color:#86efac;background:#f0fdf4;color:#15803d}
  .badge.warn{border-color:#fcd34d;background:#fffbeb;color:#92400e}
  .badge.err{border-color:#fca5a5;background:#fef2f2;color:#991b1b}
  .chapters{display:flex;gap:6px;padding:8px 22px;background:#fff;border-bottom:1px solid var(--line);flex-wrap:wrap}
  .chapter{padding:7px 14px;border-radius:7px;font-size:13px;cursor:pointer;color:var(--muted);border:1px solid transparent}
  .chapter:hover{background:#f1f5f9;color:var(--ink)}
  .chapter.active{background:#eef2ff;color:#3730a3;border-color:#c7d2fe;font-weight:600}
  .shell{display:grid;grid-template-columns:1fr 400px;min-height:calc(100vh - 132px)}
  .canvas-wrap{position:relative;overflow:hidden;background:#fbfcff}
  svg{width:100%;height:100%;min-height:880px;display:block}
  aside{border-left:1px solid var(--line);background:#fff;display:flex;flex-direction:column;min-width:0;overflow-y:auto}
  .toolbar{display:flex;gap:6px;flex-wrap:wrap;padding:7px 9px;position:absolute;z-index:3;left:14px;top:14px;background:rgba(255,255,255,.96);border:1px solid var(--line);border-radius:10px;max-width:calc(100% - 28px);box-shadow:0 2px 6px rgba(15,23,42,.06)}
  .toolbar .group{display:flex;gap:4px;align-items:center;padding:2px 6px;border:1px solid #e2e8f0;border-radius:6px;background:#fafbff}
  .toolbar .group b{font-size:11px;color:#475569;font-weight:600;margin-right:2px}
  .chip{font-size:12px;padding:3px 9px;border:1px solid #cbd5e1;border-radius:999px;background:#fff;cursor:pointer;user-select:none}
  .chip.on{background:#eef2ff;border-color:#6366f1;color:#3730a3}
  .chip.station.on{background:#fef2f2;border-color:var(--station);color:var(--station)}
  .chip.coupling.on{background:#faf5ff;border-color:var(--coupling);color:var(--coupling)}
  .legend{position:absolute;right:14px;top:14px;z-index:3;background:rgba(255,255,255,.96);border:1px solid var(--line);border-radius:10px;padding:8px 11px;font-size:11px;box-shadow:0 2px 6px rgba(15,23,42,.06);max-width:230px}
  .legend h3{margin:0 0 5px;font-size:11px;color:var(--muted);letter-spacing:1px}
  .legend .row{display:flex;align-items:center;gap:6px;margin:2px 0;color:#334155}
  .legend .sw{width:11px;height:11px;border-radius:50%;flex:none}
  .legend .swl{width:18px;height:0;border-top:3px solid;flex:none}
  .playbar{background:linear-gradient(180deg,#ffffff,#f1f5f9);border-bottom:1px solid var(--line);padding:8px 22px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
  .playbar input[type=range]{flex:1;min-width:200px}
  .playbar button{font:inherit;padding:5px 13px;border-radius:6px;border:1px solid #cbd5e1;background:#fff;cursor:pointer}
  .playbar button.primary{background:#4f46e5;border-color:#4f46e5;color:#fff}
  .playbar .label{font-size:12px;color:var(--muted);min-width:84px;font-variant-numeric:tabular-nums}
  .section{padding:13px 15px;border-bottom:1px solid var(--line)}
  h2{margin:0 0 9px;font-size:14px}
  .meta{color:var(--muted);font-size:12px;line-height:1.55}
  .pill{display:inline-flex;border:1px solid var(--line);border-radius:999px;padding:2px 9px;font-size:11px;margin:2px 4px 2px 0}
  .pill.ok{border-color:#86efac;background:#f0fdf4;color:var(--ok)}
  .pill.err{border-color:#fca5a5;background:#fef2f2;color:var(--err)}
  .pill.warn{border-color:#fcd34d;background:#fffbeb;color:var(--warn)}
  .metric-row{display:flex;justify-content:space-between;padding:4px 0;font-size:12px;border-bottom:1px dashed #f1f5f9}
  .metric-row b{font-variant-numeric:tabular-nums}
  .order-item{padding:7px 9px;border:1px solid var(--line);border-radius:6px;margin-bottom:5px;cursor:pointer;font-size:12px;display:flex;justify-content:space-between;gap:8px;align-items:center;transition:background .15s,border-color .15s}
  .order-item:hover{background:#f8fafc}
  .order-item.selected{border-color:#7c3aed;background:#faf5ff}
  .order-item.firing{border-color:#f97316;background:#fff7ed;box-shadow:0 0 0 2px rgba(249,115,22,.18)}
  .order-item .badge-sm{font-size:10px;padding:1px 7px;border-radius:999px;background:#f1f5f9;color:#475569}
  .order-item .badge-sm.perception{background:#dbeafe;color:#1e40af}
  .order-item .badge-sm.decision{background:#fed7aa;color:#9a3412}
  .order-item .badge-sm.policy{background:#bbf7d0;color:#166534}
  .order-item .badge-sm.advisor{background:#e9d5ff;color:#6b21a8}
  .order-item .verdict.ok{color:var(--ok)} .order-item .verdict.err{color:var(--err)}
  .order-item .conf{font-variant-numeric:tabular-nums;font-size:10px;color:var(--muted)}
  pre{margin:0;white-space:pre-wrap;word-break:break-word;font-size:11.5px;line-height:1.45;color:#1e293b;background:#f8fafc;border:1px solid #f1f5f9;border-radius:6px;padding:8px}
  textarea{width:100%;height:74px;resize:vertical;border:1px solid var(--line);border-radius:6px;padding:8px;font:inherit}
  button{font:inherit;border:1px solid #94a3b8;background:#fff;border-radius:6px;padding:6px 10px;cursor:pointer}
  button.primary{border-color:#7c3aed;background:#7c3aed;color:#fff}
  .node{cursor:pointer;stroke:#fff;stroke-width:1.4;transition:stroke .15s}
  .node.selected{stroke:#0f172a;stroke-width:3}
  .node.ring{stroke:#f59e0b;stroke-width:3}
  .core{pointer-events:none}
  .node.flash{animation:flashpulse .8s ease-out}
  @keyframes flashpulse{0%{stroke:#f59e0b;stroke-width:6}100%{stroke:#fff;stroke-width:1.4}}
  .edge{fill:none;stroke-opacity:.5;transition:stroke-opacity .15s,stroke-width .25s}
  .edge.hilite{stroke-opacity:1}
  .coupling-line{fill:none;stroke:var(--coupling);stroke-dasharray:7 4;stroke-width:2.4;cursor:pointer;opacity:.75}
  .coupling-line:hover,.coupling-line.hilite{opacity:1;stroke-width:4}
  .nlabel{font-size:10px;fill:#334155;pointer-events:none;text-anchor:middle;dominant-baseline:central}
  .vlabel{font-size:9.5px;fill:#b45309;pointer-events:none;text-anchor:middle;dominant-baseline:hanging;font-variant-numeric:tabular-nums}
  .region-frame{fill:#ffffff;stroke:#cbd5e1;stroke-dasharray:5 4}
  .region-frame.focused{stroke:#7c3aed;stroke-width:2;stroke-dasharray:none}
  .sub-frame{fill:none;stroke:#e2e8f0}
  .ttl{font-size:14px;fill:#334155;font-weight:700}
  .sub-ttl{font-size:11px;fill:#94a3b8;font-weight:700;letter-spacing:1.5px}
  .cf-edge.delegates{stroke:#7c3aed;stroke-dasharray:7 5}
  .cf-edge.data_flow{stroke:#f97316}
  .cf-edge.invokes{stroke:#475569;stroke-dasharray:3 4}
  .read-arrow{stroke:#16a34a;stroke-width:2;fill:none;stroke-dasharray:2 3}
  .write-arrow{stroke:#ea580c;stroke-width:2.4;fill:none}
  .cf-rect{transition:fill .2s,stroke .2s}
  .cf-rect.firing{fill:#fef3c7 !important;stroke:#d97706 !important;stroke-width:3 !important}
  .channel-note{font-size:11px;fill:#94a3b8}
  @media (max-width:1180px){.shell{grid-template-columns:1fr}aside{border-left:0;border-top:1px solid var(--line)}svg{min-height:760px}}
</style>
</head>
<body>
<div class="hero">
  <h1 id="title"></h1>
  <div class="thesis" id="thesis"></div>
  <div class="badges" id="badges"></div>
</div>
<div class="chapters" id="chapters"></div>
<div class="playbar" id="playbar"></div>
<div class="shell">
  <div class="canvas-wrap">
    <div class="toolbar" id="toolbar"></div>
    <div class="legend" id="legend"></div>
    <svg id="canvas" viewBox="0 0 1600 1200" preserveAspectRatio="xMidYMin meet">
      <defs>
        <marker id="aV" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="#7c3aed"/></marker>
        <marker id="aO" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="#f97316"/></marker>
        <marker id="aG" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="#475569"/></marker>
        <marker id="aRead" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="#16a34a"/></marker>
        <marker id="aWrite" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="#ea580c"/></marker>
      </defs>
    </svg>
  </div>
  <aside>
    <div class="section"><h2>本章在讲什么</h2><div class="meta" id="chapterDesc"></div></div>
    <div class="section"><h2>选中对象</h2><pre id="inspector">点击节点 / 边 / 代理 / 区 查看详情。</pre></div>
    <div class="section"><h2>执行顺序 · 提案 · 校验</h2><div id="executionOrder"></div></div>
    <div class="section"><h2>网络度量</h2><div id="metrics"></div></div>
    <div class="section"><h2>校验</h2><div id="errors"></div></div>
    <div class="section">
      <h2>Human-as-LLM · 你是 llm_advisor</h2>
      <div class="meta" id="advisorContext"></div>
      <textarea id="humanInput" placeholder="综合各专家代理的建议,给出最终引导。可写明充电站,例如:建议去 S5,电价更低且队列短。"></textarea>
      <div style="margin-top:6px;display:flex;gap:6px">
        <button class="primary" id="submitAdvice">提交建议</button>
        <button id="clearAdvice">清空</button>
      </div>
      <div id="advisorVerdict" style="margin-top:7px"></div>
      <pre id="proposal" style="margin-top:7px"></pre>
    </div>
  </aside>
</div>
<script>
const data = __TOPOLOGY_DATA__;
const svg = document.getElementById("canvas");
const NS = "http://www.w3.org/2000/svg";
const VW = 1600, VH = 1200;

const KIND_COLOR = {
  station:"#dc2626", substation:"#0891b2",
  residential_load:"#b45309", commercial_load:"#7c3aed", work_load:"#0369a1",
  residential:"#b45309", commercial:"#7c3aed", work:"#0369a1",
  grid_node:"#2563eb",
  agent:"#7c3aed", subagent:"#f97316", tool:"#475569",
  city:"#1e293b", region:"#2563eb"
};
const LAYER_COLOR = { grid:"#2563eb", traffic:"#0d9488", coupling:"#9333ea", control:"#7c3aed" };

const CHAPTERS = [
  { id:"city",   label:"① 城市拓扑",
    desc:"市 → 区。一座城市由若干个区组成;每个区本身就是一套完整的「电网 + 交通」数字孪生。当前是 1 个市、2 个区:District B 与 District A 拓扑同构(B 是 A 加了一点高斯扰动的副本),所以两区之间画了一条「拓扑同构」边。单击某个区聚焦,双击进入它的内部拓扑。" },
  { id:"region", label:"② 区域孪生(电网×交通)",
    desc:"把电网层和交通层画在同一张图上。7 个充电站是两层唯一的物理交汇点,用粗紫线显式连接(interlayer 边)。节点越大 / 内核越亮,表示它当前承载的量越大(负荷 / 车流 / 电价);拖动下方时间轴看一天 24 小时内的演化。" },
  { id:"agent",  label:"③ 智能体如何调控孪生",
    desc:"一个 Root Agent 调度感知 / 决策 / 策略 / 顾问代理,按 capability 依赖拓扑排序执行。每个代理从孪生体「读取」一些量(SOC、电价、车流…),做预测与决策,再把结果「写回」孪生体、改变它的 state。点击代理看它读/写哪些量;点「▶ 走一遍决策」逐步播放整条链路。" }
];

const links = data.agent_twin_links || {};
const records = data.records || [];
const cfGraph = data.control_flow_graph;
const timeline = data.timeline || [];
const regions = data.metadata.regions;

const state = {
  chapter: "city",
  focusRegion: regions[0] ? regions[0].region_id : null,
  showStations: true,
  showCoupling: true,
  showValues: true,
  timeIndex: -1,            // -1 => static snapshot state; >=0 => timeline frame
  playing: false, timer: null,
  firingIdx: -1, decisionTimer: null,
  selectedId: null,
  highlightAgentId: null
};

function el(tag, attrs={}){ const n=document.createElementNS(NS,tag); for(const k in attrs) n.setAttribute(k,attrs[k]); return n; }
function add(node){ svg.appendChild(node); return node; }
function clearSvg(){ [...svg.children].forEach(c=>{ if(c.tagName!=="defs") svg.removeChild(c); }); }
function recordFor(id){ return records.find(r=>r.agent_id===id); }
function geo(layer, regionId){ return (data.geo_graphs||[]).find(g=>g.layer===layer&&g.region_id===regionId); }
function couplingEdges(regionId){ const g=geo("coupling",regionId); return g?g.edges:[]; }
function cfNode(id){ return cfGraph ? cfGraph.nodes.find(n=>n.id===id) : null; }
function groupOf(id){ const n=cfNode(id); return (n&&n.metadata&&n.metadata.group)||"advisor"; }
function regionLabel(id){ const r=regions.find(x=>x.region_id===id); return r?r.label:id; }

// ---- effective node state (static snapshot OR timeline frame) ----
function frameEntry(regionId){
  if(state.timeIndex<0 || !timeline.length) return null;
  const f = timeline[Math.min(state.timeIndex, timeline.length-1)];
  return f && f.regions ? f.regions[regionId] : null;
}
function effState(node, regionId){
  const base = node.state || {};
  const ent = frameEntry(regionId);
  if(!ent) return base;
  if(node.layer==="traffic"){
    const idx = node.metadata && node.metadata.node_index;
    const v = ent.traffic_inflow ? ent.traffic_inflow[String(idx)] : undefined;
    const out = Object.assign({}, base); if(v!==undefined) out.inflow=v; return out;
  }
  // grid
  const out = Object.assign({}, base);
  if(node.kind==="substation" && ent.grid_renewable!==undefined) out.renewable_kw=ent.grid_renewable;
  else if(ent.grid_load && ent.grid_load[node.kind]!==undefined) out.load_kw=ent.grid_load[node.kind];
  if(node.kind==="station" && node.metadata && node.metadata.station_index!==undefined && ent.grid_prices){
    const p = ent.grid_prices[String(node.metadata.station_index)]; if(p!==undefined) out.price=p;
  }
  return out;
}
function primaryStat(s){
  if(typeof s.price==="number") return {key:"price", v:s.price, fmt:"¥"+s.price.toFixed(2)};
  if(typeof s.inflow==="number") return {key:"inflow", v:s.inflow, fmt:Math.round(s.inflow)+""};
  if(typeof s.renewable_kw==="number") return {key:"renewable_kw", v:s.renewable_kw, fmt:s.renewable_kw.toFixed(0)};
  if(typeof s.load_kw==="number") return {key:"load_kw", v:s.load_kw, fmt:"×"+s.load_kw.toFixed(2)};
  return null;
}
function heatColor(t){ // 0..1 -> yellow to deep red
  t=Math.max(0,Math.min(1,t));
  const r=Math.round(250-30*t), g=Math.round(204-180*t), b=Math.round(21+9*t);
  return `rgb(${r},${g},${b})`;
}

// ---- geometry helpers ----
function bbox(nodes){ const xs=nodes.map(n=>n.x), ys=nodes.map(n=>n.y);
  return {x0:Math.min(...xs),x1:Math.max(...xs),y0:Math.min(...ys),y1:Math.max(...ys)}; }
function fitInto(nodes, box){ // returns map id -> {x,y}; preserves aspect
  if(!nodes.length) return new Map();
  const bb=bbox(nodes); const dw=(bb.x1-bb.x0)||1, dh=(bb.y1-bb.y0)||1;
  const sc=Math.min((box.w)/dw,(box.h)/dh)*0.92;
  const ox=box.x+(box.w-dw*sc)/2, oy=box.y+(box.h-dh*sc)/2;
  const m=new Map(); nodes.forEach(n=>m.set(n.id,{x:ox+(n.x-bb.x0)*sc, y:oy+(n.y-bb.y0)*sc})); return m;
}

// ---- node drawing (shared by region + agent twin-band) ----
function drawGeoNode(node, regionId, pos, opts){
  opts = opts || {};
  const baseR = opts.scale || 1;
  const isStation = node.kind==="station";
  const s = effState(node, regionId);
  const ps = primaryStat(s);
  // intensity normalized within this kind across the graph at the current frame
  let intensity = 0;
  if(ps){
    const g = geo(node.layer, regionId);
    let mx = 1e-9;
    g.nodes.forEach(o=>{ const o2=primaryStat(effState(o,regionId)); if(o2 && o2.key===ps.key) mx=Math.max(mx,o2.v); });
    intensity = mx>0 ? ps.v/mx : 0;
  }
  const r = ((isStation||node.kind==="substation")?7:5)*baseR + 7*intensity*baseR;
  const fill = KIND_COLOR[node.kind] || LAYER_COLOR[node.layer] || "#888";
  const hi = (state.highlightAgentId && agentTouchesNode(state.highlightAgentId, node, regionId));
  const c = el("circle",{cx:pos.x, cy:pos.y, r:r.toFixed(1), class:"node"+(isStation&&state.showStations?" ring":"")+(node.id===state.selectedId?" selected":""), fill, "fill-opacity":0.92, "data-id":node.id});
  c.addEventListener("click",()=>inspect({kind:"node",layer:node.layer,region_id:regionId,data:node,effective_state:s}, c));
  add(c);
  if(intensity>0.04){ const core=el("circle",{cx:pos.x,cy:pos.y,r:(1.5+5.5*intensity*baseR).toFixed(1),class:"core",fill:heatColor(intensity),"fill-opacity":0.95}); add(core); }
  if(hi){ const ring=el("circle",{cx:pos.x,cy:pos.y,r:(r+4).toFixed(1),fill:"none",stroke:"#16a34a","stroke-width":2.5,"stroke-dasharray":"2 2"}); add(ring); }
  if(isStation||node.kind==="substation"){ const t=el("text",{x:pos.x,y:pos.y-r-5,class:"nlabel"}); t.textContent=node.label; add(t); }
  if(state.showValues && ps && baseR>=0.85){ const t=el("text",{x:pos.x,y:pos.y+r+3,class:"vlabel"}); t.textContent=ps.fmt; add(t); }
  return r;
}

function agentTouchesNode(agentId, node, regionId){
  const lk = links[agentId]; if(!lk) return false;
  const touches = new Set([...(lk.reads||[]),...(lk.writes||[])]);
  if(node.layer==="grid" && touches.has("grid")) return true;
  if(node.layer==="traffic" && touches.has("traffic")) return true;
  return false;
}

// =========================================================
// CHAPTER 1: CITY
// =========================================================
function renderCity(){
  const cg = data.city_graph || {nodes:[],edges:[]};
  const city = cg.nodes.find(n=>n.kind==="city");
  const regs = cg.nodes.filter(n=>n.kind==="region");
  const cityPos = {x:340, y:VH/2};
  const regPos = new Map();
  regs.forEach((r,i)=> regPos.set(r.id,{x:980, y: VH/2 + (i-(regs.length-1)/2)*260}));
  // edges
  cg.edges.forEach(e=>{
    const a = e.source===city.id?cityPos:regPos.get(e.source);
    const b = e.target===city.id?cityPos:regPos.get(e.target);
    if(!a||!b) return;
    if(e.kind==="contains"){
      const p=el("path",{d:`M${a.x+170},${a.y} C${(a.x+b.x)/2},${a.y} ${(a.x+b.x)/2},${b.y} ${b.x-170},${b.y}`,class:"edge",stroke:"#94a3b8","stroke-width":2.5,"marker-end":"url(#aG)"});
      add(p);
      const t=el("text",{x:(a.x+b.x)/2, y:(a.y+b.y)/2-8, class:"channel-note","text-anchor":"middle"}); t.textContent="包含 (contains)"; add(t);
    } else if(e.kind==="shares_schema"){
      const p=el("path",{d:`M${a.x},${a.y+96} C${a.x+140},${(a.y+b.y)/2} ${b.x+140},${(a.y+b.y)/2} ${b.x},${b.y-96}`,class:"edge",stroke:"#0d9488","stroke-width":1.8,"stroke-dasharray":"6 4"});
      add(p);
      const tx=(a.x+b.x)/2+110, ty=(a.y+b.y)/2;
      const t1=el("text",{x:tx,y:ty-6,class:"channel-note","text-anchor":"middle",style:"fill:#0d9488"}); t1.textContent="拓扑同构"; add(t1);
      const t2=el("text",{x:tx,y:ty+9,class:"channel-note","text-anchor":"middle","font-size":10,style:"fill:#94a3b8"}); t2.textContent="District B = District A + 扰动"; add(t2);
    }
  });
  // city node — light fill, dark text
  const cw=320, ch=110;
  const cr=el("rect",{x:cityPos.x-cw/2, y:cityPos.y-ch/2, width:cw, height:ch, rx:16, fill:"#e0e7ff", stroke:"#4338ca", "stroke-width":3.5, class:"node","data-id":city.id});
  cr.addEventListener("click",()=>inspect({kind:"city",data:{id:city.id, regions:regs.length}},cr)); add(cr);
  let t=el("text",{x:cityPos.x,y:cityPos.y-22,class:"nlabel","font-size":19,"font-weight":700}); t.textContent="🏙  城市 · "+city.id; add(t);
  t=el("text",{x:cityPos.x,y:cityPos.y+2,class:"nlabel","font-size":12.5,style:"fill:#475569"}); t.textContent=regs.length+" 个区"; add(t);
  t=el("text",{x:cityPos.x,y:cityPos.y+22,class:"nlabel","font-size":12.5,style:"fill:#475569"}); t.textContent="每个区 = 一套完整孪生(电网+交通)"; add(t);
  // region nodes
  regs.forEach(r=>{
    const p=regPos.get(r.id); const w=340,h=190;
    const focused = r.id===state.focusRegion;
    const rect=el("rect",{x:p.x-w/2,y:p.y-h/2,width:w,height:h,rx:16,fill:focused?"#faf5ff":"#fff",stroke:focused?"#7c3aed":"#2563eb","stroke-width":focused?3.5:2.5,class:"node","data-id":r.id});
    rect.addEventListener("click",()=>{ state.focusRegion=r.id; renderToolbar(); inspect({kind:"region",data:r},rect); render(); });
    rect.addEventListener("dblclick",()=>{ state.focusRegion=r.id; setChapter("region"); });
    add(rect);
    let tt=el("text",{x:p.x,y:p.y-h/2+30,class:"nlabel","font-size":17,"font-weight":700}); tt.textContent="📍  "+r.label; add(tt);
    const lines=[`${r.stats.nodes} 节点 · ${r.stats.edges} 边`,`${r.stats.stations} 充电站(电网 ∩ 交通)`,`电网层 + 交通层 + 耦合层`];
    lines.forEach((ln,i)=>{ const x=el("text",{x:p.x,y:p.y-h/2+62+i*24,class:"nlabel","font-size":13,style:"fill:#475569"}); x.textContent=ln; add(x); });
    const hint=el("text",{x:p.x,y:p.y+h/2-16,class:"channel-note","font-size":11.5}); hint.textContent="单击聚焦 · 双击进入区域孪生 ▸"; add(hint);
  });
}

// =========================================================
// CHAPTER 2: REGION (grid x traffic in one figure)
// =========================================================
function regionLayout(regionId, box){
  // box: {x,y,w,h}. left half = grid, right half = traffic, gap in middle for coupling.
  const gridG = geo("grid", regionId), trafG = geo("traffic", regionId);
  const gapFrac = 0.16;
  const halfW = box.w*(1-gapFrac)/2;
  const gridBox = {x:box.x, y:box.y, w:halfW, h:box.h};
  const trafBox = {x:box.x+halfW+box.w*gapFrac, y:box.y, w:halfW, h:box.h};
  const gpos = gridG ? fitInto(gridG.nodes, gridBox) : new Map();
  const tpos = trafG ? fitInto(trafG.nodes, trafBox) : new Map();
  const pos = new Map();
  for(const [k,v] of gpos) pos.set(k,{x:v.x,y:v.y,layer:"grid"});
  for(const [k,v] of tpos) pos.set(k,{x:v.x,y:v.y,layer:"traffic"});
  return {pos, gridBox, trafBox};
}

function renderRegion(){
  const regionId = state.focusRegion;
  const box = {x:60, y:96, w:VW-120, h:VH-150};
  // region frame
  add(el("rect",{x:box.x,y:box.y,width:box.w,height:box.h,rx:16,class:"region-frame focused"}));
  let t=el("text",{x:box.x+16,y:box.y+26,class:"ttl"}); t.textContent="📍 "+regionLabel(regionId)+" · 电网层 × 交通层(同一张拓扑)"; add(t);
  const L = regionLayout(regionId, {x:box.x+30, y:box.y+50, w:box.w-60, h:box.h-80});
  // sub-frames + titles
  add(el("rect",{x:L.gridBox.x-14,y:L.gridBox.y-14,width:L.gridBox.w+28,height:L.gridBox.h+28,rx:12,class:"sub-frame"}));
  add(el("rect",{x:L.trafBox.x-14,y:L.trafBox.y-14,width:L.trafBox.w+28,height:L.trafBox.h+28,rx:12,class:"sub-frame"}));
  let g=el("text",{x:L.gridBox.x,y:L.gridBox.y-22,class:"sub-ttl",fill:LAYER_COLOR.grid}); g.textContent="电网层  GRID"; add(g);
  let tr=el("text",{x:L.trafBox.x,y:L.trafBox.y-22,class:"sub-ttl",fill:LAYER_COLOR.traffic}); tr.textContent="交通层  TRAFFIC"; add(tr);
  // intralayer edges
  ["grid","traffic"].forEach(layer=>{
    const G = geo(layer, regionId); if(!G) return;
    const ent = frameEntry(regionId);
    G.edges.forEach(e=>{
      const a=L.pos.get(e.source), b=L.pos.get(e.target); if(!a||!b) return;
      let sw;
      if(layer==="traffic"){
        let w = e.weight||1;
        if(ent && ent.traffic_edges && ent.traffic_edges[e.id]!==undefined) w = ent.traffic_edges[e.id];
        sw = Math.max(1, Math.min(7, w/35));
      } else sw = 1.6;
      const p=el("path",{d:`M${a.x},${a.y} L${b.x},${b.y}`,class:"edge",stroke:LAYER_COLOR[layer],"stroke-width":sw,"data-eid":e.id});
      p.addEventListener("click",()=>inspect({kind:"edge",layer,region_id:regionId,data:e},p)); add(p);
    });
  });
  // coupling (interlayer) edges
  if(state.showCoupling){
    couplingEdges(regionId).forEach(e=>{
      const a=L.pos.get(e.source), b=L.pos.get(e.target); if(!a||!b) return;
      const mx=(a.x+b.x)/2, my=(a.y+b.y)/2-40;
      const p=el("path",{d:`M${a.x},${a.y} Q${mx},${my} ${b.x},${b.y}`,class:"coupling-line"+(state.highlightAgentId?"":" hilite"),"data-id":e.id});
      p.addEventListener("click",()=>inspect({kind:"coupling_edge",region_id:regionId,data:e},p)); add(p);
      const lbl=el("text",{x:mx,y:my-3,class:"nlabel",fill:LAYER_COLOR.coupling,"font-size":11,"font-weight":700}); lbl.textContent=(e.metadata&&e.metadata.label)||("S"+(e.metadata&&e.metadata.station_index)); add(lbl);
    });
  }
  // nodes (draw after edges)
  ["grid","traffic"].forEach(layer=>{
    const G = geo(layer, regionId); if(!G) return;
    G.nodes.forEach(n=>{ const p=L.pos.get(n.id); if(p) drawGeoNode(n, regionId, p, {scale:1}); });
  });
}

// =========================================================
// CHAPTER 3: AGENT controlling the twin
// =========================================================
function cfLayout(box){
  if(!cfGraph) return {pos:new Map(), groups:[], colW:0};
  const root = cfGraph.nodes.find(n=>n.kind==="agent");
  const subs = cfGraph.nodes.filter(n=>n.kind==="subagent");
  const tools = cfGraph.nodes.filter(n=>n.kind==="tool");
  const groups = ["perception","decision","policy","advisor"].filter(g=>subs.some(n=>groupOf(n.id)===g));
  const colW = box.w/Math.max(groups.length,1);
  const pos=new Map();
  if(root) pos.set(root.id,{x:box.x+box.w/2, y:box.y+24, node:root});
  groups.forEach((grp,gi)=>{
    const mem = subs.filter(n=>groupOf(n.id)===grp);
    const cx = box.x+(gi+0.5)*colW;
    const rowH = Math.min(70, (box.h-130)/Math.max(mem.length,1));
    mem.forEach((n,i)=> pos.set(n.id,{x:cx, y:box.y+92+i*rowH, node:n}));
  });
  const maxRows = Math.max(...groups.map(g=>subs.filter(n=>groupOf(n.id)===g).length),1);
  const toolsY = box.y+92+maxRows*Math.min(70,(box.h-130)/Math.max(maxRows,1))+24;
  tools.forEach((n,i)=> pos.set(n.id,{x:box.x+box.w*(0.32+0.36*i), y:Math.min(toolsY, box.y+box.h-20), node:n}));
  return {pos, groups, colW, root, tools};
}

function renderAgent(){
  const regionId = state.focusRegion;
  const cfBox = {x:60, y:96, w:VW-120, h:560};
  const twBox = {x:60, y:700, w:VW-120, h:VH-720};
  // ----- divider + titles -----
  let t=el("text",{x:cfBox.x,y:cfBox.y+2,class:"sub-ttl",fill:"#7c3aed"}); t.textContent="决策流  ROOT AGENT · SUBAGENTS · TOOLS"; add(t);
  add(el("line",{x1:30,y1:twBox.y-22,x2:VW-30,y2:twBox.y-22,stroke:"#cbd5e1","stroke-dasharray":"8 6"}));
  let t2=el("text",{x:twBox.x,y:twBox.y-4,class:"sub-ttl",fill:"#0d9488"}); t2.textContent="数字孪生(被调控的对象)· "+regionLabel(regionId); add(t2);
  // ----- twin band (compact combined region) -----
  const L = regionLayout(regionId, {x:twBox.x+30, y:twBox.y+14, w:twBox.w-60, h:twBox.h-30});
  ["grid","traffic"].forEach(layer=>{
    const G=geo(layer,regionId); if(!G) return;
    G.edges.forEach(e=>{ const a=L.pos.get(e.source),b=L.pos.get(e.target); if(!a||!b)return;
      add(el("path",{d:`M${a.x},${a.y} L${b.x},${b.y}`,class:"edge",stroke:LAYER_COLOR[layer],"stroke-width":layer==="traffic"?1.4:1,"stroke-opacity":.3})); });
  });
  if(state.showCoupling) couplingEdges(regionId).forEach(e=>{ const a=L.pos.get(e.source),b=L.pos.get(e.target); if(!a||!b)return;
    add(el("path",{d:`M${a.x},${a.y} L${b.x},${b.y}`,class:"coupling-line"})); });
  ["grid","traffic"].forEach(layer=>{ const G=geo(layer,regionId); if(!G)return;
    G.nodes.forEach(n=>{ const p=L.pos.get(n.id); if(p) drawGeoNode(n, regionId, p, {scale:0.78}); }); });
  let gl=el("text",{x:L.gridBox.x,y:twBox.y+8,class:"sub-ttl",fill:LAYER_COLOR.grid,"font-size":10}); gl.textContent="GRID"; add(gl);
  let tl=el("text",{x:L.trafBox.x,y:twBox.y+8,class:"sub-ttl",fill:LAYER_COLOR.traffic,"font-size":10}); tl.textContent="TRAFFIC"; add(tl);
  const twGridAnchor = {x:L.gridBox.x+L.gridBox.w/2, y:twBox.y+14};
  const twTrafAnchor = {x:L.trafBox.x+L.trafBox.w/2, y:twBox.y+14};
  // ----- control-flow graph -----
  const CF = cfLayout(cfBox);
  if(CF.groups){ CF.groups.forEach((grp,gi)=>{ const x=cfBox.x+(gi+0.5)*CF.colW;
    const tt=el("text",{x,y:cfBox.y+74,class:"sub-ttl","text-anchor":"middle","font-size":10}); tt.textContent=grp.toUpperCase(); add(tt); }); }
  cfGraph && cfGraph.edges.forEach(e=>{ const a=CF.pos.get(e.source),b=CF.pos.get(e.target); if(!a||!b)return;
    const my=(a.y+b.y)/2;
    const p=el("path",{d:`M${a.x},${a.y} C${a.x},${my} ${b.x},${my} ${b.x},${b.y}`,class:`edge cf-edge ${e.kind}`,"stroke-width":1.5,
      "marker-end":e.kind==="data_flow"?"url(#aO)":(e.kind==="invokes"?"url(#aG)":"url(#aV)"),"data-eid":e.id});
    p.addEventListener("click",()=>inspect({kind:"cf_edge",data:e},p)); add(p);
  });
  // ----- read/write arrows for the highlighted (or firing) agent -----
  const focusAgent = state.firingIdx>=0 ? data.execution_order[state.firingIdx] : state.highlightAgentId;
  if(focusAgent && links[focusAgent]){
    const ap = CF.pos.get(focusAgent);
    const lk = links[focusAgent];
    if(ap){
      (lk.read_detail||[]).forEach((d,i)=>{
        if(!d.layer) return;
        const src = d.layer==="grid"?twGridAnchor:twTrafAnchor;
        const off = (i-((lk.read_detail.length-1)/2))*46;
        const p=el("path",{d:`M${src.x+off},${src.y} C${src.x+off},${(src.y+ap.y)/2} ${ap.x},${(src.y+ap.y)/2} ${ap.x},${ap.y+22}`,class:"read-arrow","marker-end":"url(#aRead)"}); add(p);
        const t=el("text",{x:(src.x+off+ap.x)/2, y:(src.y+ap.y)/2-2, class:"channel-note",fill:"#16a34a","font-size":11,"text-anchor":"middle"}); t.textContent="读 "+d.label; add(t);
      });
      (lk.write_detail||[]).forEach((d,i)=>{
        if(!d.layer) return;
        const dst = d.layer==="grid"?twGridAnchor:twTrafAnchor;
        const off = (i-((lk.write_detail.length-1)/2))*46 + 18;
        const p=el("path",{d:`M${ap.x+18},${ap.y+22} C${ap.x+18},${(dst.y+ap.y)/2} ${dst.x+off},${(dst.y+ap.y)/2} ${dst.x+off},${dst.y}`,class:"write-arrow","marker-end":"url(#aWrite)"}); add(p);
        const t=el("text",{x:(ap.x+dst.x+off)/2, y:(dst.y+ap.y)/2+12, class:"channel-note",fill:"#ea580c","font-size":11,"text-anchor":"middle"}); t.textContent="写 "+d.label; add(t);
      });
    }
  } else {
    const note=el("text",{x:VW/2, y:twBox.y-40, class:"channel-note","text-anchor":"middle","font-size":12.5}); note.textContent="↑ 点击任一代理,或点「▶ 走一遍决策」—— 看它从孪生体读什么量、又把什么写回去 ↓"; add(note);
  }
  // ----- cf nodes -----
  cfGraph && cfGraph.nodes.forEach(n=>{
    const p=CF.pos.get(n.id); if(!p)return;
    const isRoot=n.kind==="agent", isTool=n.kind==="tool";
    const w=isRoot?210:(isTool?150:158), h=isRoot?42:36;
    const firing = state.firingIdx>=0 && data.execution_order[state.firingIdx]===n.id;
    const rec = recordFor(n.id);
    const accepted = rec && rec.validation && rec.validation.accepted;
    const rect=el("rect",{x:p.x-w/2,y:p.y-h/2,width:w,height:h,rx:9,
      fill:isRoot?"#faf5ff":(isTool?"#f8fafc":(rec?(accepted?"#f0fdf4":"#fef2f2"):"#fff7ed")),
      stroke:KIND_COLOR[n.kind]||"#7c3aed","stroke-width":isRoot?2.2:1.6,
      class:"cf-rect node"+(firing?" firing":"")+(n.id===state.selectedId?" selected":""),"data-id":n.id});
    rect.addEventListener("click",()=>{ state.highlightAgentId = (n.kind==="subagent")?n.id:null; state.selectedId=n.id;
      inspect({kind:"cf_node",data:n,twin_link:links[n.id]||null,record:rec||null},rect);
      if(n.kind==="subagent") highlightInOrder(n.id);
      render(); });
    add(rect);
    let lt=el("text",{x:p.x,y:p.y-3,class:"nlabel","font-size":11}); lt.textContent=n.label; add(lt);
    let st=el("text",{x:p.x,y:p.y+10,class:"nlabel","font-size":9.5,fill:"#94a3b8"}); st.textContent=(n.metadata&&(n.metadata.agent_type||n.metadata.group))||n.kind; add(st);
  });
}

// =========================================================
// MAIN RENDER
// =========================================================
function render(){
  clearSvg();
  if(state.chapter==="city") renderCity();
  else if(state.chapter==="region") renderRegion();
  else if(state.chapter==="agent") renderAgent();
}

// ---- hero / chapters / toolbar / legend ----
function renderHero(){
  document.getElementById("title").textContent=data.metadata.title;
  document.getElementById("thesis").textContent=data.metadata.thesis;
  const m=data.summary_metrics;
  const errs=data.errors.control_flow.length+data.errors.geo.length;
  const accepted=records.filter(r=>r.validation&&r.validation.accepted).length;
  const b=[
    `<span class="badge"><b>${regions.length}</b> 区 / 1 市</span>`,
    `<span class="badge"><b>${m.node_total}</b> 物理节点</span>`,
    `<span class="badge"><b>${m.edge_total}</b> 物理边</span>`,
    `<span class="badge"><b>${m.interlayer_edges}</b> 跨层耦合边</span>`,
    `<span class="badge"><b>${m.mean_avg_degree.toFixed(2)}</b> 平均度</span>`,
    `<span class="badge"><b>${m.max_diameter??"—"}</b> 最大直径</span>`,
    `<span class="badge"><b>${m.control_flow?m.control_flow.node_count:0}</b> 决策流节点</span>`,
    `<span class="badge ${accepted===records.length?"ok":"warn"}"><b>${accepted}/${records.length}</b> 提案通过</span>`,
    `<span class="badge ${errs===0?"ok":"err"}"><b>${errs}</b> 校验错</span>`
  ];
  document.getElementById("badges").innerHTML=b.join("");
}
function renderChapters(){
  const c=document.getElementById("chapters"); c.innerHTML="";
  CHAPTERS.forEach(ch=>{ const d=document.createElement("div"); d.className="chapter"+(ch.id===state.chapter?" active":"");
    d.textContent=ch.label; d.addEventListener("click",()=>setChapter(ch.id)); c.appendChild(d); });
  const ch=CHAPTERS.find(x=>x.id===state.chapter);
  document.getElementById("chapterDesc").textContent=ch?ch.desc:"";
}
function setChapter(id){
  state.chapter=id; state.timeIndex=-1; stopPlay(); stopDecision();
  state.highlightAgentId=null;
  renderChapters(); renderToolbar(); renderLegend(); renderPlaybar(); render();
}
function renderToolbar(){
  const tb=document.getElementById("toolbar"); tb.innerHTML="";
  if(state.chapter==="region" || state.chapter==="agent"){
    const rg=document.createElement("div"); rg.className="group"; rg.innerHTML="<b>区</b>";
    regions.forEach(r=>{ const ch=document.createElement("span"); ch.className="chip"+(r.region_id===state.focusRegion?" on":"");
      ch.textContent=r.label; ch.addEventListener("click",()=>{ state.focusRegion=r.region_id; renderToolbar(); render(); }); rg.appendChild(ch); });
    tb.appendChild(rg);
  }
  if(state.chapter==="region"){
    const dg=document.createElement("div"); dg.className="group"; dg.innerHTML="<b>显示</b>";
    [["showCoupling","coupling","耦合边"],["showStations","station","充电站环"],["showValues","","数值"]].forEach(([key,cls,label])=>{
      const ch=document.createElement("span"); ch.className=`chip ${cls}`+(state[key]?" on":""); ch.textContent=label;
      ch.addEventListener("click",()=>{ state[key]=!state[key]; renderToolbar(); render(); }); dg.appendChild(ch); });
    tb.appendChild(dg);
  }
}
function renderLegend(){
  const lg=document.getElementById("legend");
  if(state.chapter==="city"){ lg.innerHTML=`<h3>图例</h3>
    <div class="row"><span class="sw" style="background:#e0e7ff;border:2px solid #4338ca"></span>城市</div>
    <div class="row"><span class="sw" style="background:#fff;border:2px solid #2563eb"></span>区(= 一套完整孪生)</div>
    <div class="row"><span class="swl" style="border-color:#94a3b8"></span>包含 contains(市 → 区)</div>
    <div class="row"><span class="swl" style="border-color:#0d9488;border-style:dashed"></span>拓扑同构(B = A + 扰动)</div>`;
    return; }
  if(state.chapter==="region" || state.chapter==="agent"){ lg.innerHTML=`<h3>图例</h3>
    <div class="row"><span class="sw" style="background:#dc2626"></span>充电站(电网∩交通)</div>
    <div class="row"><span class="sw" style="background:#0891b2"></span>变电站</div>
    <div class="row"><span class="sw" style="background:#2563eb"></span>配电节点 / 负荷</div>
    <div class="row"><span class="sw" style="background:#0d9488"></span>路网节点</div>
    <div class="row"><span class="swl" style="border-color:#2563eb"></span>配电线 &nbsp; <span class="swl" style="border-color:#0d9488"></span>路段车流</div>
    <div class="row"><span class="swl" style="border-color:#9333ea;border-style:dashed"></span>跨层耦合(站点)</div>
    <div class="row" style="margin-top:4px"><span class="sw" style="background:radial-gradient(circle,#dc2626,#fde047)"></span>内核越亮越大 = 当前承载量越高</div>
    ${state.chapter==="agent"?'<div class="row"><span class="swl" style="border-color:#16a34a;border-style:dashed"></span>代理「读」孪生量 &nbsp; <span class="swl" style="border-color:#ea580c"></span>代理「写」回孪生</div>':''}`;
  }
}

// ---- side panels ----
function inspect(entry, node){
  document.querySelectorAll(".node.selected").forEach(n=>n.classList.remove("selected"));
  if(node) node.classList.add("selected");
  if(entry && entry.data && entry.data.id) state.selectedId = entry.data.id;
  document.getElementById("inspector").textContent=JSON.stringify(entry,null,2);
}
function renderMetrics(){
  const c=document.getElementById("metrics"); const rows=[];
  (data.geo_graphs||[]).forEach(g=>{ const m=g.metrics;
    rows.push(`<div class="metric-row"><span>${g.region_id} / ${g.layer}</span><b>n=${m.node_count} e=${m.edge_count} ⟨k⟩=${m.avg_degree.toFixed(2)}${m.diameter!=null?" d="+m.diameter:""}</b></div>`); });
  if(cfGraph){ const m=cfGraph.metrics; rows.push(`<div class="metric-row" style="margin-top:5px"><span>control_flow</span><b>n=${m.node_count} e=${m.edge_count} ⟨k⟩=${m.avg_degree.toFixed(2)} d=${m.diameter}</b></div>`); }
  c.innerHTML=rows.join("");
}
function renderExecutionOrder(){
  const c=document.getElementById("executionOrder");
  c.innerHTML=records.map((r,i)=>{
    const grp=groupOf(r.agent_id);
    const accepted=r.validation&&r.validation.accepted;
    const conf=r.proposal&&r.proposal.confidence;
    const lk=links[r.agent_id]||{};
    const rw=[(lk.reads||[]).length?"读 "+(lk.reads||[]).join("/"):"",(lk.writes||[]).length?"写 "+(lk.writes||[]).join("/"):""].filter(Boolean).join(" · ");
    return `<div class="order-item${i===state.firingIdx?" firing":""}${r.agent_id===state.selectedId?" selected":""}" data-agent="${r.agent_id}">
      <div><div><b>${i+1}.</b> ${r.agent_id}</div>
      <div class="meta">${r.proposal.proposal_type}${conf!=null?'<span class="conf"> · conf='+conf+'</span>':''}${rw?'<br><span style="color:#94a3b8">'+rw+'</span>':''}</div></div>
      <div><span class="badge-sm ${grp}">${grp}</span> <span class="verdict ${accepted?'ok':'err'}">${accepted?'✓':'✗'}</span></div></div>`;
  }).join("");
  c.querySelectorAll("[data-agent]").forEach(it=>it.addEventListener("click",()=>{
    const aid=it.getAttribute("data-agent"); highlightInOrder(aid);
    state.highlightAgentId=aid; const rec=recordFor(aid);
    inspect({kind:"agent",data:rec,twin_link:links[aid]||null},null);
    if(state.chapter==="agent") render();
  }));
}
function highlightInOrder(aid){ state.selectedId=aid; renderExecutionOrder(); }
function renderErrors(){
  const e=[...(data.errors.control_flow||[]).map(m=>['control_flow',m]),...(data.errors.geo||[]).map(m=>['geo',m])];
  const c=document.getElementById("errors");
  c.innerHTML = e.length? e.map(([s,m])=>`<div class="pill err">[${s}] ${m}</div>`).join("") : `<div class="pill ok">✓ 全部校验通过</div>`;
}

// ---- playbar (context-aware) ----
function renderPlaybar(){
  const pb=document.getElementById("playbar");
  if(state.chapter==="region"){
    pb.style.display="flex";
    pb.innerHTML=`<span class="label" id="timeLabel">快照 t=8.0h</span>
      <input type="range" id="timeRange" min="-1" max="${Math.max(timeline.length-1,0)}" value="-1">
      <button id="playBtn">▶ 24h 播放</button>
      <button id="resetTime">↺ 回到快照</button>`;
    const range=document.getElementById("timeRange");
    range.addEventListener("input",()=>{ state.timeIndex=+range.value; applyTime(); render(); });
    document.getElementById("playBtn").addEventListener("click",togglePlay);
    document.getElementById("resetTime").addEventListener("click",()=>{ state.timeIndex=-1; range.value=-1; applyTime(); render(); });
    applyTime();
  } else if(state.chapter==="agent"){
    pb.style.display="flex";
    pb.innerHTML=`<span class="label" id="stepLabel">未开始</span>
      <button class="primary" id="decisionBtn">▶ 走一遍决策</button>
      <button id="stopDecisionBtn">■ 停止</button>
      <span class="meta">逐步播放:每个代理读孪生→预测/决策→写回孪生(被写的节点会闪一下)</span>`;
    document.getElementById("decisionBtn").addEventListener("click",playDecision);
    document.getElementById("stopDecisionBtn").addEventListener("click",()=>{ stopDecision(); render(); });
  } else { pb.style.display="none"; }
}
function applyTime(){
  const lab=document.getElementById("timeLabel"); if(!lab) return;
  if(state.timeIndex<0) lab.textContent="快照 t=8.0h";
  else { const f=timeline[Math.min(state.timeIndex,timeline.length-1)]; lab.textContent=f?`t = ${f.hour.toFixed(1)} h`:"—"; }
}
function togglePlay(){
  state.playing=!state.playing;
  const btn=document.getElementById("playBtn"); if(btn) btn.textContent=state.playing?"❚❚ 暂停":"▶ 24h 播放";
  if(state.timer){clearInterval(state.timer);state.timer=null;}
  if(state.playing && timeline.length){
    if(state.timeIndex<0) state.timeIndex=0;
    state.timer=setInterval(()=>{ state.timeIndex=(state.timeIndex+1)%timeline.length;
      const r=document.getElementById("timeRange"); if(r) r.value=state.timeIndex; applyTime(); render(); }, 520);
  }
}
function stopPlay(){ state.playing=false; if(state.timer){clearInterval(state.timer);state.timer=null;} }
function playDecision(){
  stopDecision(); state.firingIdx=-1; let i=0; const total=(data.execution_order||[]).length;
  const lab=document.getElementById("stepLabel");
  state.decisionTimer=setInterval(()=>{
    if(i>=total){ stopDecision(); if(lab) lab.textContent="完成 ✓"; renderExecutionOrder(); render(); return; }
    state.firingIdx=i; state.selectedId=data.execution_order[i]; state.highlightAgentId=data.execution_order[i];
    if(lab) lab.textContent=`第 ${i+1}/${total} 步: ${data.execution_order[i]}`;
    renderExecutionOrder(); render(); i++;
  }, 900);
}
function stopDecision(){ if(state.decisionTimer){clearInterval(state.decisionTimer);state.decisionTimer=null;} state.firingIdx=-1; }

// ---- Human-as-LLM ----
function pickStationFromText(txt){
  let m = txt.match(/S\s*([0-9]+)/i) || txt.match(/站\s*([0-9]+)/) || txt.match(/station[_\s-]*([0-9]+)/i);
  if(!m) return null; return parseInt(m[1],10);
}
function decisionStationIndex(){
  const rec=records.find(r=>r.agent_id==="station_decision_agent");
  if(!rec||!rec.proposal||!rec.proposal.payload) return null;
  const pp=rec.proposal.payload;
  if(Array.isArray(pp.ranked_stations)&&pp.ranked_stations.length){
    const best=[...pp.ranked_stations].sort((a,b)=>(a.score??0)-(b.score??0))[0];
    if(best&&best.station_index!=null) return best.station_index;
  }
  return pp.station_index??pp.chosen_station_index??pp.station??null;
}
function chargingMode(){
  const rec=records.find(r=>r.agent_id==="charging_mode_agent");
  if(!rec||!rec.proposal||!rec.proposal.payload) return "—";
  const pp=rec.proposal.payload;
  return pp.mode_preference||pp.mode||pp.charging_mode||"—";
}
function renderAdvisorContext(){
  const obs=data.observation||{};
  const cand=(obs.reachable_stations||[]).map(s=>`S${s.station_index+1}(行${(s.travel_time_h*60).toFixed(0)}min`+(obs.station_prices&&obs.station_prices[s.station_index]!=null?`,¥${obs.station_prices[s.station_index]}`:'')+`)`).join("、");
  const decIdx=decisionStationIndex();
  const decPick = decIdx!=null?("S"+(decIdx+1)):"—";
  const mode=chargingMode();
  document.getElementById("advisorContext").innerHTML=
    `当前:node ${obs.current_node??'—'} · t=${obs.current_time??'—'}h · SOC ${obs.soc!=null?(obs.soc*100).toFixed(0)+'%':'—'}<br>`+
    `候选站:${cand||'—'}<br>`+
    `专家代理已建议 → 充电站 <b>${decPick}</b> · 模式 <b>${mode}</b><br>`+
    `<span style="color:#94a3b8">你的任务:同意或推翻,给出最终引导。</span>`;
}
function bindHuman(){
  document.getElementById("proposal").textContent=JSON.stringify(data.human_llm_proposal,null,2);
  renderAdvisorContext();
  document.getElementById("submitAdvice").addEventListener("click",()=>{
    const advice=document.getElementById("humanInput").value.trim()||"(空)";
    const pick=pickStationFromText(advice);
    const obs=data.observation||{};
    const reachable = pick!=null && (obs.reachable_stations||[]).some(s=>s.station_index===pick-1);
    const decIdx=decisionStationIndex();
    const proposal={
      agent_name:"human_llm_agent", proposal_type:"llm_advice",
      payload:{ source:"human", summary:advice, recommended_station: pick!=null?("S"+pick):null, station_index: pick!=null?pick-1:null,
        consistent_with_decision_agent: (pick!=null&&decIdx!=null)?(pick-1===decIdx):null,
        observed:["reachable_stations","station_prices","specialist_proposals"] },
      confidence:null, rationale:"浏览器内人类扮演 LLM 顾问的回复", metadata:{agent_family:"llm",interactive:true}
    };
    document.getElementById("proposal").textContent=JSON.stringify(proposal,null,2);
    const v=document.getElementById("advisorVerdict"); const bits=[];
    if(pick==null) bits.push(`<span class="pill warn">未识别到充电站编号(写明「Sx」可触发一致性检查)</span>`);
    else{
      bits.push(reachable?`<span class="pill ok">✓ S${pick} 在可达集合内</span>`:`<span class="pill err">✗ S${pick} 不在当前可达集合</span>`);
      if(decIdx!=null) bits.push(pick-1===decIdx?`<span class="pill ok">与 station_decision_agent 一致</span>`:`<span class="pill warn">与 station_decision_agent 分歧(代理选 S${decIdx+1})</span>`);
    }
    v.innerHTML=bits.join(" ");
  });
  document.getElementById("clearAdvice").addEventListener("click",()=>{ document.getElementById("humanInput").value=""; document.getElementById("advisorVerdict").innerHTML=""; });
}

// ---- boot ----
renderHero(); renderChapters(); renderToolbar(); renderLegend();
renderMetrics(); renderExecutionOrder(); renderErrors(); renderPlaybar(); bindHuman();
render();
</script>
</body>
</html>
"""
