"""Biologically consolidated measurements derived from a colony skeleton.

These functions contain the stable metrics developed in the exploratory
notebook. Junction nodes linked by short graph paths are treated as one hub;
internal hub paths are excluded from the biological branch list.
"""

from collections import Counter
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

from cldm.morpho_features import summarize_skeleton


@dataclass(frozen=True)
class BiologicalMetricConfig:
    """Tunable parameters for graph consolidation and biological metrics."""

    hub_merge_distance_px: float | None = None
    hub_merge_width_factor: float = 1.0
    min_branch_length_px: float = 3.0
    boundary_tip_margin_px: float = 5.0

    terminal_width_fraction: float = 0.25
    terminal_width_max_length_px: float = 20.0
    tip_width_exclusion_px: float = 2.0

    angle_tangent_distance_px: float = 8.0
    angle_hub_degree: int = 3

    density_boundary_margin_px: float = 5.0
    pixel_size: float | None = None
    physical_unit: str = "mm"


DEFAULT_BIOLOGICAL_CONFIG = BiologicalMetricConfig()


def path_cumulative_distance(coordinates):
    """Cumulative Euclidean step length along an ordered pixel path."""
    coordinates = np.asarray(coordinates, dtype=float)
    if len(coordinates) < 2:
        return np.zeros(max(len(coordinates), 1), dtype=float)
    steps = np.linalg.norm(np.diff(coordinates, axis=0), axis=1)
    return np.r_[0.0, np.cumsum(steps)]


def _empty_graph(shape=None):
    return {
        "hubs": {},
        "branches": [],
        "tips": np.empty((0, 2), dtype=float),
        "degree": Counter(),
        "node_coordinates": {},
        "distance_map": np.zeros(shape or (0, 0), dtype=float),
        "image_shape": shape,
        "hub_merge_distance_px": np.nan,
    }


def build_biological_graph(result, config=DEFAULT_BIOLOGICAL_CONFIG):
    """Collapse connected nearby junctions and retain biological branches.

    Retained branches connect a hub to a tip or two distinct hubs. Direct
    tip-to-tip paths and paths internal to a consolidated hub are excluded.
    """
    skeleton = np.asarray(result["final"], dtype=bool)
    mask = np.asarray(result["mask"], dtype=bool)
    summarized = summarize_skeleton(skeleton)
    if summarized is None:
        return _empty_graph(skeleton.shape)

    skan_skeleton, table = summarized
    degree = Counter()
    node_coordinates = {}
    paths = {}
    for branch_index, row in table.iterrows():
        src = int(row["node_id_src"])
        dst = int(row["node_id_dst"])
        coordinates = skan_skeleton.path_coordinates(int(branch_index)).astype(int)
        degree[src] += 1
        degree[dst] += 1
        node_coordinates.setdefault(src, coordinates[0].astype(float))
        node_coordinates.setdefault(dst, coordinates[-1].astype(float))
        paths[int(branch_index)] = coordinates

    junction_nodes = {node for node, value in degree.items() if value >= 3}
    endpoint_nodes = [node for node, value in degree.items() if value == 1]
    distance_map = distance_transform_edt(mask)
    endpoint_widths = [
        2.0 * distance_map[tuple(node_coordinates[node].astype(int))]
        for node in endpoint_nodes
    ]
    automatic_threshold = (
        float(np.median(endpoint_widths)) * config.hub_merge_width_factor
        if endpoint_widths else 1.0
    )
    merge_threshold = float(
        config.hub_merge_distance_px
        if config.hub_merge_distance_px is not None else automatic_threshold
    )

    parent = {node: node for node in junction_nodes}

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(first, second):
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    # Connectivity is required, so spatially close parallel branches do not merge.
    for _, row in table.iterrows():
        src = int(row["node_id_src"])
        dst = int(row["node_id_dst"])
        if (
            src in junction_nodes
            and dst in junction_nodes
            and float(row["branch_distance"]) <= merge_threshold
        ):
            union(src, dst)

    groups = {}
    for node in junction_nodes:
        groups.setdefault(find(node), []).append(node)
    hub_for_node = {
        node: root for root, nodes in groups.items() for node in nodes
    }
    hubs = {
        root: np.mean([node_coordinates[node] for node in nodes], axis=0)
        for root, nodes in groups.items()
    }

    branches = []
    for branch_index, row in table.iterrows():
        src = int(row["node_id_src"])
        dst = int(row["node_id_dst"])
        src_kind = "hub" if src in junction_nodes else "tip" if degree[src] == 1 else "other"
        dst_kind = "hub" if dst in junction_nodes else "tip" if degree[dst] == 1 else "other"
        src_id = hub_for_node.get(src, src)
        dst_id = hub_for_node.get(dst, dst)

        if src_kind == dst_kind == "hub" and src_id == dst_id:
            continue
        if {src_kind, dst_kind} not in ({"hub", "tip"}, {"hub"}):
            continue
        length = float(row["branch_distance"])
        if length < config.min_branch_length_px:
            continue
        branches.append({
            "index": int(branch_index),
            "src": src,
            "dst": dst,
            "src_kind": src_kind,
            "dst_kind": dst_kind,
            "src_id": src_id,
            "dst_id": dst_id,
            "length_px": length,
            "coordinates": paths[int(branch_index)],
        })

    height, width = skeleton.shape
    valid_tips = []
    for node in endpoint_nodes:
        y, x = node_coordinates[node]
        edge_distance = min(y, x, height - 1 - y, width - 1 - x)
        if edge_distance >= config.boundary_tip_margin_px:
            valid_tips.append(node_coordinates[node])

    return {
        "hubs": hubs,
        "branches": branches,
        "tips": np.asarray(valid_tips, dtype=float).reshape(-1, 2),
        "degree": degree,
        "node_coordinates": node_coordinates,
        "distance_map": distance_map,
        "image_shape": skeleton.shape,
        "hub_merge_distance_px": merge_threshold,
    }


def branch_length_table(graph):
    """Return one row per retained hub-tip or distinct hub-hub branch."""
    records = []
    for branch in graph["branches"]:
        records.append({
            "branch_index": branch["index"],
            "src_kind": branch["src_kind"],
            "dst_kind": branch["dst_kind"],
            "length_px": branch["length_px"],
        })
    return pd.DataFrame.from_records(
        records,
        columns=["branch_index", "src_kind", "dst_kind", "length_px"],
    )


def terminal_width_table(graph, config=DEFAULT_BIOLOGICAL_CONFIG):
    """Measure median width near the tip of each retained hub-tip branch."""
    records = []
    distance_map = graph["distance_map"]
    for branch in graph["branches"]:
        if {branch["src_kind"], branch["dst_kind"]} != {"hub", "tip"}:
            continue
        coordinates = branch["coordinates"]
        ordered = coordinates if branch["src_kind"] == "tip" else coordinates[::-1]
        cumulative = path_cumulative_distance(ordered)
        terminal_limit = min(
            branch["length_px"] * config.terminal_width_fraction,
            config.terminal_width_max_length_px,
        )
        keep = (
            (cumulative >= config.tip_width_exclusion_px)
            & (cumulative <= terminal_limit)
        )
        sampled = ordered[keep]
        if not len(sampled):
            continue
        widths = 2.0 * distance_map[sampled[:, 0], sampled[:, 1]]
        records.append({
            "branch_index": branch["index"],
            "median_width_px": float(np.median(widths)),
            "mean_width_px": float(np.mean(widths)),
            "sample_count": int(len(widths)),
            "sample_coordinates": sampled,
        })
    return pd.DataFrame.from_records(
        records,
        columns=[
            "branch_index", "median_width_px", "mean_width_px",
            "sample_count", "sample_coordinates",
        ],
    )


def branch_angle_table(graph, config=DEFAULT_BIOLOGICAL_CONFIG):
    """Return unordered pairwise angles at hubs of the selected degree."""
    hub_vectors = {hub: [] for hub in graph["hubs"]}
    for branch in graph["branches"]:
        for side in ("src", "dst"):
            if branch[f"{side}_kind"] != "hub":
                continue
            hub = branch[f"{side}_id"]
            ordered = (
                branch["coordinates"]
                if side == "src" else branch["coordinates"][::-1]
            )
            cumulative = path_cumulative_distance(ordered)
            index = min(
                int(np.searchsorted(cumulative, config.angle_tangent_distance_px)),
                len(ordered) - 1,
            )
            vector = ordered[index].astype(float) - ordered[0].astype(float)
            norm = np.linalg.norm(vector)
            if norm > 0:
                hub_vectors[hub].append(vector / norm)

    records = []
    for hub, vectors in hub_vectors.items():
        if len(vectors) != config.angle_hub_degree:
            continue
        for pair_index, (first, second) in enumerate(combinations(vectors, 2)):
            angle = np.degrees(
                np.arccos(np.clip(np.dot(first, second), -1.0, 1.0))
            )
            records.append({
                "hub_id": hub,
                "pair_index": pair_index,
                "angle_degrees": float(angle),
            })
    return pd.DataFrame.from_records(
        records, columns=["hub_id", "pair_index", "angle_degrees"]
    )


def tip_density_table(graph, config=DEFAULT_BIOLOGICAL_CONFIG):
    """Calculate local tip density as inverse nearest-tip distance."""
    tips = np.asarray(graph["tips"], dtype=float).reshape(-1, 2)
    shape = graph.get("image_shape")
    if shape is not None and len(tips):
        height, width = shape
        margin = config.density_boundary_margin_px
        edge_distance = np.minimum.reduce([
            tips[:, 0], tips[:, 1],
            height - 1 - tips[:, 0], width - 1 - tips[:, 1],
        ])
        tips = tips[edge_distance >= margin]
    if len(tips) < 2:
        return pd.DataFrame(columns=[
            "tip_index", "y", "x", "nearest_tip_index",
            "distance_px", "density_per_px", "distance", "density", "unit",
        ])

    nearest_distance, nearest_index = cKDTree(tips).query(tips, k=2)
    distance_px = nearest_distance[:, 1]
    scale = config.pixel_size if config.pixel_size is not None else 1.0
    unit = f"1/{config.physical_unit}" if config.pixel_size is not None else "1/px"
    physical_distance = distance_px * scale
    density = np.divide(
        1.0, physical_distance,
        out=np.full_like(physical_distance, np.nan),
        where=physical_distance > 0,
    )
    return pd.DataFrame({
        "tip_index": np.arange(len(tips), dtype=int),
        "y": tips[:, 0],
        "x": tips[:, 1],
        "nearest_tip_index": nearest_index[:, 1].astype(int),
        "distance_px": distance_px,
        "density_per_px": 1.0 / distance_px,
        "distance": physical_distance,
        "density": density,
        "unit": unit,
    })


def calculate_biological_metrics(result, config=DEFAULT_BIOLOGICAL_CONFIG):
    """Calculate all previously developed biological skeleton metrics."""
    graph = build_biological_graph(result, config)
    lengths = branch_length_table(graph)
    widths = terminal_width_table(graph, config)
    angles = branch_angle_table(graph, config)
    density = tip_density_table(graph, config)

    length_values = lengths["length_px"].to_numpy(float)
    width_values = widths["median_width_px"].to_numpy(float)
    angle_values = angles["angle_degrees"].to_numpy(float)
    density_values = density["density"].to_numpy(float)

    def safe_median(values):
        return float(np.median(values)) if len(values) else np.nan

    summary = {
        "tip_count": int(len(graph["tips"])),
        "hub_count": int(len(graph["hubs"])),
        "biological_branch_count": int(len(lengths)),
        "mean_branch_length_px": (
            float(np.mean(length_values)) if len(length_values) else np.nan
        ),
        "median_branch_length_px": safe_median(length_values),
        "terminal_width_count": int(len(widths)),
        "median_terminal_width_px": safe_median(width_values),
        "branch_angle_count": int(len(angles)),
        "median_branch_angle_degrees": safe_median(angle_values),
        "tip_density_count": int(len(density)),
        "median_tip_density": safe_median(density_values),
        "hub_merge_distance_px": float(graph["hub_merge_distance_px"]),
    }
    return {
        "summary": summary,
        "graph": graph,
        "branch_lengths": lengths,
        "terminal_widths": widths,
        "branch_angles": angles,
        "tip_density": density,
    }
