"""Branching angles at consolidated biological hubs.

For a degree-k hub, the outgoing tangent directions are ordered around the
circle and the k adjacent angular sectors are measured.  These sectors sum to
360 degrees and have an equal-spacing reference of 360 / k degrees.  This is
well-defined for any hub degree >= 3 and avoids treating non-adjacent branches
at high-degree hubs as if they formed local branching sectors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import label

from cldm.colony_graph import build_consolidated_hub_graph


ANGLE_COLUMNS = [
    "source", "image_id", "hub_id", "hub_role", "hub_degree", "angle_index",
    "branch_index_clockwise", "branch_index_counterclockwise", "angle_deg",
    "ideal_angle_deg", "absolute_deviation_deg", "relative_deviation",
]

HUB_COLUMNS = [
    "source", "image_id", "hub_id", "hub_role", "hub_degree", "angle_count",
    "mean_angle_deg", "median_angle_deg", "min_angle_deg", "max_angle_deg",
    "angle_sd_deg", "mean_absolute_deviation_deg",
    "mean_relative_deviation", "hub_center_y", "hub_center_x",
    "hub_merge_distance_px", "tangent_distance_px",
]


def _path_distance(coordinates):
    coordinates = np.asarray(coordinates, dtype=float)
    if len(coordinates) < 2:
        return np.zeros(len(coordinates), dtype=float)
    return np.r_[
        0.0,
        np.cumsum(np.linalg.norm(np.diff(coordinates, axis=0), axis=1)),
    ]


def _incident_tangents(graph, tangent_distance_px):
    """Return one outward unit tangent and sample point per hub-side branch."""
    tangents = {hub_id: [] for hub_id in graph["hubs"]}
    for branch in graph["incident_branches"]:
        sides = []
        if branch["src_hub_id"] is not None:
            sides.append((branch["src_hub_id"], branch["coordinates"]))
        if branch["dst_hub_id"] is not None:
            sides.append((branch["dst_hub_id"], branch["coordinates"][::-1]))

        for hub_id, ordered in sides:
            distance = _path_distance(ordered)
            sample_index = min(
                int(np.searchsorted(distance, tangent_distance_px)), len(ordered) - 1
            )
            attachment = ordered[0].astype(float)
            sample = ordered[sample_index].astype(float)
            vector_yx = sample - attachment
            norm = float(np.linalg.norm(vector_yx))
            if norm == 0:
                continue
            unit_yx = vector_yx / norm
            # Convert image (y, x) coordinates to a standard mathematical polar
            # direction: +x to the right and positive angles counter-clockwise.
            polar_deg = float(
                np.degrees(np.arctan2(-unit_yx[0], unit_yx[1])) % 360.0
            )
            tangents[hub_id].append({
                "branch_index": branch["branch_index"],
                "attachment_yx": attachment,
                "sample_yx": sample,
                "unit_yx": unit_yx,
                "polar_angle_deg": polar_deg,
                "sample_distance_px": float(distance[sample_index]),
            })
    return tangents


def assign_hub_roles(graph, skeleton):
    """Label one 'center' hub per connected colony; all other hubs are 'peripheral'.

    The center hub is the hub with the deepest junction in the colony mask
    (largest distance-transform radius), i.e. the broad core around the seed.
    """
    components, _ = label(np.asarray(skeleton, dtype=bool), structure=np.ones((3, 3)))
    best = {}
    for hub_id, hub in graph["hubs"].items():
        y, x = hub["center_yx"]
        component = int(components[y, x])
        if component not in best or hub["center_radius_px"] > graph["hubs"][best[component]]["center_radius_px"]:
            best[component] = hub_id
    center_ids = set(best.values())
    return {hub_id: "center" if hub_id in center_ids else "peripheral" for hub_id in graph["hubs"]}


def measure_branching_angles(
    result,
    source=None,
    image_id=None,
    tangent_distance_px=8.0,
    minimum_hub_degree=3,
    **graph_kwargs,
):
    """Measure adjacent branch sectors for every eligible consolidated hub.

    Returns ``(angle_table, hub_table, graph)``.  The angle table contains one
    row per adjacent sector.  The hub table contains equal-spacing deviation
    summaries suitable for plate-level aggregation without treating individual
    angles as independent replicates.
    """
    if tangent_distance_px <= 0:
        raise ValueError("tangent_distance_px must be positive")
    if minimum_hub_degree < 2:
        raise ValueError("minimum_hub_degree must be at least 2")

    graph = build_consolidated_hub_graph(result, **graph_kwargs)
    tangents = _incident_tangents(graph, float(tangent_distance_px))
    roles = assign_hub_roles(graph, result["final"])
    graph["hub_roles"] = roles
    angle_records = []
    hub_records = []

    for hub_id, incident in tangents.items():
        degree = len(incident)
        if degree < minimum_hub_degree:
            continue
        ordered = sorted(incident, key=lambda item: item["polar_angle_deg"])
        directions = np.array([item["polar_angle_deg"] for item in ordered])
        gaps = np.diff(np.r_[directions, directions[0] + 360.0])
        ideal = 360.0 / degree
        deviations = np.abs(gaps - ideal)

        for angle_index, (angle, deviation) in enumerate(zip(gaps, deviations)):
            first = ordered[angle_index]
            second = ordered[(angle_index + 1) % degree]
            angle_records.append({
                "source": source,
                "image_id": image_id,
                "hub_id": int(hub_id),
                "hub_role": roles[hub_id],
                "hub_degree": degree,
                "angle_index": angle_index,
                "branch_index_clockwise": first["branch_index"],
                "branch_index_counterclockwise": second["branch_index"],
                "angle_deg": float(angle),
                "ideal_angle_deg": ideal,
                "absolute_deviation_deg": float(deviation),
                "relative_deviation": float(deviation / ideal),
            })

        center = graph["hubs"][hub_id]["center_yx"]
        hub_records.append({
            "source": source,
            "image_id": image_id,
            "hub_id": int(hub_id),
            "hub_role": roles[hub_id],
            "hub_degree": degree,
            "angle_count": degree,
            "mean_angle_deg": float(np.mean(gaps)),
            "median_angle_deg": float(np.median(gaps)),
            "min_angle_deg": float(np.min(gaps)),
            "max_angle_deg": float(np.max(gaps)),
            "angle_sd_deg": float(np.std(gaps, ddof=1)) if degree > 1 else np.nan,
            "mean_absolute_deviation_deg": float(np.mean(deviations)),
            "mean_relative_deviation": float(np.mean(deviations / ideal)),
            "hub_center_y": int(center[0]),
            "hub_center_x": int(center[1]),
            "hub_merge_distance_px": graph["hub_merge_distance_px"],
            "tangent_distance_px": float(tangent_distance_px),
        })

    graph["hub_tangents"] = tangents
    return (
        pd.DataFrame(angle_records, columns=ANGLE_COLUMNS),
        pd.DataFrame(hub_records, columns=HUB_COLUMNS),
        graph,
    )


def summarize_branching_angles_by_plate(hub_table):
    """Aggregate hub summaries to one row per plate with equal hub weight."""
    columns = [
        "source", "image_id", "qualifying_hub_count", "angle_count",
        "mean_hub_degree", "median_hub_degree", "mean_hub_median_angle_deg",
        "mean_hub_absolute_deviation_deg", "mean_hub_relative_deviation",
        "hub_merge_distance_px", "tangent_distance_px",
    ]
    if hub_table.empty:
        return pd.DataFrame(columns=columns)

    records = []
    for (source, image_id), hubs in hub_table.groupby(
        ["source", "image_id"], dropna=False
    ):
        records.append({
            "source": source,
            "image_id": image_id,
            "qualifying_hub_count": int(len(hubs)),
            "angle_count": int(hubs["angle_count"].sum()),
            "mean_hub_degree": float(hubs["hub_degree"].mean()),
            "median_hub_degree": float(hubs["hub_degree"].median()),
            "mean_hub_median_angle_deg": float(hubs["median_angle_deg"].mean()),
            "mean_hub_absolute_deviation_deg": float(
                hubs["mean_absolute_deviation_deg"].mean()
            ),
            "mean_hub_relative_deviation": float(
                hubs["mean_relative_deviation"].mean()
            ),
            "hub_merge_distance_px": float(hubs["hub_merge_distance_px"].iloc[0]),
            "tangent_distance_px": float(hubs["tangent_distance_px"].iloc[0]),
        })
    return pd.DataFrame(records, columns=columns)


def plot_branching_angle_overlay(ax, image, graph, minimum_hub_degree=3):
    """Show consolidated hub centers and the tangent samples used for angles."""
    ax.imshow(image, cmap="gray" if np.asarray(image).ndim == 2 else None)
    colors = {3: "tab:blue", 4: "tab:orange", 5: "tab:green"}
    for hub_id, incident in graph.get("hub_tangents", {}).items():
        if len(incident) < minimum_hub_degree:
            continue
        center = graph["hubs"][hub_id]["center_yx"]
        color = colors.get(len(incident), "tab:red")
        ax.scatter(center[1], center[0], s=30, color=color, edgecolor="white")
        for tangent in incident:
            start = tangent["attachment_yx"]
            end = tangent["sample_yx"]
            ax.plot([start[1], end[1]], [start[0], end[0]], color=color, lw=1.5)
            ax.scatter(end[1], end[0], s=8, color=color)
    ax.axis("off")
    return ax
