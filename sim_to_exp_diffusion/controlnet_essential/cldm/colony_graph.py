"""Shared colony-graph construction used by retained morphology metrics.

The graph builder is preserved unchanged from the former radial-intensity
module. The separate radial color-profile analysis is archived under Misc.
"""
from __future__ import annotations
from collections import Counter
import numpy as np
from scipy.ndimage import distance_transform_edt
from cldm.morpho_features import summarize_skeleton


def build_consolidated_hub_graph(
    result,
    hub_merge_distance_px=None,
    hub_merge_width_factor=1.0,
    boundary_tip_margin_px=5,
    min_branch_length_px=3.0,
):
    """Build hub-tip branches and choose each hub's deepest junction as center.

    Junctions connected by a short junction-to-junction path are consolidated,
    matching the exploratory branching notebook.  Within each consolidated hub,
    the member junction having the largest mask distance-transform value is the
    "innermost" junction and is used as the center.
    """
    skeleton = np.asarray(result["final"], dtype=bool)
    mask = np.asarray(result["mask"], dtype=bool)
    summarized = summarize_skeleton(skeleton)
    if summarized is None:
        return {
            "hubs": {}, "incident_branches": [], "hub_tip_branches": [],
            "tips": np.empty((0, 2), int),
            "hub_merge_distance_px": np.nan, "distance_map": distance_transform_edt(mask),
            "node_coordinates": {},
        }

    skan_skeleton, table = summarized
    degree = Counter()
    node_coordinates = {}
    paths = {}
    for branch_index, row in table.iterrows():
        src, dst = int(row["node_id_src"]), int(row["node_id_dst"])
        coordinates = skan_skeleton.path_coordinates(int(branch_index)).astype(int)
        degree[src] += 1
        degree[dst] += 1
        node_coordinates.setdefault(src, coordinates[0].astype(int))
        node_coordinates.setdefault(dst, coordinates[-1].astype(int))
        paths[int(branch_index)] = coordinates

    junction_nodes = {node for node, value in degree.items() if value >= 3}
    endpoint_nodes = [node for node, value in degree.items() if value == 1]
    distance_map = distance_transform_edt(mask)
    endpoint_widths = [
        2.0 * distance_map[tuple(node_coordinates[node])] for node in endpoint_nodes
    ]
    automatic_threshold = (
        np.median(endpoint_widths) * float(hub_merge_width_factor)
        if endpoint_widths else 1.0
    )
    merge_threshold = float(
        hub_merge_distance_px if hub_merge_distance_px is not None else automatic_threshold
    )

    parent = {node: node for node in junction_nodes}

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(first, second):
        root_first, root_second = find(first), find(second)
        if root_first != root_second:
            parent[root_second] = root_first

    for _, row in table.iterrows():
        src, dst = int(row["node_id_src"]), int(row["node_id_dst"])
        if (
            src in junction_nodes and dst in junction_nodes
            and float(row["branch_distance"]) <= merge_threshold
        ):
            union(src, dst)

    groups = {}
    for node in junction_nodes:
        groups.setdefault(find(node), []).append(node)
    hub_for_node = {node: root for root, nodes in groups.items() for node in nodes}

    hubs = {}
    for root, members in groups.items():
        radii = np.array([
            distance_map[tuple(node_coordinates[node])] for node in members
        ])
        center_node = members[int(np.argmax(radii))]
        hubs[root] = {
            "members": tuple(members),
            "center_node": center_node,
            "center_yx": node_coordinates[center_node].astype(int),
            "center_radius_px": float(radii.max()),
            "mean_yx": np.mean([node_coordinates[node] for node in members], axis=0),
        }

    height, width = skeleton.shape
    valid_endpoint_nodes = {
        node for node in endpoint_nodes
        if min(
            node_coordinates[node][0], node_coordinates[node][1],
            height - 1 - node_coordinates[node][0],
            width - 1 - node_coordinates[node][1],
        ) >= boundary_tip_margin_px
    }

    # Retain every external branch incident on a consolidated hub.  The angle
    # analysis uses this complete list (hub-to-tip and hub-to-hub), while the
    # radial-intensity analysis below continues to use only valid hub-to-tip
    # branches.  Paths whose two junction endpoints collapsed into the same
    # hub are internal bookkeeping edges and are intentionally omitted.
    incident_branches = []
    for branch_index, row in table.iterrows():
        src, dst = int(row["node_id_src"]), int(row["node_id_dst"])
        branch_length = float(row["branch_distance"])
        if branch_length < min_branch_length_px:
            continue
        src_hub = hub_for_node.get(src)
        dst_hub = hub_for_node.get(dst)
        if src_hub is not None and dst_hub is not None and src_hub == dst_hub:
            continue
        if src_hub is not None and (dst_hub is not None or dst in endpoint_nodes):
            incident_branches.append({
                "branch_index": int(branch_index),
                "src_hub_id": int(src_hub),
                "dst_hub_id": int(dst_hub) if dst_hub is not None else None,
                "src_node": src,
                "dst_node": dst,
                "coordinates": paths[int(branch_index)].astype(int),
                "branch_length_px": branch_length,
            })
        elif dst_hub is not None and src in endpoint_nodes:
            incident_branches.append({
                "branch_index": int(branch_index),
                "src_hub_id": None,
                "dst_hub_id": int(dst_hub),
                "src_node": src,
                "dst_node": dst,
                "coordinates": paths[int(branch_index)].astype(int),
                "branch_length_px": branch_length,
            })

    hub_tip_branches = []
    for branch in incident_branches:
        src, dst = branch["src_node"], branch["dst_node"]
        if branch["src_hub_id"] is not None and dst in valid_endpoint_nodes:
            hub_node, tip_node = src, dst
            hub_id = branch["src_hub_id"]
            ordered = branch["coordinates"]
        elif branch["dst_hub_id"] is not None and src in valid_endpoint_nodes:
            hub_node, tip_node = dst, src
            hub_id = branch["dst_hub_id"]
            ordered = branch["coordinates"][::-1]
        else:
            continue
        hub_tip_branches.append({
            "branch_index": branch["branch_index"],
            "hub_id": int(hub_id),
            "hub_node": int(hub_node),
            "tip_node": int(tip_node),
            "hub_center_yx": hubs[hub_id]["center_yx"],
            "tip_yx": node_coordinates[tip_node].astype(int),
            "coordinates": ordered.astype(int),
            "branch_length_px": branch["branch_length_px"],
        })

    tips = np.asarray(
        [node_coordinates[node] for node in sorted(valid_endpoint_nodes)], dtype=int
    ).reshape(-1, 2)
    return {
        "hubs": hubs,
        "incident_branches": incident_branches,
        "hub_tip_branches": hub_tip_branches,
        "tips": tips,
        "hub_merge_distance_px": merge_threshold,
        "distance_map": distance_map,
        "node_coordinates": node_coordinates,
    }
