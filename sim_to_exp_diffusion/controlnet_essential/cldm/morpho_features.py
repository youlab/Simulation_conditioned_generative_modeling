"""Core morphology and skeleton measurements for colony images.

The pipeline is adapted from the LAMM rhizomorphic-networks implementation and
keeps plotting separate from numerical analysis.
"""

from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt
from skan import Skeleton, summarize
from skimage.filters import gaussian
from skimage.graph import MCP_Geometric
from skimage.measure import label
from skimage.morphology import (
    binary_dilation,
    closing,
    disk,
    medial_axis,
    remove_small_objects,
    skeletonize,
)


@dataclass(frozen=True)
class SkeletonConfig:
    """All parameters that control mask preparation and skeleton cleanup."""

    mask_smooth_sigma: float = 1.25
    mask_smooth_threshold: float = 0.50
    mask_min_object_px: int = 64
    mask_closing_radius_px: int = 0

    spur_factor: float = 1.0
    spur_min_px: float = 2.0
    radius_cap_px: float = 8.0
    max_prune_fraction: float = 0.02
    min_branches_for_pruning: int = 5

    blob_radius_factor: float = 1.5
    blob_min_radius_px: float = 6.0
    blob_min_core_area_px: int = 50
    blob_core_expansion_px: int = 5
    blob_interface_tolerance_px: int = 4
    blob_attachment_cluster_radius_px: int = 2
    blob_max_attachments: int = 24
    blob_min_attachments_for_hub: int = 2
    blob_path_cost_power: float = 1.5
    strand_radius_quantile: float = 0.85


DEFAULT_SKELETON_CONFIG = SkeletonConfig()


def prepare_mask(mask, config=DEFAULT_SKELETON_CONFIG):
    """Remove small objects and smooth the colony boundary once."""
    cleaned = remove_small_objects(
        np.asarray(mask, dtype=bool), min_size=int(config.mask_min_object_px)
    )
    smoothed = gaussian(
        cleaned.astype(np.float32),
        sigma=float(config.mask_smooth_sigma),
        preserve_range=True,
    )
    prepared = smoothed >= float(config.mask_smooth_threshold)
    if config.mask_closing_radius_px > 0:
        prepared = closing(
            prepared, footprint=disk(int(config.mask_closing_radius_px))
        )
    return remove_small_objects(
        prepared, min_size=int(config.mask_min_object_px)
    )


def summarize_skeleton(skeleton):
    """Return a Skan object and branch table, or ``None`` for an empty graph."""
    skeleton = np.asarray(skeleton, dtype=bool)
    if not skeleton.any():
        return None
    skan_skeleton = Skeleton(skeleton)
    table = summarize(skan_skeleton, separator="_")
    if len(table) == 0 or "branch_type" not in table:
        return None
    return skan_skeleton, table


def endpoint_pixel_map(skeleton):
    """Return skeleton pixels having exactly one 8-connected neighbour."""
    skeleton = np.asarray(skeleton, dtype=bool)
    padded = np.pad(skeleton.astype(np.uint8), 1)
    neighbours = (
        padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:]
        + padded[1:-1, :-2] + padded[1:-1, 2:]
        + padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
    )
    return skeleton & (neighbours == 1)


def junction_pixel_map(skeleton):
    """Return display-oriented skeleton pixels with at least three neighbours."""
    skeleton = np.asarray(skeleton, dtype=bool)
    padded = np.pad(skeleton.astype(np.uint8), 1)
    neighbours = (
        padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:]
        + padded[1:-1, :-2] + padded[1:-1, 2:]
        + padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
    )
    return skeleton & (neighbours >= 3)


def prune_spurs(skeleton, mask, config=DEFAULT_SKELETON_CONFIG):
    """Iteratively remove short terminal branches using local colony radius."""
    current = np.asarray(skeleton, dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    initial = summarize_skeleton(current)
    if initial is None or len(initial[1]) < config.min_branches_for_pruning:
        return current

    distance = distance_transform_edt(mask)
    previous_pixels = -1
    while int(current.sum()) != previous_pixels:
        previous_pixels = int(current.sum())
        result = summarize_skeleton(current)
        if result is None or len(result[1]) <= 1:
            break
        skan_skeleton, table = result
        total_length = float(table["branch_distance"].sum())
        removable = np.zeros(len(table), dtype=bool)

        for branch_index, row in table.iterrows():
            if int(row["branch_type"]) != 1:
                continue
            coords = skan_skeleton.path_coordinates(int(branch_index)).astype(int)
            radii = distance[coords[:, 0], coords[:, 1]]
            radii = radii[np.isfinite(radii) & (radii > 0)]
            radius = (
                min(float(np.median(radii)), config.radius_cap_px)
                if radii.size else np.nan
            )
            threshold = (
                max(config.spur_min_px, config.spur_factor * radius)
                if np.isfinite(radius) else config.spur_min_px
            )
            length = float(row["branch_distance"])
            protected = (
                config.max_prune_fraction > 0
                and total_length > 0
                and length / total_length >= config.max_prune_fraction
            )
            removable[int(branch_index)] = length < threshold and not protected

        if not removable.any():
            break
        rebuilt = np.zeros_like(current)
        for branch_index in np.flatnonzero(~removable):
            coords = skan_skeleton.path_coordinates(int(branch_index)).astype(int)
            rebuilt[coords[:, 0], coords[:, 1]] = True
        current = skeletonize(rebuilt)
    return current


def _typical_radius(distance_map, skeleton, config):
    radii = distance_map[np.asarray(skeleton, dtype=bool)]
    radii = radii[np.isfinite(radii) & (radii > 0)]
    if not radii.size:
        return 1.0
    cutoff = np.quantile(radii, config.strand_radius_quantile)
    ordinary = radii[radii <= cutoff]
    return max(1.0, float(np.median(ordinary if ordinary.size else radii)))


def _blob_attachments(external_skeleton, expanded_blob, hub, config):
    interface = binary_dilation(
        expanded_blob, footprint=disk(config.blob_interface_tolerance_px)
    ) & ~expanded_blob
    contacts = external_skeleton & interface
    if not contacts.any():
        return []
    support = binary_dilation(
        contacts, footprint=disk(config.blob_attachment_cluster_radius_px)
    )
    labels = label(support, connectivity=2)
    candidates = []
    for cluster_id in range(1, int(labels.max()) + 1):
        coordinates = np.argwhere(contacts & (labels == cluster_id))
        if not coordinates.size:
            continue
        squared_distance = (
            (coordinates[:, 0] - hub[0]) ** 2
            + (coordinates[:, 1] - hub[1]) ** 2
        )
        selected = coordinates[int(np.argmin(squared_distance))]
        candidates.append((len(coordinates), tuple(map(int, selected))))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [coord for _, coord in candidates[:config.blob_max_attachments]]


def _hub_connections(mask, distance_map, external_skeleton,
                     expanded_blob, original_core, config):
    result = np.zeros_like(mask, dtype=bool)
    core_coordinates = np.argwhere(original_core)
    if not core_coordinates.size:
        return result, 0
    core_distances = distance_map[core_coordinates[:, 0], core_coordinates[:, 1]]
    hub = tuple(map(int, core_coordinates[int(np.argmax(core_distances))]))
    attachments = _blob_attachments(external_skeleton, expanded_blob, hub, config)
    if len(attachments) < config.blob_min_attachments_for_hub:
        return result, len(attachments)

    foreground_labels = label(mask, connectivity=2)
    component_id = int(foreground_labels[hub])
    if component_id == 0:
        return result, len(attachments)
    allowed = foreground_labels == component_id
    interior_cost = 1.0 / np.power(
        np.maximum(distance_map, 0.5), config.blob_path_cost_power
    )
    mcp = MCP_Geometric(np.where(allowed, interior_cost, 1.0e6), fully_connected=True)
    mcp.find_costs([hub])
    result[hub] = True
    for attachment in attachments:
        if not allowed[attachment]:
            continue
        try:
            path = mcp.traceback(attachment)
        except Exception:
            continue
        if path:
            coordinates = np.asarray(path, dtype=int)
            result[coordinates[:, 0], coordinates[:, 1]] = True
    return result, len(attachments)


def blob_aware_skeletonize(prepared_mask, config=DEFAULT_SKELETON_CONFIG):
    """Replace medial-axis tangles inside broad cores with hub paths."""
    prepared = np.asarray(prepared_mask, dtype=bool)
    if not prepared.any():
        empty = np.zeros_like(prepared)
        return {
            "raw": empty, "reconstructed": empty, "distance": empty.astype(float),
            "blob_core": empty, "typical_radius_px": 0.0,
            "blob_threshold_px": float(config.blob_min_radius_px),
            "blob_core_count": 0, "replaced_blob_core_count": 0,
        }

    # Fix the random tie-breaking order so repeated analyses of the same mask
    # produce the same graph topology and measurements.
    raw, distance = medial_axis(prepared, return_distance=True, rng=0)
    typical_radius = _typical_radius(distance, raw, config)
    blob_threshold = max(
        config.blob_min_radius_px, config.blob_radius_factor * typical_radius
    )
    blob_core = remove_small_objects(
        distance >= blob_threshold, min_size=config.blob_min_core_area_px
    )
    reconstructed = raw.copy()
    core_labels = label(blob_core, connectivity=2)
    replaced = 0
    for core_id in range(1, int(core_labels.max()) + 1):
        original_core = core_labels == core_id
        expanded = binary_dilation(
            original_core, footprint=disk(config.blob_core_expansion_px)
        ) & prepared
        connections, attachment_count = _hub_connections(
            prepared, distance, reconstructed & ~expanded,
            expanded, original_core, config,
        )
        if (
            attachment_count >= config.blob_min_attachments_for_hub
            and connections.any()
        ):
            reconstructed[expanded] = False
            reconstructed |= connections
            replaced += 1
    return {
        "raw": raw,
        "reconstructed": skeletonize(reconstructed),
        "distance": distance,
        "blob_core": blob_core,
        "typical_radius_px": typical_radius,
        "blob_threshold_px": blob_threshold,
        "blob_core_count": int(core_labels.max()),
        "replaced_blob_core_count": replaced,
    }


def measure_skeleton(mask, skeleton):
    """Compute basic graph, length, and width summaries."""
    skeleton = np.asarray(skeleton, dtype=bool)
    result = summarize_skeleton(skeleton)
    widths = 2.0 * distance_transform_edt(mask)[skeleton]
    if result is None:
        return {
            "tips": 0, "branches": 0, "junctions": 0,
            "total_length_px": 0.0, "mean_segment_length_px": np.nan,
            "median_segment_length_px": np.nan, "mean_width_px": np.nan,
            "median_width_px": np.nan, "segment_lengths": np.array([]),
            "width_values": widths, "branch_table": None,
        }
    _, table = result
    lengths = table["branch_distance"].to_numpy(float)
    degree = Counter()
    for src, dst in zip(table["node_id_src"], table["node_id_dst"]):
        degree[int(src)] += 1
        degree[int(dst)] += 1
    return {
        "tips": int(endpoint_pixel_map(skeleton).sum()),
        "branches": int(len(table)),
        "junctions": int(sum(value >= 3 for value in degree.values())),
        "total_length_px": float(lengths.sum()),
        "mean_segment_length_px": float(lengths.mean()),
        "median_segment_length_px": float(np.median(lengths)),
        "mean_width_px": float(widths.mean()) if widths.size else np.nan,
        "median_width_px": float(np.median(widths)) if widths.size else np.nan,
        "segment_lengths": lengths,
        "width_values": widths,
        "branch_table": table,
    }


def skeletonize_colony(mask, config=DEFAULT_SKELETON_CONFIG,
                       use_blob_reconstruction=True):
    """Run the complete smoothed, blob-aware skeletonization pipeline."""
    prepared = prepare_mask(mask, config)
    blob = blob_aware_skeletonize(prepared, config)
    pre_prune = blob["reconstructed"] if use_blob_reconstruction else blob["raw"]
    final = prune_spurs(pre_prune, prepared, config)
    return {
        **blob,
        "mask": prepared,
        "pre_prune": pre_prune,
        "final": final,
        "metrics": measure_skeleton(prepared, final),
    }


# Compatibility alias for code copied from the exploratory notebook.
lamm_final_skeleton = skeletonize_colony
