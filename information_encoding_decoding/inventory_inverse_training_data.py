"""Prepare the Phase 1 experimental inventory for inverse-model ablations.

This script is intentionally read-only with respect to the source datasets. It
does not create rotations or training images. When run, it inventories the
unaugmented experimental plates and seed targets, identifies configurations
with exactly two biological replicates, and writes reproducible nested
configuration selections for N=10, 50, and 100.

The default selection is balanced across observed seed counts. Set
``INVERSE_ABLATION_SELECTION_MODE`` in ``utils/config.py`` to ``random`` for an
unstratified deterministic ordering.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict

from utils.config import (
    EXP_FILES_TRAIN_100AUG,
    EXP_FOLDER_TRAIN_100AUG,
    INVERSE_ABLATION_PHASE1_OUTPUT_DIR,
    INVERSE_ABLATION_SELECTION_MODE,
    INVERSE_ABLATION_SELECTION_SEED,
    INVERSE_ABLATION_SELECTION_SIZES,
    SEED_COUNT_JSON_TRAIN_UNAUGMENTED,
    SEED_FILES_TRAIN_100AUG,
    SEED_FOLDER_TRAIN_100AUG,
)
SUPPORTED_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def natural_key(value: str):
    """Return a case-insensitive natural-sort key."""
    return [int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", value)]


def parse_plate_name(filename: str):
    """Return (configuration_id, replicate_id) from an unaugmented filename."""
    stem = os.path.splitext(filename)[0]
    match = re.fullmatch(r"(?P<configuration>.+)_(?P<replicate>\d+)", stem)
    if match is None:
        raise ValueError(
            f"Cannot parse configuration and replicate from '{filename}'. "
            "Expected a final '_<integer replicate>' suffix."
        )
    return match.group("configuration"), int(match.group("replicate"))


def zero_rotation_file_map(folder, configured_names):
    """Map each original plate filename to its existing 0-degree file."""
    zero_rotation_files = {}
    rotation_suffix = "_rot0.0"
    for filename in configured_names:
        stem, extension = os.path.splitext(filename)
        if extension.casefold() not in SUPPORTED_SUFFIXES:
            continue
        if not stem.endswith(rotation_suffix):
            continue
        if not os.path.isfile(os.path.join(folder, filename)):
            continue
        original_filename = stem[: -len(rotation_suffix)] + extension
        if original_filename in zero_rotation_files:
            raise ValueError(f"Duplicate zero-rotation plate: {original_filename}")
        zero_rotation_files[original_filename] = filename
    return zero_rotation_files


def stable_group_seed(selection_seed: int, group_name: str):
    digest = hashlib.sha256(f"{selection_seed}:{group_name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def balanced_seed_count_order(configuration_rows, selection_seed: int):
    """Round-robin shuffled seed-count strata to produce one nested ordering."""
    by_seed_count = defaultdict(list)
    for row in configuration_rows:
        by_seed_count[int(row["consensus_seed_count"])].append(row["configuration_id"])

    for seed_count, configuration_ids in by_seed_count.items():
        configuration_ids.sort(key=natural_key)
        random.Random(stable_group_seed(selection_seed, str(seed_count))).shuffle(
            configuration_ids
        )

    ordered = []
    seed_counts = sorted(by_seed_count)
    next_index = {seed_count: 0 for seed_count in seed_counts}
    while True:
        added = False
        for seed_count in seed_counts:
            index = next_index[seed_count]
            values = by_seed_count[seed_count]
            if index < len(values):
                ordered.append(values[index])
                next_index[seed_count] += 1
                added = True
        if not added:
            return ordered


def random_order(configuration_rows, selection_seed: int):
    ordered = sorted(
        (row["configuration_id"] for row in configuration_rows),
        key=natural_key,
    )
    random.Random(selection_seed).shuffle(ordered)
    return ordered


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    selection_sizes = sorted(set(INVERSE_ABLATION_SELECTION_SIZES))
    if not selection_sizes or selection_sizes[0] <= 0:
        raise ValueError("Selection sizes must be positive integers")

    for path, description in (
        (EXP_FOLDER_TRAIN_100AUG, "experimental image directory"),
        (SEED_FOLDER_TRAIN_100AUG, "seed target directory"),
        (SEED_COUNT_JSON_TRAIN_UNAUGMENTED, "seed-count JSON"),
    ):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {description}: {path}")

    experiment_file_map = zero_rotation_file_map(
        EXP_FOLDER_TRAIN_100AUG,
        EXP_FILES_TRAIN_100AUG,
    )
    target_file_map = zero_rotation_file_map(
        SEED_FOLDER_TRAIN_100AUG,
        SEED_FILES_TRAIN_100AUG,
    )
    experiment_names = set(experiment_file_map)
    target_names = set(target_file_map)
    with open(SEED_COUNT_JSON_TRAIN_UNAUGMENTED) as handle:
        seed_counts = json.load(handle)

    all_names = sorted(
        experiment_names | target_names | set(seed_counts),
        key=natural_key,
    )
    plate_rows = []
    audit_rows = []
    rows_by_configuration = defaultdict(list)

    for filename in all_names:
        try:
            configuration_id, replicate_id = parse_plate_name(filename)
        except ValueError as error:
            audit_rows.append(
                {
                    "severity": "error",
                    "scope": "filename",
                    "identifier": filename,
                    "issue": str(error),
                }
            )
            continue

        row = {
            "configuration_id": configuration_id,
            "replicate_id": replicate_id,
            "filename": filename,
            "experimental_path": os.path.join(
                EXP_FOLDER_TRAIN_100AUG,
                experiment_file_map.get(filename, filename),
            ),
            "target_path": os.path.join(
                SEED_FOLDER_TRAIN_100AUG,
                target_file_map.get(filename, filename),
            ),
            "experimental_exists": filename in experiment_names,
            "target_exists": filename in target_names,
            "seed_count_available": filename in seed_counts,
            "seed_count": seed_counts.get(filename, ""),
        }
        plate_rows.append(row)
        rows_by_configuration[configuration_id].append(row)

        for present, issue in (
            (row["experimental_exists"], "missing experimental image"),
            (row["target_exists"], "missing seed target"),
            (row["seed_count_available"], "missing seed-count metadata"),
        ):
            if not present:
                audit_rows.append(
                    {
                        "severity": "error",
                        "scope": "plate",
                        "identifier": filename,
                        "issue": issue,
                    }
                )

    configuration_rows = []
    for configuration_id in sorted(rows_by_configuration, key=natural_key):
        rows = sorted(rows_by_configuration[configuration_id], key=lambda row: row["replicate_id"])
        replicate_ids = [row["replicate_id"] for row in rows]
        available_seed_counts = [
            int(row["seed_count"])
            for row in rows
            if row["seed_count_available"]
        ]
        unique_seed_counts = sorted(set(available_seed_counts))
        all_pairs_complete = all(
            row["experimental_exists"]
            and row["target_exists"]
            and row["seed_count_available"]
            for row in rows
        )
        unique_replicates = len(set(replicate_ids)) == len(replicate_ids)
        seed_count_consistent = len(unique_seed_counts) == 1
        exactly_two_replicates = len(rows) == 2 and set(replicate_ids) == {1, 2}
        eligible = (
            exactly_two_replicates
            and unique_replicates
            and all_pairs_complete
            and seed_count_consistent
        )

        configuration_row = {
            "configuration_id": configuration_id,
            "plate_count": len(rows),
            "replicate_ids": ";".join(map(str, replicate_ids)),
            "seed_counts": ";".join(map(str, unique_seed_counts)),
            "consensus_seed_count": unique_seed_counts[0] if seed_count_consistent else "",
            "all_pairs_complete": all_pairs_complete,
            "seed_count_consistent": seed_count_consistent,
            "exactly_two_replicates": exactly_two_replicates,
            "eligible_two_replicate_curve": eligible,
        }
        configuration_rows.append(configuration_row)

        if not unique_replicates:
            audit_rows.append(
                {
                    "severity": "error",
                    "scope": "configuration",
                    "identifier": configuration_id,
                    "issue": "duplicate replicate identifier",
                }
            )
        if available_seed_counts and not seed_count_consistent:
            audit_rows.append(
                {
                    "severity": "error",
                    "scope": "configuration",
                    "identifier": configuration_id,
                    "issue": f"inconsistent seed counts: {unique_seed_counts}",
                }
            )

    eligible_rows = [
        row for row in configuration_rows if row["eligible_two_replicate_curve"]
    ]
    if INVERSE_ABLATION_SELECTION_MODE == "balanced_seed_count":
        selection_order = balanced_seed_count_order(
            eligible_rows, INVERSE_ABLATION_SELECTION_SEED
        )
    elif INVERSE_ABLATION_SELECTION_MODE == "random":
        selection_order = random_order(
            eligible_rows, INVERSE_ABLATION_SELECTION_SEED
        )
    else:
        raise ValueError(
            "INVERSE_ABLATION_SELECTION_MODE must be 'balanced_seed_count' "
            "or 'random'"
        )

    if selection_sizes[-1] > len(selection_order):
        raise ValueError(
            f"Requested N={selection_sizes[-1]}, but only {len(selection_order)} "
            "complete two-replicate configurations are eligible"
        )

    selection_rank = {
        configuration_id: rank
        for rank, configuration_id in enumerate(selection_order, start=1)
    }
    for row in configuration_rows:
        rank = selection_rank.get(row["configuration_id"])
        row["selection_rank"] = rank or ""
        for size in selection_sizes:
            row[f"selected_n{size}"] = rank is not None and rank <= size
        row["selected_nmax"] = rank is not None

    configuration_lookup = {
        row["configuration_id"]: row for row in configuration_rows
    }
    selection_rows = []
    conditions = [(f"N{size}", size) for size in selection_sizes]
    conditions.append(("NMAX", len(selection_order)))
    for condition, size in conditions:
        for within_condition_rank, configuration_id in enumerate(
            selection_order[:size], start=1
        ):
            row = configuration_lookup[configuration_id]
            selection_rows.append(
                {
                    "condition": condition,
                    "within_condition_rank": within_condition_rank,
                    "configuration_id": configuration_id,
                    "seed_count": row["consensus_seed_count"],
                    "plate_count": row["plate_count"],
                    "replicate_ids": row["replicate_ids"],
                    "selection_seed": INVERSE_ABLATION_SELECTION_SEED,
                    "selection_mode": INVERSE_ABLATION_SELECTION_MODE,
                }
            )

    os.makedirs(INVERSE_ABLATION_PHASE1_OUTPUT_DIR, exist_ok=True)
    write_csv(
        os.path.join(
            INVERSE_ABLATION_PHASE1_OUTPUT_DIR,
            "experimental_plate_inventory.csv",
        ),
        plate_rows,
        [
            "configuration_id",
            "replicate_id",
            "filename",
            "experimental_path",
            "target_path",
            "experimental_exists",
            "target_exists",
            "seed_count_available",
            "seed_count",
        ],
    )
    configuration_fields = [
        "configuration_id",
        "plate_count",
        "replicate_ids",
        "seed_counts",
        "consensus_seed_count",
        "all_pairs_complete",
        "seed_count_consistent",
        "exactly_two_replicates",
        "eligible_two_replicate_curve",
        "selection_rank",
        *[f"selected_n{size}" for size in selection_sizes],
        "selected_nmax",
    ]
    write_csv(
        os.path.join(
            INVERSE_ABLATION_PHASE1_OUTPUT_DIR,
            "experimental_configuration_inventory.csv",
        ),
        configuration_rows,
        configuration_fields,
    )
    write_csv(
        os.path.join(
            INVERSE_ABLATION_PHASE1_OUTPUT_DIR,
            "experimental_nested_selections.csv",
        ),
        selection_rows,
        [
            "condition",
            "within_condition_rank",
            "configuration_id",
            "seed_count",
            "plate_count",
            "replicate_ids",
            "selection_seed",
            "selection_mode",
        ],
    )
    write_csv(
        os.path.join(INVERSE_ABLATION_PHASE1_OUTPUT_DIR, "phase1_audit_issues.csv"),
        audit_rows,
        ["severity", "scope", "identifier", "issue"],
    )

    plate_count_distribution = Counter(row["plate_count"] for row in configuration_rows)
    eligible_seed_count_distribution = Counter(
        int(row["consensus_seed_count"]) for row in eligible_rows
    )
    summary = {
        "inputs": {
            "experiment_dir": EXP_FOLDER_TRAIN_100AUG,
            "target_dir": SEED_FOLDER_TRAIN_100AUG,
            "seed_count_json": SEED_COUNT_JSON_TRAIN_UNAUGMENTED,
            "rotation_selected": 0.0,
        },
        "selection": {
            "selection_seed": INVERSE_ABLATION_SELECTION_SEED,
            "selection_mode": INVERSE_ABLATION_SELECTION_MODE,
            "requested_sizes": selection_sizes,
            "nested": True,
            "eligibility": (
                "exactly replicate IDs 1 and 2, complete experimental-target pair, "
                "available and consistent seed count"
            ),
        },
        "counts": {
            "experimental_files": len(experiment_names),
            "target_files": len(target_names),
            "seed_count_entries": len(seed_counts),
            "inventory_plates": len(plate_rows),
            "configurations": len(configuration_rows),
            "eligible_two_replicate_configurations": len(eligible_rows),
            "audit_issues": len(audit_rows),
        },
        "plate_count_distribution": {
            str(key): value for key, value in sorted(plate_count_distribution.items())
        },
        "eligible_seed_count_distribution": {
            str(key): value
            for key, value in sorted(eligible_seed_count_distribution.items())
        },
        "outputs": {
            "plate_inventory": "experimental_plate_inventory.csv",
            "configuration_inventory": "experimental_configuration_inventory.csv",
            "nested_selections": "experimental_nested_selections.csv",
            "audit_issues": "phase1_audit_issues.csv",
        },
    }
    summary_path = os.path.join(
        INVERSE_ABLATION_PHASE1_OUTPUT_DIR, "phase1_summary.json"
    )
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
