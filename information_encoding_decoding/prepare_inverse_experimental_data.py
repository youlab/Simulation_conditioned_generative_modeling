"""Build the Phase 2 experimental unique-configuration learning curve.

The four nested Phase 1 conditions are each expanded to exactly 90,000 paired
experimental/seed images using deterministic, evenly spaced rotations. Source
datasets are read only. Experiments reuse the already border-processed
zero-angle images and the existing background-white crop. Seeds are rebuilt
from the original 37 x 32 images. Settings and paths are defined in
``utils/config.py``.
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.config import (
    EXP_FILES_TRAIN_100AUG,
    EXP_FOLDER_TRAIN_100AUG,
    INVERSE_ABLATION_PHASE1_OUTPUT_DIR,
    INVERSE_ABLATION_PHASE2_CONDITIONS,
    INVERSE_ABLATION_PHASE2_OUTPUT_ROOT,
    INVERSE_ABLATION_PHASE2_OVERWRITE_EXISTING,
    INVERSE_ABLATION_PHASE2_SEED_LEFT_CROP,
    INVERSE_ABLATION_PHASE2_SEED_RIGHT_CROP,
    INVERSE_ABLATION_PHASE2_TOTAL_SAMPLES,
    SEED_FILES_TRAIN_NONAUG,
    SEED_FOLDER_TRAIN_NONAUG,
)
from utils.preprocess import preprocess_experimental_backgroundwhite


def format_angle(angle):
    """Format an angle compactly while retaining `.0` for whole degrees."""
    value = f"{float(angle):.8f}".rstrip("0").rstrip(".")
    return value if "." in value else value + ".0"


def output_filename(original_filename, angle):
    """Apply the existing `_rot<angle>` naming convention."""
    stem, extension = os.path.splitext(original_filename)
    if stem.endswith("_rot0.0"):
        stem = stem[: -len("_rot0.0")]
    return f"{stem}_rot{format_angle(angle)}{extension}"


def load_experimental(path):
    """Crop one already border-processed zero-angle experimental image."""
    image_rgb = preprocess_experimental_backgroundwhite(path)
    if image_rgb is None:
        raise FileNotFoundError(f"Could not read experimental image: {path}")
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


def rotate_experimental(image, angle):
    """Rotate a cropped 256 x 256 experiment using the established direction."""
    center = (image.shape[1] // 2, image.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(
        center,
        -float(angle),
        1.0,
    )
    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return rotated


def crop_and_resize_seed(image):
    """Crop the original 37 x 32 seed by 3/2, then enlarge to 256 x 256."""
    if image is None:
        raise ValueError("Seed image is empty")
    left_crop = INVERSE_ABLATION_PHASE2_SEED_LEFT_CROP
    right_crop = INVERSE_ABLATION_PHASE2_SEED_RIGHT_CROP
    if image.shape[1] <= left_crop + right_crop:
        raise ValueError(f"Seed image is too narrow for 3/2 crop: {image.shape}")
    cropped = image[:, left_crop : image.shape[1] - right_crop]
    if cropped.shape[:2] != (32, 32):
        raise ValueError(
            f"Expected a 32 x 32 seed after 3/2 crop, received {cropped.shape[:2]}"
        )
    return cv2.resize(cropped, (256, 256), interpolation=cv2.INTER_LINEAR)


def rotate_seed_image(resized, angle):
    """Mirror the optimized backward-mapping rotation used by existing scripts."""
    center_x, center_y = resized.shape[1] // 2, resized.shape[0] // 2
    radius = min(center_x, center_y)
    rotation_matrix = cv2.getRotationMatrix2D(
        (center_x, center_y),
        -float(angle),
        1.0,
    )
    rotated = cv2.warpAffine(
        resized,
        rotation_matrix,
        (resized.shape[1], resized.shape[0]),
        flags=cv2.INTER_NEAREST,
    )
    mask = np.zeros(resized.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (center_x, center_y), radius, 255, -1)
    return cv2.bitwise_and(rotated, rotated, mask=mask)


def load_phase1_tables():
    selection_csv = os.path.join(
        INVERSE_ABLATION_PHASE1_OUTPUT_DIR,
        "experimental_nested_selections.csv",
    )
    plate_csv = os.path.join(
        INVERSE_ABLATION_PHASE1_OUTPUT_DIR,
        "experimental_plate_inventory.csv",
    )
    selections = pd.read_csv(selection_csv, dtype={"configuration_id": str})
    plates = pd.read_csv(plate_csv, dtype={"configuration_id": str})
    plates = plates[plates["experimental_exists"] & plates["target_exists"]].copy()

    if not set(INVERSE_ABLATION_PHASE2_CONDITIONS).issubset(
        set(selections["condition"])
    ):
        raise ValueError("One or more Phase 2 conditions are absent from Phase 1")
    if not plates["filename"].is_unique:
        raise ValueError("Phase 1 plate filenames are not unique")

    experimental_paths = {
        os.path.join(EXP_FOLDER_TRAIN_100AUG, name)
        for name in EXP_FILES_TRAIN_100AUG
    }
    target_names = set(SEED_FILES_TRAIN_NONAUG)
    missing_targets = sorted(set(plates["filename"]) - target_names)
    if not set(plates["experimental_path"]).issubset(experimental_paths):
        raise FileNotFoundError("Phase 1 references a missing zero-angle experiment")
    if missing_targets:
        raise FileNotFoundError(
            f"Missing {len(missing_targets)} original seed targets; "
            f"first: {missing_targets[0]}"
        )
    plates["target_path"] = plates["filename"].map(
        lambda name: os.path.join(SEED_FOLDER_TRAIN_NONAUG, name)
    )
    return selections, plates


def condition_plates(selections, plates, condition):
    # The full-data baseline uses every valid experimental plate, including
    # configurations with 1, 4, or 6 replicates. The smaller controlled
    # conditions retain exactly two replicates per configuration.
    if condition == "NMAX":
        result = plates.copy().reset_index(drop=True)
        result["within_condition_rank"] = (
            pd.factorize(result["configuration_id"], sort=False)[0] + 1
        )
        return result

    selected = selections.loc[
        selections["condition"].eq(condition),
        ["configuration_id", "within_condition_rank"],
    ].drop_duplicates("configuration_id")
    result = selected.merge(
        plates,
        on="configuration_id",
        how="left",
        validate="one_to_many",
    )
    result = result.sort_values(
        ["within_condition_rank", "replicate_id"],
        kind="stable",
    ).reset_index(drop=True)
    if result["filename"].isna().any():
        raise ValueError(f"{condition} contains a configuration without plate files")
    if not result.groupby("configuration_id")["replicate_id"].nunique().eq(2).all():
        raise ValueError(f"{condition} does not contain exactly two plates per configuration")
    return result


def assign_rotation_counts(selections, plates, condition):
    """Assign floor/ceiling rotation counts while preserving an exact total."""
    result = condition_plates(selections, plates, condition)
    base_count, remainder = divmod(
        INVERSE_ABLATION_PHASE2_TOTAL_SAMPLES,
        len(result),
    )
    result["rotation_count"] = base_count
    if remainder:
        result.loc[: remainder - 1, "rotation_count"] += 1
    if int(result["rotation_count"].sum()) != INVERSE_ABLATION_PHASE2_TOTAL_SAMPLES:
        raise AssertionError(f"Rotation allocation failed for {condition}")
    return result


def build_manifest(condition, schedule, experimental_output_folder, target_output_folder):
    manifest_rows = []
    dataset_index = 0
    for row in schedule.itertuples(index=False):
        angles = np.linspace(0, 360, int(row.rotation_count), endpoint=False)
        for angle in angles:
            filename = output_filename(row.filename, angle)
            manifest_rows.append(
                {
                    "condition": condition,
                    "dataset_index": dataset_index,
                    "configuration_id": row.configuration_id,
                    "replicate_id": int(row.replicate_id),
                    "seed_count": int(row.seed_count),
                    "source_filename": row.filename,
                    "angle_deg": format_angle(angle),
                    "rotation_count_for_plate": int(row.rotation_count),
                    "experimental_output_path": os.path.join(
                        experimental_output_folder,
                        filename,
                    ),
                    "target_output_path": os.path.join(
                        target_output_folder,
                        filename,
                    ),
                }
            )
            dataset_index += 1

    manifest = pd.DataFrame(manifest_rows)
    if len(manifest) != INVERSE_ABLATION_PHASE2_TOTAL_SAMPLES:
        raise AssertionError(f"Manifest size failed for {condition}")
    if not manifest["experimental_output_path"].is_unique:
        raise ValueError(f"Duplicate experimental output path in {condition}")
    if not manifest["target_output_path"].is_unique:
        raise ValueError(f"Duplicate seed output path in {condition}")
    return manifest


def write_image(path, image):
    if os.path.exists(path) and not INVERSE_ABLATION_PHASE2_OVERWRITE_EXISTING:
        return
    if not cv2.imwrite(path, image):
        raise OSError(f"Failed to write image: {path}")


def materialize_condition(condition, schedule, experimental_output_folder, target_output_folder):
    os.makedirs(experimental_output_folder, exist_ok=True)
    os.makedirs(target_output_folder, exist_ok=True)

    for row in tqdm(
        schedule.itertuples(index=False),
        total=len(schedule),
        desc=condition,
    ):
        experimental_image = load_experimental(row.experimental_path)
        seed_image = cv2.imread(row.target_path, cv2.IMREAD_COLOR)
        if seed_image is None:
            raise FileNotFoundError(f"Could not read seed image: {row.target_path}")
        resized_seed = crop_and_resize_seed(seed_image)
        angles = np.linspace(0, 360, int(row.rotation_count), endpoint=False)

        for angle in angles:
            filename = output_filename(row.filename, angle)
            write_image(
                os.path.join(experimental_output_folder, filename),
                rotate_experimental(experimental_image, angle),
            )
            write_image(
                os.path.join(target_output_folder, filename),
                rotate_seed_image(resized_seed, angle),
            )


def audit_condition(condition, manifest, experimental_output_folder, target_output_folder):
    experimental_files = sorted(os.listdir(experimental_output_folder))
    target_files = sorted(os.listdir(target_output_folder))
    complete = (
        len(manifest) == INVERSE_ABLATION_PHASE2_TOTAL_SAMPLES
        and len(experimental_files) == len(manifest)
        and len(target_files) == len(manifest)
        and experimental_files == target_files
    )
    return {
        "condition": condition,
        "manifest_rows": len(manifest),
        "experimental_files": len(experimental_files),
        "target_files": len(target_files),
        "paired_filenames_match": experimental_files == target_files,
        "complete": complete,
    }


def main():
    selections, plates = load_phase1_tables()
    os.makedirs(INVERSE_ABLATION_PHASE2_OUTPUT_ROOT, exist_ok=True)

    schedule_summary_rows = []
    manifest_summary_rows = []
    audit_rows = []

    for condition in INVERSE_ABLATION_PHASE2_CONDITIONS:
        schedule = assign_rotation_counts(selections, plates, condition)
        condition_root = os.path.join(INVERSE_ABLATION_PHASE2_OUTPUT_ROOT, condition)
        experimental_output_folder = os.path.join(condition_root, "experimental")
        target_output_folder = os.path.join(condition_root, "seed")
        manifest_path = os.path.join(condition_root, "manifest.csv")
        os.makedirs(condition_root, exist_ok=True)

        manifest = build_manifest(
            condition,
            schedule,
            experimental_output_folder,
            target_output_folder,
        )
        manifest.to_csv(manifest_path, index=False)

        counts = schedule["rotation_count"]
        schedule_summary_rows.append(
            {
                "condition": condition,
                "unique_configurations": schedule["configuration_id"].nunique(),
                "experimental_plates": len(schedule),
                "minimum_rotations_per_plate": int(counts.min()),
                "maximum_rotations_per_plate": int(counts.max()),
                "total_examples": int(counts.sum()),
            }
        )
        manifest_summary_rows.append(
            {
                "condition": condition,
                "manifest_path": manifest_path,
                "rows": len(manifest),
                "unique_configurations": manifest["configuration_id"].nunique(),
                "unique_plates": manifest[
                    ["configuration_id", "replicate_id"]
                ].drop_duplicates().shape[0],
                "experimental_output_folder": experimental_output_folder,
                "target_output_folder": target_output_folder,
            }
        )

        materialize_condition(
            condition,
            schedule,
            experimental_output_folder,
            target_output_folder,
        )
        audit_rows.append(
            audit_condition(
                condition,
                manifest,
                experimental_output_folder,
                target_output_folder,
            )
        )

    schedule_summary = pd.DataFrame(schedule_summary_rows)
    manifest_summary = pd.DataFrame(manifest_summary_rows)
    dataset_audit = pd.DataFrame(audit_rows)
    schedule_summary.to_csv(
        os.path.join(INVERSE_ABLATION_PHASE2_OUTPUT_ROOT, "phase2_rotation_schedule.csv"),
        index=False,
    )
    manifest_summary.to_csv(
        os.path.join(INVERSE_ABLATION_PHASE2_OUTPUT_ROOT, "phase2_manifest_summary.csv"),
        index=False,
    )
    dataset_audit.to_csv(
        os.path.join(INVERSE_ABLATION_PHASE2_OUTPUT_ROOT, "phase2_dataset_audit.csv"),
        index=False,
    )

    summary = {
        "output_root": INVERSE_ABLATION_PHASE2_OUTPUT_ROOT,
        "total_examples_per_condition": INVERSE_ABLATION_PHASE2_TOTAL_SAMPLES,
        "seed_crop": {
            "left": INVERSE_ABLATION_PHASE2_SEED_LEFT_CROP,
            "right": INVERSE_ABLATION_PHASE2_SEED_RIGHT_CROP,
        },
        "experimental_source": EXP_FOLDER_TRAIN_100AUG,
        "experimental_source_rotation": 0.0,
        "experimental_crop": "preprocess_experimental_backgroundwhite",
        "experimental_outputs_are_preprocessed": True,
        "seed_source": SEED_FOLDER_TRAIN_NONAUG,
        "seed_crop_before_resize": True,
        "seed_output_size": [256, 256],
        "overwrite_existing": INVERSE_ABLATION_PHASE2_OVERWRITE_EXISTING,
        "conditions": schedule_summary_rows,
        "all_conditions_complete": bool(dataset_audit["complete"].all()),
    }
    with open(
        os.path.join(INVERSE_ABLATION_PHASE2_OUTPUT_ROOT, "phase2_summary.json"),
        "w",
    ) as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
