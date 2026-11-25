"""Evaluate inverse-ablation checkpoints on the established experimental test set.

This produces the two reviewer-facing comparisons:
1. Experimental and synthetic unique-configuration learning curves.
2. Experimental/synthetic training-mixture curve.

Every checkpoint is evaluated on the same 96 random-condition and 36
fixed-condition experimental patterns.  The headline metric is per-image
Hungarian point-matching F1 with a four-pixel tolerance, matching the earlier
experimental-test evaluation.
"""
import csv
import gc
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from expcolortoseed_dataset_ablation import MyDataset
from models.UNet_Resblocks import UNet_Resblocks
from utils import config
from utils.metrics import point_matching


PREDICTION_THRESHOLD = 0.6
MATCHING_TOLERANCE = 4
BATCH_SIZE = 16
NUM_WORKERS = 4
OUTPUT_DIR = Path("inverse_ablation_evaluation_real_test")

PHASE4_ROOT = Path(config.INVERSE_ABLATION_PHASE4_LOG_ROOT)
PHASE5_ROOT = Path(config.INVERSE_ABLATION_PHASE5_LOG_ROOT)

UNIQUE_CONFIGURATION_RUNS = [
    *(dict(comparison="unique_configuration", family="experimental", condition=f"N{n}",
           x=n, label=f"Experimental N{n}", root=PHASE4_ROOT / f"experimental_N{n}_seed42")
      for n in (10, 50, 100)),
    dict(comparison="unique_configuration", family="experimental", condition="NMAX",
         x=195, label="Experimental NMAX (195)", root=PHASE4_ROOT / "experimental_NMAX_seed42"),
    *(dict(comparison="unique_configuration", family="synthetic", condition=f"N{n}",
           x=n, label=f"Synthetic N{n}", root=PHASE4_ROOT / f"synthetic_N{n}_seed42")
      for n in (10, 50, 100, 500, 1000, 5000, 10000, 30000)),
]

MIXTURE_RUNS = [
    dict(comparison="mixture", family="mixture", condition="E000_S100", x=0,
         label="0% experimental", root=PHASE4_ROOT / "synthetic_N30000_seed42"),
    *(dict(comparison="mixture", family="mixture", condition=f"E{p:03d}_S{100-p:03d}", x=p,
           label=f"{p}% experimental", root=PHASE5_ROOT / f"E{p:03d}_S{100-p:03d}_seed42")
      for p in (25, 50, 75)),
    dict(comparison="mixture", family="mixture", condition="E100_S000", x=100,
         label="100% experimental", root=PHASE4_ROOT / "experimental_NMAX_seed42"),
]


def find_last_checkpoint(run):
    """Find the most recently updated last checkpoint, including retry folders."""
    roots = [run["root"]]
    if run["comparison"] == "mixture" and run["x"] in (25, 50, 75):
        roots.extend(sorted(PHASE5_ROOT.glob(f"{run['condition']}_seed42_retry*")))
    checkpoints = [root / "checkpoints" / "last.ckpt" for root in roots]
    checkpoints = [path for path in checkpoints if path.is_file()]
    if not checkpoints:
        searched = ", ".join(str(root) for root in roots)
        raise FileNotFoundError(f"No last checkpoint found under: {searched}")
    return max(checkpoints, key=lambda path: (path.stat().st_mtime_ns, str(path)))


def make_test_datasets():
    arguments = dict(
        use_vae=True,
        preprocess_input="backgroundwhite",
        ablation_type="none",
        gaussian_sigma=None,
    )
    return [
        ("random", MyDataset(
            source_folder=config.EXP_FOLDER_TEST,
            target_folder=config.SEED_FOLDER_EXPTOSIM_TEST_32X32,
            **arguments,
        )),
        ("fixed", MyDataset(
            source_folder=config.EXP_FOLDER_EXPTOSIM_TEST_FIXED,
            target_folder=config.SEED_FOLDER_EXPTOSIM_TEST_FIXED_32X32,
            **arguments,
        )),
    ]


def audit_test_set(datasets):
    inventory = []
    seen = set()
    for subset, dataset in datasets:
        source_stems = [Path(path).stem for path in dataset.source_files]
        target_stems = [Path(path).stem for path in dataset.target_files]
        if source_stems != target_stems:
            raise ValueError(f"Source/target stem mismatch in the {subset} test subset")
        duplicate_keys = {(subset, stem) for stem in source_stems} & seen
        if duplicate_keys:
            raise ValueError(f"Duplicate test keys: {sorted(duplicate_keys)[:5]}")
        seen.update((subset, stem) for stem in source_stems)
        inventory.append({"subset": subset, "n_images": len(dataset)})
    return inventory


def evaluate_run(run, checkpoint, datasets, device):
    print(f"Evaluating {run['label']}: {checkpoint}", flush=True)
    model = UNet_Resblocks.load_from_checkpoint(
        checkpoint, map_location=device,
    ).eval().to(device)
    rows = []
    for subset, dataset in datasets:
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=device.type == "cuda",
        )
        with torch.inference_mode():
            for batch in loader:
                source = batch["source"].to(device)
                prediction = torch.sigmoid(model(model.encode_source(source)))
                prediction = ((prediction > PREDICTION_THRESHOLD).to(torch.uint8) * 255).cpu()
                target = ((batch["target"] * 255).round().to(torch.uint8)).cpu()
                for index, stem in enumerate(batch["stem"]):
                    precision, recall, f1 = point_matching(
                        prediction[index, 0],
                        target[index, 0],
                        tolerance_radius=MATCHING_TOLERANCE,
                        algorithm="hungarian",
                    )
                    rows.append({
                        "comparison": run["comparison"],
                        "family": run["family"],
                        "condition": run["condition"],
                        "x": run["x"],
                        "label": run["label"],
                        "subset": subset,
                        "stem": stem,
                        "tolerance_precision": precision,
                        "tolerance_recall": recall,
                        "tolerance_f1": f1,
                        "checkpoint": str(checkpoint.resolve()),
                    })
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def summarize(run, checkpoint, rows, subset):
    selected = rows if subset == "all" else [row for row in rows if row["subset"] == subset]
    result = {
        "comparison": run["comparison"],
        "family": run["family"],
        "condition": run["condition"],
        "x": run["x"],
        "label": run["label"],
        "subset": subset,
        "n_images": len(selected),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_selection": "last.ckpt",
    }
    for metric in ("tolerance_precision", "tolerance_recall", "tolerance_f1"):
        values = np.asarray([row[metric] for row in selected], dtype=float)
        result[f"{metric}_mean"] = float(values.mean())
        result[f"{metric}_std"] = float(values.std())
        result[f"{metric}_sem"] = float(values.std(ddof=1) / np.sqrt(len(values)))
    return result


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def snapshot_checkpoints(selected):
    """Hard-link selected checkpoints so ongoing training cannot replace them."""
    snapshot_dir = OUTPUT_DIR / "checkpoint_snapshot"
    snapshot_dir.mkdir()
    snapshots = {}
    for index, checkpoint in enumerate(dict.fromkeys(path for _, path in selected)):
        destination = snapshot_dir / f"{index:02d}_{checkpoint.parent.parent.name}_{checkpoint.name}"
        os.link(checkpoint, destination)
        snapshots[checkpoint] = destination
    return [(run, snapshots[checkpoint]) for run, checkpoint in selected]


def plot_unique_configuration_curve(summaries):
    figure, axis = plt.subplots(figsize=(6.5, 4.5))
    styles = {"experimental": ("o", "#0072B2"), "synthetic": ("s", "#D55E00")}
    for family in ("experimental", "synthetic"):
        rows = sorted(
            (row for row in summaries if row["comparison"] == "unique_configuration"
             and row["family"] == family and row["subset"] == "all"),
            key=lambda row: row["x"],
        )
        marker, color = styles[family]
        axis.errorbar(
            [row["x"] for row in rows],
            [row["tolerance_f1_mean"] for row in rows],
            yerr=[row["tolerance_f1_std"] for row in rows],
            marker=marker, color=color, linewidth=1.8, capsize=3,
            label=family.capitalize(),
        )
    axis.set_xscale("log")
    axis.set_xlabel("Unique training configurations")
    axis.set_ylabel("Hungarian tolerance F1")
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "unique_configuration_tolerance_f1.png", dpi=300)
    figure.savefig(OUTPUT_DIR / "unique_configuration_tolerance_f1.pdf")
    plt.close(figure)


def plot_mixture_curve(summaries):
    rows = sorted(
        (row for row in summaries if row["comparison"] == "mixture" and row["subset"] == "all"),
        key=lambda row: row["x"],
    )
    figure, axis = plt.subplots(figsize=(6.5, 4.5))
    axis.errorbar(
        [row["x"] for row in rows],
        [row["tolerance_f1_mean"] for row in rows],
        yerr=[row["tolerance_f1_std"] for row in rows],
        marker="o", color="#009E73", linewidth=1.8, capsize=3,
    )
    axis.set_xticks([0, 25, 50, 75, 100])
    axis.set_xlabel("Experimental share of 90,000 training + validation pairs (%)")
    axis.set_ylabel("Hungarian tolerance F1")
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "experimental_synthetic_mixture_tolerance_f1.png", dpi=300)
    figure.savefig(OUTPUT_DIR / "experimental_synthetic_mixture_tolerance_f1.pdf")
    plt.close(figure)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = make_test_datasets()
    test_inventory = audit_test_set(datasets)
    runs = UNIQUE_CONFIGURATION_RUNS + MIXTURE_RUNS
    selected = [(run, find_last_checkpoint(run)) for run in runs]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    checkpoints = snapshot_checkpoints(selected)
    settings = {
        "device": str(device),
        "prediction_threshold": PREDICTION_THRESHOLD,
        "matching_algorithm": "hungarian",
        "matching_tolerance_pixels": MATCHING_TOLERANCE,
        "checkpoint_selection": "last.ckpt",
        "test_set": test_inventory,
        "number_test_images": sum(item["n_images"] for item in test_inventory),
        "uncertainty_in_plots": "95% normal CI of the per-image mean (1.96 * SEM)",
        "runs": [{
            "comparison": run["comparison"],
            "family": run["family"],
            "condition": run["condition"],
            "x": run["x"],
            "source_checkpoint": str(source.resolve()),
            "snapshot_checkpoint": str(snapshot.resolve()),
        } for (run, source), (_, snapshot) in zip(selected, checkpoints)],
    }
    with (OUTPUT_DIR / "evaluation_settings.json").open("w") as handle:
        json.dump(settings, handle, indent=2)

    all_rows = []
    summaries = []
    for run, checkpoint in checkpoints:
        rows = evaluate_run(run, checkpoint, datasets, device)
        all_rows.extend(rows)
        for subset in ("all", "random", "fixed"):
            summaries.append(summarize(run, checkpoint, rows, subset))

    write_csv(OUTPUT_DIR / "per_image_metrics.csv", all_rows)
    write_csv(OUTPUT_DIR / "summary_metrics.csv", summaries)
    plot_unique_configuration_curve(summaries)
    plot_mixture_curve(summaries)
    print(f"Saved evaluation outputs to {OUTPUT_DIR.resolve()}", flush=True)


if __name__ == "__main__":
    main()
