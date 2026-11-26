"""Matched full-test image metrics for Fig. 4d and Supplementary Fig. 18a.

Reuses saved predictions and the existing LPIPS/SSIM/ORB preprocessing/metrics.
CMMD is computed from the same processed RGB images, with no dropped final batch.
Run: python image_metric_evaluation.py --output-dir ../../source_data/image_metrics
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import tempfile
from zoneinfo import ZoneInfo

import numpy as np

from third_party.cmmd_pytorch import distance as cmmd_distance

CLIP_MODEL = "openai/clip-vit-large-patch14-336"
CMMD_SIGMA = cmmd_distance._SIGMA
CMMD_SCALE = cmmd_distance._SCALE
CMMD_IMPLEMENTATION = "sayakpaul/cmmd-pytorch@f7673a9f46320ac958878c19f7686e203376ad9d"
COMPARISONS = {
    "exp_sim": ("experiment", "simulation"),
    "exp_pred": ("experiment", "simulation_conditioned"),
    "exp_seed": ("experiment", "seed_conditioned"),
    "sim_pred": ("simulation", "simulation_conditioned"),
}
ROLE_COLUMNS = {
    "experiment": "experiment_path",
    "simulation": "simulation_path",
    "simulation_conditioned": "simulation_conditioned_path",
    "seed_conditioned": "seed_conditioned_path",
}
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "source_data/image_metrics"


def _files_by_stem(folder):
    result = {}
    for path in sorted(Path(folder).iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}:
            continue
        if path.stem in result:
            raise ValueError(f"Duplicate image stem {path.stem!r} in {folder}")
        result[path.stem] = path.resolve()
    return result


def build_test_manifest(config=None, *, include_fixed=True, fixed_sample=1,
                        seed_sample=1, expected_counts=None, include_seed_conditioned=True):
    """Resolve all pairs explicitly; never intersect away missing test images.

    Simulation-conditioned fixed outputs use sample 1, as in the morphology
    notebook. Corrected seed-conditioned original outputs use sample 1 per stem.
    Fixed seed-conditioned outputs are per configuration; draw 1/2 is paired
    with experimental replicate 1/2 (chosen by index, never by metric score).
    """
    if config is None:
        from cldm import config
    if expected_counts is None:
        expected_counts = {"primary": 96, "fixed": 36 if include_fixed else 0}
    primary = _files_by_stem(config.EXP_FOLDER_TEST)
    fixed = _files_by_stem(config.EXP_FOLDER_TEST_FIXED) if include_fixed else {}
    for subset, inventory in (("primary", primary), ("fixed", fixed)):
        if len(inventory) != expected_counts[subset]:
            raise ValueError(f"Expected {expected_counts[subset]} {subset} experiments, found {len(inventory)}")
    if primary.keys() & fixed.keys():
        raise ValueError("Primary and fixed experimental identifiers overlap")
    simulations = _files_by_stem(config.SIM_FOLDER_TEST)
    predictions = _files_by_stem(config.MAIN_FOLDER)
    seeds = _files_by_stem(config.SEED_FOLDER_TEST)
    corrected = _files_by_stem(config.PRED_FOLDER_SEEDTOEXP_CORRECTED_96) if include_seed_conditioned else {}
    fixed_sim = _files_by_stem(config.SIM_FOLDER_TEST_FIXED_OUTPUT) if include_fixed else {}
    fixed_pred = _files_by_stem(config.DIFFUSION_FOLDER_TEST_FIXED) if include_fixed else {}
    fixed_seed_pred = _files_by_stem(config.SEED_CONDITIONED_FOLDER_TEST_FIXED) if include_fixed and include_seed_conditioned else {}
    fixed_seeds = _files_by_stem(config.SIM_FOLDER_TEST_FIXED_INPUT) if include_fixed else {}
    manifest, missing = [], []
    for subset, inventory in (("primary", primary), ("fixed", fixed)):
        for stem in sorted(inventory):
            pattern = r"(\d+)_(\d+)" if subset == "primary" else r"Fixed_(\d+)_(\d+)"
            match = re.fullmatch(pattern, stem)
            if match is None:
                raise ValueError(f"Unexpected experimental image identifier: {stem}")
            number, replicate = map(int, match.groups())
            if subset == "primary":
                config_id = str(number)
                paths = {"simulation_path": simulations.get(stem),
                         "simulation_conditioned_path": predictions.get(stem),
                         "seed_conditioned_path": corrected.get(f"{stem}_{seed_sample}"),
                         "seed_path": seeds.get(stem)}
                seed_draw = seed_sample
            else:
                config_id = f"Fixed_{number}"
                paths = {"simulation_path": fixed_sim.get(f"Output_{config_id}"),
                         "simulation_conditioned_path": fixed_pred.get(f"{stem}_{fixed_sample}"),
                         "seed_conditioned_path": fixed_seed_pred.get(f"{config_id}_{replicate}"),
                         "seed_path": fixed_seeds.get(f"Input_{config_id}")}
                seed_draw = replicate
            for role, path in paths.items():
                if role == "seed_conditioned_path" and not include_seed_conditioned:
                    continue
                if path is None:
                    missing.append(f"{stem}: {role}")
            manifest.append({"image_id": stem, "subset": subset,
                             "configuration_id": config_id, "experimental_replicate": replicate,
                             "experiment_path": str(inventory[stem]),
                             **{key: str(path) if path is not None else "" for key, path in paths.items()},
                             "simulation_fixed_sample": fixed_sample if subset == "fixed" else "paired saved output",
                             "seed_conditioned_sample": seed_draw})
    if missing:
        raise FileNotFoundError("Missing matched test images:\n" + "\n".join(missing))
    return manifest


def load_test_images(manifest, *, include_seeds=False):
    """Apply the existing image preparation once, keeping manifest order."""
    import cv2
    from PIL import Image
    import torch
    from cldm.preprocess import (preprocess_experimental_backgroundwhite,
                                 preprocess_simulation_graybackground,
                                 preprocess_seed_graybackground)
    batches = {role: [] for role, column in ROLE_COLUMNS.items()
               if all(row.get(column) for row in manifest)}
    if include_seeds:
        batches["seed_map"] = []
    for row in manifest:
        for role in batches:
            if role == "experiment":
                array = preprocess_experimental_backgroundwhite(row["experiment_path"])
            elif role == "simulation":
                array = preprocess_simulation_graybackground(row["simulation_path"])
            elif role == "seed_map":
                with Image.open(row["seed_path"]) as image:
                    is_native_fixed = row["subset"] == "fixed" and image.size == (37, 32)
                crop = {"left_crop": 3, "right_crop": 2} if is_native_fixed else {}
                array = preprocess_seed_graybackground(row["seed_path"], **crop)
            else:
                with Image.open(row[ROLE_COLUMNS[role]]) as image:
                    array = np.array(image.convert("RGB"))
            if array is None:
                raise ValueError(f"Failed to preprocess {role} for {row['image_id']}")
            if array.ndim == 2:
                array = np.repeat(array[:, :, None], 3, axis=2)
            if array.shape[:2] != (256, 256):
                array = cv2.resize(array, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            if array.shape != (256, 256, 3) or array.dtype != np.uint8:
                raise ValueError(f"Unexpected {role} image format: {array.shape}, {array.dtype}")
            batches[role].append(torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).float() / 255)
    return {role: torch.stack(batch) for role, batch in batches.items()}


def cmmd_from_embeddings(x, y):
    """Validate embeddings and call the original CMMD distance implementation."""
    import torch
    x = torch.as_tensor(x, dtype=torch.float32, device="cpu").detach()
    y = torch.as_tensor(y, dtype=torch.float32, device="cpu").detach()
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1] or not len(x) or not len(y):
        raise ValueError("CMMD requires nonempty embedding matrices of matching width")
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("CMMD embeddings must be finite")
    return float(cmmd_distance.mmd(x.numpy(), y.numpy()))


def clip_embeddings(images, *, batch_size=8):
    """Use the original CMMD image loader and CLIP model on the prepared images.

    Lossless temporary PNGs bridge the matched RGB tensors to the same directory
    API used by Slurm_scripts/cmmd.sh. Numbered filenames preserve manifest order.
    The original DataLoader includes the final partial batch.
    """
    import torch
    from PIL import Image
    from third_party.cmmd_pytorch.embedding import ClipEmbeddingModel
    from third_party.cmmd_pytorch.io_util import compute_embeddings_for_dir
    if batch_size <= 0:
        raise ValueError("CLIP batch size must be positive")
    model = ClipEmbeddingModel()
    outputs = {}
    with tempfile.TemporaryDirectory(prefix="cmmd_matched_images_") as temporary:
        for role in images:
            batch = images[role].detach().cpu()
            if batch.ndim != 4 or tuple(batch.shape[1:]) != (3, 256, 256) or not len(batch):
                raise ValueError(f"Expected nonempty prepared RGB images for {role}")
            if not torch.isfinite(batch).all() or batch.min() < 0 or batch.max() > 1:
                raise ValueError(f"Prepared {role} images must be finite and in [0, 1]")
            folder = Path(temporary) / role
            folder.mkdir()
            for index, image in enumerate(batch):
                array = (image.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
                Image.fromarray(array).save(folder / f"{index:08d}.png")
            outputs[role] = compute_embeddings_for_dir(str(folder), model, batch_size).astype(np.float32)
            if len(outputs[role]) != len(images[role]):
                raise AssertionError("CLIP embedding count differs from the image count")
            print(f"CLIP: {role}, {len(outputs[role])} images", flush=True)
    revision = getattr(model._model.config, "_commit_hash", None)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return outputs, revision


def _write_csv(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _provenance(manifest):
    paths = {row[column] for row in manifest for column in ROLE_COLUMNS.values() if row.get(column)}
    sources = [Path(__file__), Path(__file__).parent / "cldm/metrics.py",
               Path(__file__).parent / "cldm/preprocess.py"]
    sources.extend(Path(__file__).parent / "third_party/cmmd_pytorch" / name
                   for name in ("distance.py", "embedding.py", "io_util.py"))
    hashes = {path: _sha256(path) for path in sorted(paths)}
    source_hashes = {str(path): _sha256(path) for path in sources}
    signature = hashlib.sha256(json.dumps({"manifest": manifest, "inputs": hashes,
        "sources": source_hashes, "clip": CLIP_MODEL, "sigma": CMMD_SIGMA,
        "scale": CMMD_SCALE}, sort_keys=True).encode()).hexdigest()
    return signature, hashes, source_hashes


def metric_scores(evaluation, comparison, metric):
    return np.array([float(row[metric]) for row in evaluation["per_image"]
                     if row["comparison"] == comparison])


def evaluate_metrics(manifest, images, output_dir=DEFAULT_OUTPUT_DIR, *, recompute=False,
                     clip_batch_size=8):
    """Compute or load matching saved results. Cached results must match inputs/code."""
    comparisons = {name: pair for name, pair in COMPARISONS.items()
                   if all(role in images for role in pair)}
    output_dir = Path(output_dir)
    signature, hashes, source_hashes = _provenance(manifest)
    metadata_path = output_dir / "evaluation_settings.json"
    if metadata_path.exists() and not recompute:
        metadata = json.loads(metadata_path.read_text())
        if metadata["input_signature"] != signature:
            raise ValueError("Saved metrics differ from current inputs/code; set RECOMPUTE_METRICS=True or choose a new output directory")
        result = {key: list(csv.DictReader((output_dir / filename).open()))
                  for key, filename in (("per_image", "per_image_metrics.csv"),
                                        ("summary", "summary_metrics.csv"),
                                        ("cmmd", "cmmd_metrics.csv"))}
        result["metadata"] = metadata
        print(f"Loaded matched {len(manifest)}-image metrics from {output_dir}", flush=True)
        return result
    from cldm.metrics import (calculate_lpips_score_batch, calculate_ssim_batch,
                              calculate_orb_similarity_batch)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "test_manifest.csv", manifest)
    per_image, summaries = [], []
    for comparison, (reference, target) in comparisons.items():
        print(f"LPIPS/SSIM/ORB: {comparison}, {len(manifest)} pairs", flush=True)
        metrics = {"lpips": calculate_lpips_score_batch(images[reference], images[target]),
                   "ssim": calculate_ssim_batch(images[reference], images[target]),
                   "orb": calculate_orb_similarity_batch(images[reference], images[target])}
        for metric, values in metrics.items():
            if len(values) != len(manifest) or not np.isfinite(values).all():
                raise ValueError(f"Invalid or missing {metric} scores for {comparison}")
        for index, row in enumerate(manifest):
            per_image.append({"image_id": row["image_id"], "subset": row["subset"],
                              "configuration_id": row["configuration_id"], "comparison": comparison,
                              **{metric: float(values[index]) for metric, values in metrics.items()}})
        for subset in ("all", "primary", "fixed"):
            indices = [i for i, row in enumerate(manifest) if subset == "all" or row["subset"] == subset]
            if not indices:
                continue
            for metric, values in metrics.items():
                summaries.append({"comparison": comparison, "subset": subset,
                                  "metric": metric, "n_images": len(indices),
                                  "mean": float(values[indices].mean()),
                                  "std": float(values[indices].std(ddof=0))})
    embeddings, revision = clip_embeddings(images, batch_size=clip_batch_size)
    cmmd_rows = []
    for subset in ("all", "primary", "fixed"):
        indices = [i for i, row in enumerate(manifest) if subset == "all" or row["subset"] == subset]
        if not indices:
            continue
        for comparison, (reference, target) in comparisons.items():
            cmmd_rows.append({"comparison": comparison, "subset": subset,
                              "n_reference": len(indices), "n_target": len(indices),
                              "cmmd": cmmd_from_embeddings(embeddings[reference][indices], embeddings[target][indices])})
    _write_csv(output_dir / "per_image_metrics.csv", per_image)
    _write_csv(output_dir / "summary_metrics.csv", summaries)
    _write_csv(output_dir / "cmmd_metrics.csv", cmmd_rows)
    np.savez_compressed(output_dir / "clip_embeddings.npz", **embeddings)
    metadata = {"created_at": datetime.now(ZoneInfo("America/New_York")).isoformat(),
                "n_images": len(manifest), "n_configurations": len({r['configuration_id'] for r in manifest}),
                "subset_counts": {s: sum(r['subset'] == s for r in manifest) for s in ('primary', 'fixed')},
                "pairing": "Explicit filename manifest; no silent intersections or score-based sample selection",
                "sample_selection": {"simulation_conditioned_fixed": "sample index recorded per row; default 1, matching morphology",
                                     "seed_conditioned_primary": "corrected generator, sample 1 per experimental stem by default",
                                     "seed_conditioned_fixed": "corrected generator, sample index equals experimental replicate index"},
                "preprocessing": {"experiment": "existing preprocess_experimental_backgroundwhite; existing cohort-specific intensity adjustments already applied",
                                  "simulation": "existing preprocess_simulation_graybackground",
                                  "generated": "RGB, no additional crop; 256x256"},
                "metrics": {"LPIPS": "VGG, inputs [-1,1]", "SSIM": "grayscale, data_range=1", "ORB": "cross-checked Hamming matches / max keypoint count",
                            "CMMD": {"implementation": CMMD_IMPLEMENTATION,
                                     "clip_model": CLIP_MODEL, "clip_revision": revision,
                                     "sigma": CMMD_SIGMA, "scale": CMMD_SCALE, "estimator": "biased RBF MMD squared",
                                     "embedding_normalization": "L2", "batch_size": clip_batch_size, "drop_last": False,
                                     "inputs": "same processed RGB images used for LPIPS/SSIM/ORB"}},
                "summary": "image-level mean and population SD (ddof=0); no SD assigned to dataset-level CMMD",
                "simulation_weighting": "one simulation counterpart per experimental image; deterministic counterparts repeated for experimental replicates",
                "input_signature": signature, "input_sha256": hashes, "source_sha256": source_hashes}
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved full-test metrics: {output_dir}", flush=True)
    return {"per_image": per_image, "summary": summaries, "cmmd": cmmd_rows, "metadata": metadata}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--primary-only", action="store_true", help="Evaluate the original 96-image cohort only")
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--clip-batch-size", type=int, default=8)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(4)
    manifest = build_test_manifest(include_fixed=not args.primary_only)
    print(f"Matched {len(manifest)} test images", flush=True)
    images = load_test_images(manifest)
    result = evaluate_metrics(manifest, images, args.output_dir,
                              recompute=args.recompute, clip_batch_size=args.clip_batch_size)
    for row in result['summary']:
        if row['subset'] == 'all':
            print(row)
    for row in result['cmmd']:
        if row['subset'] == 'all':
            print(row)


if __name__ == "__main__":
    main()
