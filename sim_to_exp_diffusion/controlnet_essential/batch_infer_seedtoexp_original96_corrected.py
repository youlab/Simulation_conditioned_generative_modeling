"""Regenerate two corrected seed-conditioned predictions per original test stem."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re

import cv2

from cldm.config import (
    CKPT_PATH_SEEDTOEXP_CORRECTED,
    EXP_FOLDER_TEST,
    PRED_FOLDER_SEEDTOEXP_CORRECTED_96,
    SEED_FOLDER_TEST,
)
from cldm.preprocess import preprocess_seed_graybackground


ORIGINAL_STEM_PATTERN = re.compile(r"^\d+_\d+$")
DEFAULT_BASE_SEED = 729397049


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path(SEED_FOLDER_TEST))
    parser.add_argument("--experiment-dir", type=Path, default=Path(EXP_FOLDER_TEST))
    parser.add_argument(
        "--output-dir", type=Path, default=Path(PRED_FOLDER_SEEDTOEXP_CORRECTED_96)
    )
    parser.add_argument("--checkpoint", type=Path, default=Path(CKPT_PATH_SEEDTOEXP_CORRECTED))
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def numeric_stem_key(path):
    return tuple(int(part) for part in path.stem.split("_"))


def sampling_seed(base_seed, stem):
    digest = hashlib.sha256(stem.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:8], "big")
    return (base_seed + offset) % (2**31)


def main():
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    input_paths = sorted(
        [
            path for path in args.input_dir.glob("*.png")
            if ORIGINAL_STEM_PATTERN.fullmatch(path.stem)
        ],
        key=numeric_stem_key,
    )
    experiment_stems = {
        path.stem for path in args.experiment_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    }
    input_stems = {path.stem for path in input_paths}
    if len(input_paths) != 96:
        raise RuntimeError(f"Expected 96 original seed inputs, found {len(input_paths)}")
    if input_stems != experiment_stems:
        raise RuntimeError(
            "Original seed and experimental stems differ: "
            f"missing seeds={sorted(experiment_stems - input_stems)}, "
            f"missing experiments={sorted(input_stems - experiment_stems)}"
        )

    planned = []
    for input_path in input_paths:
        output_paths = [
            args.output_dir / f"{input_path.stem}_{sample}.png"
            for sample in (1, 2)
        ]
        if args.overwrite or not all(path.is_file() for path in output_paths):
            planned.append((input_path, output_paths))
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Matched original stems: {len(input_paths)}")
    print(f"Stems requiring generation: {len(planned)}")
    print(f"Expected final PNG count: {len(input_paths) * 2}")
    if args.dry_run or not planned:
        return

    os.environ["SEEDTOEXP_CKPT_PATH"] = str(args.checkpoint)
    from pipeline_seedtoexp import process

    args.output_dir.mkdir(parents=True, exist_ok=True)
    inference_args = {
        "prompt": "",
        "a_prompt": "",
        "n_prompt": "longbody, lowres, bad anatomy, cropped, worst quality, low quality",
        "num_samples": 2,
        "image_resolution": 256,
        "ddim_steps": 50,
        "guess_mode": False,
        "strength": 1.0,
        "scale": 15.1,
        "eta": 0.0,
    }
    generated = []
    seeds_by_stem = {}
    for index, (input_path, output_paths) in enumerate(planned, start=1):
        conditioning = preprocess_seed_graybackground(
            str(input_path), top_crop=0, bottom_crop=0, left_crop=3, right_crop=2
        )
        conditioning = cv2.resize(
            conditioning, (256, 256), interpolation=cv2.INTER_LINEAR
        )
        seed = sampling_seed(args.base_seed, input_path.stem)
        predictions = process(conditioning, **inference_args, seed=seed)
        for prediction, output_path in zip(predictions, output_paths):
            if output_path.exists() and not args.overwrite:
                continue
            output_bgr = cv2.cvtColor(prediction, cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(str(output_path), output_bgr):
                raise OSError(f"Failed to write prediction: {output_path}")
            generated.append(output_path.name)
        seeds_by_stem[input_path.stem] = seed
        print(f"[{index}/{len(planned)}] Finished {input_path.stem}", flush=True)

    final_paths = sorted(args.output_dir.glob("*.png"), key=numeric_stem_key)
    expected_output_stems = {
        f"{input_stem}_{sample}" for input_stem in input_stems for sample in (1, 2)
    }
    final_stems = {path.stem for path in final_paths}
    if len(final_paths) != 192 or final_stems != expected_output_stems:
        raise RuntimeError(
            f"Expected exactly 192 corrected predictions, found {len(final_paths)}"
        )

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint.resolve()),
        "input_directory": str(args.input_dir.resolve()),
        "experiment_directory": str(args.experiment_dir.resolve()),
        "output_directory": str(args.output_dir.resolve()),
        "input_count": len(input_paths),
        "samples_per_input": 2,
        "output_count": len(final_paths),
        "generated_or_replaced_this_run": generated,
        "preprocessing": {
            "crop": {"top": 0, "bottom": 0, "left": 3, "right": 2},
            "resize": [256, 256],
            "interpolation": "cv2.INTER_LINEAR",
        },
        "sampling": inference_args,
        "base_seed": args.base_seed,
        "per_stem_sampling_seeds": seeds_by_stem,
    }
    with open(args.output_dir / "generation_manifest.json", "w") as file:
        json.dump(manifest, file, indent=2)
    print("Validated 192 corrected predictions and wrote generation_manifest.json")


if __name__ == "__main__":
    main()
