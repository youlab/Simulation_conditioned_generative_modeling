"""Generate seed-conditioned predictions for the 19 fixed test patterns.

Each Input_Fixed_N.png conditioning map produces Fixed_N_1.png through
Fixed_N_3.png. The first two samples align with the two experimental plate
replicates when both exist; the third is retained as an additional generated
replicate. Existing complete triples are skipped unless --overwrite is used.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re

import cv2

from cldm.config import (
    CKPT_PATH_SEEDTOEXP_CORRECTED,
    SEED_CONDITIONED_FOLDER_TEST_FIXED,
    SIM_FOLDER_TEST_FIXED_INPUT,
)
from cldm.preprocess import preprocess_seed_graybackground


FIXED_PATTERN = re.compile(r"^Input_Fixed_(\d+)\.png$", re.IGNORECASE)
DEFAULT_BASE_SEED = 729397049


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path(SIM_FOLDER_TEST_FIXED_INPUT))
    parser.add_argument("--output-dir", type=Path, default=Path(SEED_CONDITIONED_FOLDER_TEST_FIXED))
    parser.add_argument("--checkpoint", type=Path, default=Path(CKPT_PATH_SEEDTOEXP_CORRECTED))
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def fixed_inputs(input_dir):
    records = []
    for path in input_dir.glob("*.png"):
        match = FIXED_PATTERN.fullmatch(path.name)
        if match:
            records.append((int(match.group(1)), path))
    return sorted(records)


def main():
    args = parse_args()
    if args.num_samples < 1:
        raise ValueError("--num-samples must be positive")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    inputs = fixed_inputs(args.input_dir)
    if len(inputs) != 19:
        raise RuntimeError(
            f"Expected 19 Input_Fixed_N.png files in {args.input_dir}, found {len(inputs)}"
        )

    planned = []
    for fixed_number, input_path in inputs:
        outputs = [
            args.output_dir / f"Fixed_{fixed_number}_{sample}.png"
            for sample in range(1, args.num_samples + 1)
        ]
        if args.overwrite or not all(path.is_file() for path in outputs):
            planned.append((fixed_number, input_path, outputs))

    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Fixed configurations: {len(inputs)}")
    print(f"Configurations requiring generation: {len(planned)}")
    print(f"Expected final PNG count: {len(inputs) * args.num_samples}")
    if args.dry_run or not planned:
        return

    # pipeline_seedtoexp loads its checkpoint at import time.
    os.environ["SEEDTOEXP_CKPT_PATH"] = str(args.checkpoint)
    from pipeline_seedtoexp import process

    args.output_dir.mkdir(parents=True, exist_ok=True)
    inference_args = {
        "prompt": "",
        "a_prompt": "",
        "n_prompt": "longbody, lowres, bad anatomy, cropped, worst quality, low quality",
        "num_samples": args.num_samples,
        "image_resolution": 256,
        "ddim_steps": 50,
        "guess_mode": False,
        "strength": 1.0,
        "scale": 15.1,
        "eta": 0.0,
    }

    generated = []
    for index, (fixed_number, input_path, output_paths) in enumerate(planned, start=1):
        conditioning = preprocess_seed_graybackground(
            str(input_path), top_crop=0, bottom_crop=0, left_crop=3, right_crop=2
        )
        conditioning = cv2.resize(
            conditioning, (256, 256), interpolation=cv2.INTER_LINEAR
        )
        sampling_seed = (args.base_seed + fixed_number * 1009) % (2**31)
        predictions = process(conditioning, **inference_args, seed=sampling_seed)

        for prediction, output_path in zip(predictions, output_paths):
            if output_path.exists() and not args.overwrite:
                continue
            output_bgr = cv2.cvtColor(prediction, cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(str(output_path), output_bgr):
                raise OSError(f"Failed to write prediction: {output_path}")
            generated.append(output_path.name)
        print(f"[{index}/{len(planned)}] Finished Fixed_{fixed_number}", flush=True)

    final_outputs = sorted(args.output_dir.glob("Fixed_*_*.png"))
    expected_count = len(inputs) * args.num_samples
    if len(final_outputs) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} fixed predictions, found {len(final_outputs)}"
        )

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint.resolve()),
        "input_directory": str(args.input_dir.resolve()),
        "output_directory": str(args.output_dir.resolve()),
        "input_count": len(inputs),
        "samples_per_input": args.num_samples,
        "output_count": len(final_outputs),
        "generated_or_replaced_this_run": generated,
        "preprocessing": {
            "crop": {"top": 0, "bottom": 0, "left": 3, "right": 2},
            "resize": [256, 256],
            "interpolation": "cv2.INTER_LINEAR",
        },
        "sampling": inference_args,
        "base_seed": args.base_seed,
        "per_configuration_seed": "(base_seed + fixed_number * 1009) % 2**31",
    }
    with open(args.output_dir / "generation_manifest.json", "w") as file:
        json.dump(manifest, file, indent=2)
    print(f"Validated {len(final_outputs)} predictions and wrote generation_manifest.json")


if __name__ == "__main__":
    main()
