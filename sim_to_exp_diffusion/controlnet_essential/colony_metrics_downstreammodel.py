# Save colony areas and boundary tip counts for each image in a JSON file.

import argparse
import numpy as np
import os
import json
from natsort import natsorted


def sample_boundary_ring(mask, radius_frac=0.95, n_samples=1440):
    # sample mask values along a circle just inside the plate edge -> 1D binary profile
    h, w = mask.shape
    cy, cx = h / 2, w / 2
    radius = min(h, w) / 2 * radius_frac
    angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    xs = np.clip((cx + radius * np.cos(angles)).astype(int), 0, w - 1)
    ys = np.clip((cy + radius * np.sin(angles)).astype(int), 0, h - 1)
    return (mask[ys, xs] == 255), xs, ys


def count_circular_runs(binary_profile):
    # count contiguous "on" runs, treating the array as circular (wraps around)
    if binary_profile.all():
        return 1
    if not binary_profile.any():
        return 0
    start = np.argmax(~binary_profile)  # rotate to a "off" sample so no run is split by wraparound
    rolled = np.roll(binary_profile, -start)
    return int(np.sum(np.diff(rolled.astype(int)) == 1))


def count_boundary_tips(mask, radius_frac=0.95, n_samples=1440):
    profile, xs, ys = sample_boundary_ring(mask, radius_frac, n_samples)
    return count_circular_runs(profile), profile, xs, ys


def calculate_colony_metrics(mask):
    """Return JSON-serializable colony area and boundary-tip metrics."""
    colony_area_px = int(np.count_nonzero(mask == 255))
    tips, _, _, _ = count_boundary_tips(mask)
    return {
        "colony_area_px": colony_area_px,
        "colony_area_fraction": float(colony_area_px / mask.size),
        "number_of_tips": int(tips),
    }


def save_colony_metrics(image_folder):
    """Analyze images and save colony_metrics.json one level above the image folder."""
    from cldm.preprocess import preprocess_experimental_backgroundblack_blackandwhite_nocropping

    image_files = natsorted(
        filename
        for filename in os.listdir(image_folder)
        if os.path.isfile(os.path.join(image_folder, filename))
        and os.path.splitext(filename)[1].lower() in {".tif", ".tiff", ".png"}
    )

    records = []
    for filename in image_files:
        image_path = os.path.join(image_folder, filename)
        # Diffusion predictions have a white outer plate boundary. Match the
        # notebook workflow by masking that boundary to black before thresholding,
        # without applying the experimental-image crop.
        mask = preprocess_experimental_backgroundblack_blackandwhite_nocropping(image_path)
        height, width = mask.shape
        records.append({
            "image": {
                "filename": filename,
                "path": image_path,
                "width_px": int(width),
                "height_px": int(height),
            },
            "metrics": calculate_colony_metrics(mask),
        })

    parent_folder = os.path.dirname(os.path.abspath(image_folder))
    output_path = os.path.join(parent_folder, "colony_metrics.json")
    with open(output_path, "w", encoding="utf-8") as file_handle:
        json.dump({"images": records}, file_handle, indent=2)

    print(f"Saved metrics for {len(records)} image(s) to {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Save colony area and boundary-tip metrics for a folder of images."
    )
    parser.add_argument("-i", "--image_folder", required=True, help="Folder containing TIF/TIFF/PNG images")
    # for output folder, save in the location one level up to the input file 
    arguments = parser.parse_args()
    save_colony_metrics(arguments.image_folder)



