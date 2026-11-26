"""Display helpers for colony skeletons."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.morphology import binary_dilation, disk

from cldm.morpho_features import (
    DEFAULT_SKELETON_CONFIG,
    endpoint_pixel_map,
    junction_pixel_map,
    skeletonize_colony,
)


def _as_rgb_uint8(image):
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = image[..., :3]
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be grayscale, RGB, or RGBA")
    image = image.astype(float)
    finite = image[np.isfinite(image)]
    if finite.size and finite.max() <= 1.0:
        image *= 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def create_skeleton_overlay(image, skeleton, skeleton_color=(255, 45, 45),
                            alpha=0.90, line_radius=1):
    """Blend a binary skeleton into an image and return uint8 RGB."""
    base = _as_rgb_uint8(image)
    skeleton = np.asarray(skeleton, dtype=bool)
    if base.shape[:2] != skeleton.shape:
        raise ValueError(f"image shape {base.shape[:2]} != skeleton shape {skeleton.shape}")
    visible = (
        binary_dilation(skeleton, footprint=disk(int(line_radius)))
        if line_radius > 0 else skeleton
    )
    overlay = base.astype(float)
    color = np.asarray(skeleton_color, dtype=float)
    overlay[visible] = (1.0 - alpha) * overlay[visible] + alpha * color
    return np.clip(overlay, 0, 255).astype(np.uint8)


def plot_skeleton_result(image, result, ax=None, show_nodes=True,
                         title=None, skeleton_color=(255, 45, 45),
                         alpha=0.90, line_radius=1, save_path=None):
    """Plot a completed skeleton result over the source colony image."""
    overlay = create_skeleton_overlay(
        image, result["final"], skeleton_color, alpha, line_radius
    )
    created_axis = ax is None
    if created_axis:
        _, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(overlay)
    if show_nodes:
        tip_y, tip_x = np.where(endpoint_pixel_map(result["final"]))
        junction_y, junction_x = np.where(junction_pixel_map(result["final"]))
        ax.scatter(tip_x, tip_y, s=18, c="cyan", marker="o",
                   edgecolors="black", linewidths=0.25, label="tips")
        ax.scatter(junction_x, junction_y, s=13, c="royalblue", marker="s",
                   edgecolors="none", label="junction pixels")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
    metrics = result["metrics"]
    default_title = (
        f"Final skeleton overlay | {metrics['branches']} segments | "
        f"{metrics['tips']} tips | {metrics['junctions']} junctions"
    )
    ax.set_title(title or default_title)
    ax.axis("off")
    if save_path is not None:
        plt.imsave(save_path, overlay)
    if created_axis:
        plt.tight_layout()
    return overlay, ax


def analyze_skeleton_image(image, mask, config=DEFAULT_SKELETON_CONFIG,
                           ax=None, show_nodes=True, title=None,
                           save_path=None):
    """Analyze one mask and display its final skeleton."""
    mask = np.asarray(mask, dtype=bool)
    if np.asarray(image).shape[:2] != mask.shape:
        raise ValueError(f"image shape {np.asarray(image).shape[:2]} != mask shape {mask.shape}")
    result = skeletonize_colony(mask, config=config)
    overlay, ax = plot_skeleton_result(
        image, result, ax=ax, show_nodes=show_nodes,
        title=title, save_path=save_path,
    )
    return overlay, result


def analyze_skeleton_batch(
    filenames,
    image_folder,
    color_preprocessor,
    mask_preprocessor,
    n_images=10,
    random_seed=42,
    save_dir=None,
    config=DEFAULT_SKELETON_CONFIG,
):
    """Analyze a reproducible subset and display one large image per row."""
    available = sorted(filenames)
    if not available:
        raise ValueError("No input images were supplied")
    rng = np.random.default_rng(random_seed)
    count = min(int(n_images), len(available))
    selected = list(rng.choice(available, size=count, replace=False))
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(count, 1, figsize=(10, 9 * count), squeeze=False)
    axes = axes.ravel()
    records, results = [], {}
    for ax, filename in zip(axes, selected):
        path = os.path.join(image_folder, filename)
        save_path = (
            os.path.join(save_dir, f"{os.path.splitext(filename)[0]}_skeleton_overlay.png")
            if save_dir is not None else None
        )
        _, result = analyze_skeleton_image(
            color_preprocessor(path), mask_preprocessor(path),
            config=config, ax=ax, save_path=save_path,
        )
        ax.set_title(filename + "\n" + ax.get_title(), fontsize=10)
        results[filename] = result
        scalar_metrics = {
            key: value for key, value in result["metrics"].items()
            if np.isscalar(value) or value is None
        }
        records.append({
            "filename": filename,
            **scalar_metrics,
            "mask_smooth_sigma": config.mask_smooth_sigma,
            "mask_smooth_threshold": config.mask_smooth_threshold,
            "skeleton_pixels": int(result["final"].sum()),
            "broad_core_count": int(result["blob_core_count"]),
            "reconstructed_core_count": int(result["replaced_blob_core_count"]),
        })
    plt.tight_layout()
    plt.show()
    return pd.DataFrame(records).set_index("filename"), results


# Compatibility alias for exploratory-notebook calls.
overlay_lamm_skeleton_on_colony = analyze_skeleton_image
