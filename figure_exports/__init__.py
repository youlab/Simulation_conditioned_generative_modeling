"""Shared publication exports for all paper notebooks.

Use export_figure(fig, path, mode='vector'|'hybrid'|'images_only'). Image arrays,
metric values, and checkpoint selection remain the responsibility of callers.
"""
from functools import partial
from .vector import save_vector_pdf as _save_vector_pdf
from .panels import (
    save_hybrid_pdf as _save_hybrid_pdf,
    save_images_only_pdf as _save_images_only_pdf,
    save_pixel_pdf,
    capture_figure,
    add_row_labels,
    save_replicate_panels,
    save_conditioning_metric_panels,
    save_conditioning_kde_panels,
)


def export_figure(figure, path, *, mode='vector', **options):
    """Export an existing figure with the established publication settings."""
    handlers = {'vector': _save_vector_pdf, 'hybrid': _save_hybrid_pdf,
                'images_only': _save_images_only_pdf}
    if mode not in handlers:
        raise ValueError(f'Unknown figure export mode: {mode}')
    return handlers[mode](figure, path, **options)


# Small named adapters keep specialized notebook layouts separate from export
# mechanics while routing ordinary figure saving through the shared entry point.
save_vector_pdf = export_figure
save_hybrid_pdf = partial(export_figure, mode='hybrid')
save_images_only_pdf = partial(export_figure, mode='images_only')


def save_image_grid_pdf(*args, **kwargs):
    from .grids import save_image_grid_pdf as save
    return save(*args, **kwargs)


def save_ssim_hybrid(*args, **kwargs):
    from .ssim import save_ssim_hybrid as save
    return save(*args, **kwargs)


def save_combined_ssim_hybrid(*args, **kwargs):
    from .ssim import save_combined_ssim_hybrid as save
    return save(*args, **kwargs)


def save_training_panel(*args, **kwargs):
    from .surrogate_panels import save_training_panel as save
    return save(*args, **kwargs)


def save_error_panel(*args, **kwargs):
    from .surrogate_panels import save_error_panel as save
    return save(*args, **kwargs)
