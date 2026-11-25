"""Save notebook image grids as separate 300-dpi image objects in a PDF."""
from pathlib import Path
from unittest.mock import patch
import gc
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def save_image_grid_pdf(images, output_path, *, dpi=300, figsize=(60, 60),
                        grid_size=(10, 10), cache_path=None, filenames=None):
    # Reuse the actual notebook plotting helper, including borders and spacing.
    from utils.display import display_images_grid, tensor_to_pil_v2
    captured = []
    def capture(*args, **kwargs):
        captured.append(plt.gcf())
    with patch.object(plt, 'show', capture):
        display_images_grid(images, grid_size=grid_size, figsize=figsize)
    if len(captured) != 1:
        raise RuntimeError('Expected exactly one grid Figure from display_images_grid')
    fig = captured[0]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        for ax in fig.axes:
            for image in ax.images:
                # PDF imshow artists are already bitmap objects. Do not rasterize
                # entire axes/figures; keep all 100 images separately editable.
                image.set_rasterized(False)
                image.set_interpolation('nearest')
        with mpl.rc_context({'pdf.fonttype': 42, 'pdf.use14corefonts': False,
                             'image.composite_image': False}):
            fig.savefig(output_path, format='pdf', dpi=dpi, bbox_inches='tight')
        count = sum(len(ax.images) for ax in fig.axes)
        if count != len(images):
            raise ValueError(f'Grid contains {count} image objects for {len(images)} inputs')
        if cache_path is not None:
            cache_path = Path(cache_path)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            display_arrays = [
                image if isinstance(image, np.ndarray) else np.asarray(tensor_to_pil_v2(image))
                for image in images
            ]
            np.savez_compressed(cache_path, images=np.stack(display_arrays),
                                filenames=np.asarray([] if filenames is None else filenames, dtype=str))
        print(f'Saved {count} separate image objects at {dpi} dpi: {output_path}', flush=True)
        return output_path
    finally:
        plt.close(fig)
        del fig, captured
        gc.collect()
