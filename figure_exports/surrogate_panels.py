"""PDF exports for the current Supplementary Figure 7 source panels."""
from pathlib import Path
from unittest.mock import patch
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.font_manager import FontProperties
import numpy as np


def _numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def _plot_pdf(plotter, args, output_path, *, num_samples, dpi=300):
    captured = []
    def capture(*args, **kwargs):
        captured.append(plt.gcf())
    with patch.object(plt, 'show', capture):
        plotter(*args, num_samples=num_samples)
    if len(captured) != 1:
        raise RuntimeError('Expected one Figure from the original plotting helper')
    fig = captured[0]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # These row labels and panel letters were assembled outside the notebook
        # in the manuscript. Include them here as separate editable Text artists.
        from utils.config import FPATH
        is_error_panel = plotter.__name__ == 'display_images_with_ssim'
        labels = (['Simulation\n(Ground truth)', 'Prediction', 'Absolute error', 'SSIM score']
                  if is_error_panel else
                  ['Seed', 'Latent', 'Prediction', 'Predicted\nsimulation', 'Ground truth'])
        font_size = 44 if is_error_panel else 20
        font = FontProperties(fname=str(FPATH), size=font_size)
        fig.canvas.draw()
        for row, label in enumerate(labels):
            row_axes = fig.axes[row * num_samples:(row + 1) * num_samples]
            center_y = sum(ax.get_position().y0 + ax.get_position().height / 2
                           for ax in row_axes) / len(row_axes)
            fig.text(-0.015, center_y, label, ha='right', va='center',
                     fontproperties=font, rasterized=False)
        fig.text(-0.15 if is_error_panel else -0.25, 1.01,
                 'B' if is_error_panel else 'A', ha='left', va='bottom',
                 fontproperties=font, rasterized=False)
        for ax in fig.axes:
            for image in ax.images:
                # PDF image artists remain separate bitmaps; only their sampled
                # resolution is controlled by dpi. Never rasterize the text.
                image.set_interpolation('nearest')
                image.set_rasterized(False)
        for text in fig.findobj(match=Text):
            text.set_rasterized(False)
        with mpl.rc_context({'pdf.fonttype': 42, 'pdf.use14corefonts': False,
                             'text.usetex': False, 'image.composite_image': False}):
            fig.savefig(output_path, format='pdf', dpi=dpi, bbox_inches='tight')
        print(f'Saved Supplementary Fig. 7 panel ({dpi} dpi images): {output_path}', flush=True)
    finally:
        plt.close(fig)
    return output_path


def save_training_panel(seed, actual_latent, predicted_latent, prediction, truth,
                        output_path, *, cache_path=None, num_samples=4, dpi=300):
    from utils.display import display_predicted_images_5X
    arrays = (seed, actual_latent, predicted_latent, prediction, truth)
    result = _plot_pdf(display_predicted_images_5X, arrays, output_path,
                       num_samples=num_samples, dpi=dpi)
    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, **{
            key: _numpy(value[:num_samples]) for key, value in zip(
                ('seed', 'actual_latent', 'predicted_latent', 'prediction', 'truth'), arrays)
        })
    return result


def save_error_panel(reference, prediction, output_path, *, cache_path=None,
                     sample_indices=(), num_samples=7, dpi=300):
    from utils.display import display_images_with_ssim, compute_ssim_high_precision
    result = _plot_pdf(display_images_with_ssim, (reference, prediction), output_path,
                       num_samples=num_samples, dpi=dpi)
    scores = np.asarray(compute_ssim_high_precision(
        reference[:num_samples], prediction[:num_samples]))
    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, reference=_numpy(reference[:num_samples]),
                            prediction=_numpy(prediction[:num_samples]), scores=scores,
                            sample_indices=np.asarray(sample_indices, dtype=int))
    return scores
