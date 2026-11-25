"""Supplementary Fig. 2: raster image panels with editable SSIM text."""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity


def _numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def _pil(image):
    # Match utils.display.tensor_to_pil_v2 exactly, including uint8 conversion.
    pixels = np.moveaxis(_numpy(image), 0, -1)
    return Image.fromarray((pixels * 255).astype('uint8').squeeze())


def make_ssim_figure(reference, reconstruction, font_path, num_samples=9):
    """Recreate the original three-row layout without encoding/decoding again."""
    if min(len(reference), len(reconstruction)) < num_samples:
        raise ValueError('Not enough images for the requested SSIM grid')
    fig, axes = plt.subplots(
        3, num_samples, figsize=(num_samples * 4, 12),
        layout='constrained', squeeze=False,
    )
    font = FontProperties(fname=str(font_path), size=60)
    scores = []
    for column in range(num_samples):
        original = _pil(reference[column])
        decoded = _pil(reconstruction[column])
        score = structural_similarity(
            np.array(original.convert('L')) / 255.0,
            np.array(decoded.convert('L')) / 255.0,
            data_range=1.0,
        )
        scores.append(score)
        # Rasterize only image artists. The separate Text artists stay vector.
        for row, image in enumerate((original, decoded)):
            axes[row, column].imshow(
                image, cmap='gray', interpolation='nearest', rasterized=True,
            )
            axes[row, column].axis('off')
        axes[2, column].text(
            0.5, 0.5, f' {score:.3f}', ha='center', va='center',
            fontproperties=font, rasterized=False,
        )
        axes[2, column].axis('off')
    return fig, np.asarray(scores)


def save_ssim_hybrid(reference, reconstruction, output_path, font_path,
                     *, cache_path=None, filenames=(), dpi=300, num_samples=9):
    """Save one grid; optionally cache its exact source arrays for later exports.

    DPI applies at the original Figure's physical dimensions. Upsampling does
    not create additional detail beyond the original 256x256 source arrays.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context({'pdf.fonttype': 42, 'pdf.use14corefonts': False,
                         'text.usetex': False, 'image.composite_image': False}):
        fig, scores = make_ssim_figure(reference, reconstruction, font_path, num_samples)
        try:
            fig.savefig(output_path, format='pdf', dpi=dpi, bbox_inches='tight')
        finally:
            plt.close(fig)
    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path, reference=_numpy(reference[:num_samples]),
            reconstruction=_numpy(reconstruction[:num_samples]), scores=scores,
            filenames=np.asarray(list(filenames)[:num_samples], dtype=str),
        )
    print(f'Saved hybrid PDF ({dpi} dpi images, editable Arial text): {output_path}')
    return scores


def save_combined_ssim_hybrid(cache_paths, output_path, font_path, *, dpi=300):
    """Save the simulation and experimental grids as two pages of one PDF."""
    cache_paths = [Path(path) for path in cache_paths]
    if not all(path.is_file() for path in cache_paths):
        raise FileNotFoundError('Run both individual SSIM export cells first')
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context({'pdf.fonttype': 42, 'pdf.use14corefonts': False,
                         'text.usetex': False, 'image.composite_image': False}):
        with PdfPages(output_path, metadata={
            'Title': 'Supplementary Fig. 2: simulation and experiment SSIM',
            'Subject': '300 dpi raster images with editable Arial SSIM values',
        }) as pdf:
            for path in cache_paths:
                with np.load(path, allow_pickle=False) as cache:
                    fig, scores = make_ssim_figure(
                        cache['reference'], cache['reconstruction'], font_path,
                        num_samples=len(cache['scores']),
                    )
                    try:
                        np.testing.assert_array_equal(scores, cache['scores'])
                        pdf.savefig(fig, dpi=dpi, bbox_inches='tight')
                    finally:
                        plt.close(fig)
    print(f'Saved combined hybrid PDF ({len(cache_paths)} pages): {output_path}')
    return output_path
