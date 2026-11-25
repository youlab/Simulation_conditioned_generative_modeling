"""Save manuscript panels with raster images and editable TrueType text.

Export only: no training, inference, preprocessing or metric changes.
"""
from pathlib import Path
from unittest.mock import patch
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np


def save_hybrid_pdf(figure, path, *, dpi=300, rasterize_image_axes=False):
    """Keep text/legends editable; sample displayed image artists at dpi."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    settings = []
    image_settings = []
    text_settings = [(artist, artist.get_rasterized())
                     for artist in figure.findobj(match=Text)]
    try:
        for ax in figure.axes:
            settings.append((ax, ax.get_rasterized(), ax.get_rasterization_zorder()))
            ax.set_rasterization_zorder(None)
            if rasterize_image_axes and ax.images:
                # Skeleton overlays are part of the image; figure legends stay vector.
                if any(t.get_visible() and t.get_text() for t in ax.texts):
                    raise ValueError('Rasterizing an image axes would flatten its text')
                ax.set_rasterized(True)
            else:
                ax.set_rasterized(False)
            for im in ax.images:
                image_settings.append((im, im.get_interpolation()))
                im.set_interpolation('nearest')
        for artist, _ in text_settings:
            artist.set_rasterized(False)
        with mpl.rc_context({'pdf.fonttype': 42, 'pdf.use14corefonts': False,
                             'text.usetex': False, 'image.composite_image': False}):
            figure.savefig(path, format='pdf', dpi=dpi, bbox_inches='tight')
    finally:
        for ax, rasterized, zorder in settings:
            ax.set_rasterized(rasterized)
            ax.set_rasterization_zorder(zorder)
        for im, interpolation in image_settings:
            im.set_interpolation(interpolation)
        for artist, rasterized in text_settings:
            artist.set_rasterized(rasterized)
    print(f'Saved hybrid PDF ({dpi} dpi images): {path}', flush=True)
    return path


def capture_figure(plotter, *args, **kwargs):
    captured = []
    with patch.object(plt, 'show', lambda *a, **k: captured.append(plt.gcf())):
        plotter(*args, **kwargs)
    if len(captured) != 1:
        raise RuntimeError(f'Expected one displayed figure, got {len(captured)}')
    return captured[0]


def add_row_labels(figure, labels, columns, font_path, *, fontsize=24):
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname=str(font_path), size=fontsize)
    figure.canvas.draw()
    for row, label in enumerate(labels):
        row_axes = figure.axes[row * columns:(row + 1) * columns]
        y = np.mean([a.get_position().y0 + a.get_position().height / 2 for a in row_axes])
        figure.text(-0.015, y, label, ha='right', va='center', fontproperties=font)


def save_pixel_pdf(image_path, output_path, *, dpi=300):
    """Embed the original pixels at exactly dpi without fabricating detail."""
    from PIL import Image
    image_path, output_path = Path(image_path), Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as im:
        rgb = Image.new('RGB', im.size, 'white')
        if im.mode == 'RGBA':
            rgb.paste(im, mask=im.getchannel('A'))
        else:
            rgb.paste(im.convert('RGB'))
        # Pillow's PDF writer uses JPEG for RGB, so use Matplotlib's lossless PDF
        # image stream and an axes that fills the entire native-pixel-sized page.
        w, h = rgb.size
        fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(np.asarray(rgb), interpolation='none', aspect='auto')
        ax.axis('off')
        fig.savefig(output_path, format='pdf', dpi=dpi, pad_inches=0)
        plt.close(fig)
    print(f'Saved pixel PDF ({w} x {h}, {dpi} dpi): {output_path}', flush=True)
    return output_path


def save_images_only_pdf(figure, path, *, dpi=300):
    """Save original image artists at 300 dpi, temporarily hiding all text.

    Existing image arrays, colormaps, limits and positions are retained. Axes
    without images (e.g. a score-only row) are omitted from the saved crop.
    """
    text_settings = [(t, t.get_visible()) for t in figure.findobj(match=Text)]
    empty_axes = [(ax, ax.get_visible()) for ax in figure.axes if not ax.images]
    try:
        for t, _ in text_settings:
            t.set_visible(False)
        for ax, _ in empty_axes:
            ax.set_visible(False)
        return save_hybrid_pdf(figure, path, dpi=dpi)
    finally:
        for t, visible in text_settings:
            t.set_visible(visible)
        for ax, visible in empty_axes:
            ax.set_visible(visible)


def save_replicate_panels(plotter, output_dir):
    """Capture the existing panels; redirect only their saves to 300 dpi PDFs."""
    from matplotlib.figure import Figure
    original_savefig = Figure.savefig
    paths = []
    for letter, source in [('A', 'Experiment'), ('B', 'Diffusion')]:
        output = Path(output_dir) / f'SuppFig16{letter}_{source.lower()}_replicates_hybrid_300dpi.pdf'
        def redirect(fig, *args, **kwargs):
            if str(args[0]).endswith('.pdf'):
                with patch.object(Figure, 'savefig', original_savefig):
                    paths.append(save_hybrid_pdf(fig, output))
        with patch.object(Figure, 'savefig', redirect), patch.object(plt, 'show', lambda: None):
            plotter(source)
    return paths


def save_conditioning_metric_panels(comparison_data, cmmd_values, colors, output_dir):
    """Separate vector plots for S18A, reusing the notebook's summary data."""
    from . import export_figure as save_vector_pdf
    paths=[]
    names=['LPIPS','SSIM','ORB','CMMD']
    for number,name in enumerate(names):
        fig,ax=plt.subplots(figsize=(10,9))
        for position,index in enumerate([1,2]):
            label,means,errors=comparison_data[index]
            ax.bar(position, cmmd_values[index] if name=='CMMD' else means[number],
                   yerr=None if name=='CMMD' else errors[number],capsize=6,
                   color=colors[index],edgecolor='white',linewidth=.8,label=label)
        ax.set_xticks([])
        ax.set_ylabel('Distance' if name=='CMMD' else 'Metric score')
        ax.set_xlabel(name+(' ↓' if name in ['CMMD','LPIPS'] else ' ↑'))
        ax.grid(axis='y',linestyle='--',alpha=.35);ax.set_axisbelow(True)
        fig.legend(*ax.get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False,fontsize=28)
        fig.subplots_adjust(left=.22,right=.98,top=.97,bottom=.30)
        paths.append(save_vector_pdf(fig,Path(output_dir)/f'SuppFig18A_{name}_individual.pdf'))
        plt.close(fig)
    return paths


def save_conditioning_kde_panels(figure, axes, output_dir, xlabel):
    from . import export_figure as save_vector_pdf
    paths=[]
    for ax,name in zip(axes,['simulation_conditioned','seed_conditioned']):
        old_xlabel,old_ylabel=ax.get_xlabel(),ax.get_ylabel()
        try:
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Kernel density\nestimate')
            ax.tick_params(labelleft=True)
            paths.append(save_vector_pdf(figure,Path(output_dir)/f'SuppFig18B_{name}_individual.pdf',panel=ax))
        finally:
            ax.set_xlabel(old_xlabel);ax.set_ylabel(old_ylabel)
    return paths
