"""Export original Matplotlib artists as editable manuscript PDFs.

Images made with imshow remain embedded bitmaps; plotted artists and text
remain vectors. No image tracing, inference, or changes to analysis data.
"""
from pathlib import Path
import matplotlib as mpl


def save_vector_pdf(figure, path, *, panel=None):
    """Save a figure, or crop to one Axes including its labels and legend.

    TrueType embedding preserves editable text on PDF import. Temporarily
    disable artist/axes rasterization (including automatic colorbar meshes),
    then restore every setting even if saving fails. The Figure need not still
    be registered with pyplot, so this also works after show/close.
    """
    path = Path(path)
    if path.suffix.lower() != '.pdf':
        raise ValueError('Vector manuscript exports must use a .pdf suffix')
    if panel is not None and panel.figure is not figure:
        raise ValueError('The selected panel must belong to this figure')
    path.parent.mkdir(parents=True, exist_ok=True)
    rasterized = [artist for artist in figure.findobj() if artist.get_rasterized()]
    axes_settings = [(ax, ax.get_rasterization_zorder()) for ax in figure.axes]
    hidden_artists = []
    if panel is not None:
        hidden_artists = [
            (artist, artist.get_visible())
            for artist in [*figure.axes, *figure.legends, *figure.texts]
            if artist is not panel
        ]
    try:
        for artist, _ in hidden_artists:
            artist.set_visible(False)
        for artist in rasterized:
            artist.set_rasterized(False)
        for ax, _ in axes_settings:
            ax.set_rasterization_zorder(None)
        with mpl.rc_context({'pdf.fonttype': 42, 'pdf.use14corefonts': False,
                             'text.usetex': False}):
            bbox = 'tight'
            if panel is not None:
                figure.canvas.draw()
                bbox = panel.get_tightbbox(figure.canvas.get_renderer())
                bbox = bbox.transformed(figure.dpi_scale_trans.inverted()).padded(0.1)
            figure.savefig(path, format='pdf', bbox_inches=bbox)
    finally:
        for artist, visible in hidden_artists:
            artist.set_visible(visible)
        for artist in rasterized:
            artist.set_rasterized(True)
        for ax, zorder in axes_settings:
            ax.set_rasterization_zorder(zorder)
    print(f'Saved vector PDF: {path}')
    return path
