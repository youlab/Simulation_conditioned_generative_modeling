"""Current main-figure entry point; uses reviewed notebook recipes."""
from .main_figures import generate_tasks


def generate_fig5(output_dir, **kwargs):
    return generate_tasks(output_dir, figures=[5], **kwargs)
