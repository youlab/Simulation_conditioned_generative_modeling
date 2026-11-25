"""Current main-figure entry point; uses reviewed notebook recipes."""
from .main_figures import generate_tasks


def generate_fig3(output_dir, **kwargs):
    return generate_tasks(output_dir, keys=['3B'], **kwargs)
