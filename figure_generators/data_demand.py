"""Current Figure 3 data-efficiency panels; no supplementary exports."""
from .main_figures import generate_tasks


def generate_fig3cd(output_dir, **kwargs):
    return generate_tasks(output_dir, keys=['3CD'], **kwargs)


def generate_fig3ef(output_dir, **kwargs):
    return generate_tasks(output_dir, keys=['3EF'], **kwargs)
