import click


@click.group()
def glass():
    """CLI for training and running the glass uncond_denoiser workflows."""
    pass


from glass.cli.train import train_score, train_spec
from glass.cli.denoise import uncond_denoise, cond_denoise
from glass.cli.analysis import (
    plot_loss,
    write_spec_feature,
    calc_metrics,
    build_ref_stats,
)

glass.add_command(train_score)
glass.add_command(train_spec)
glass.add_command(plot_loss)
glass.add_command(uncond_denoise)
glass.add_command(cond_denoise)
glass.add_command(write_spec_feature)
glass.add_command(calc_metrics)
glass.add_command(build_ref_stats)

# Register Tersoff potential subcommands (md, energy) under the `glass` group.
from glass.potentials.torch_tersoff.cli import md as _tersoff_md
from glass.potentials.torch_tersoff.cli import energy as _tersoff_energy

glass.add_command(_tersoff_md, name="md")
glass.add_command(_tersoff_energy, name="energy")


if __name__ == "__main__":
    glass()