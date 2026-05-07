import glob
import os

import click
import torch

from glass.lit.datamodules import StructureSpecDataModule
from glass.lit.modules import LitScoreNet


@click.command(
    "uncond_denoise",
    help="""
Run unconditional denoising (GPU-accelerated) for one or more score models.

Loops over each SCORE_MODEL, finds matching .xyz files under ATOMS_PATH,
and runs N_RUNS independent denoising trajectories per structure.

Checkpoints are expected at: CKPT_DIR/{system}_score_{model}.ckpt
Data modules are loaded from:  SCORE_PATH/data/{system}_{model}/

EXAMPLES:

  # Single model, single file
  glass uncond_denoise --score-model 1.5_2.5_3.5 --system Si \\
      --score-data-path /home/jwguo/03_denoiser/00_score_models \\
      --score-model-path /home/jwguo/03_denoiser/demo_Si/Si_score_1.5_2.5_3.5.ckpt \\
      --init-path /home/jwguo/03_denoiser/reference/start_Si_216/Si_01.xyz

  # Directory input (loads all Si_*.xyz inside)
  glass uncond_denoise --score-model 1.5_2.5_3.5 --system Si \\
      --score-data-path /home/jwguo/03_denoiser/00_score_models \\
      --score-model-path /home/jwguo/03_denoiser/demo_Si/Si_score_1.5_2.5_3.5.ckpt \\
      --init-path /home/jwguo/03_denoiser/reference/start_Si_216

  # Glob pattern input
  glass uncond_denoise --score-model 1.5_2.5_3.5 --system Si \\
      --score-data-path /home/jwguo/03_denoiser/00_score_models \\
      --score-model-path /home/jwguo/03_denoiser/demo_Si/Si_score_1.5_2.5_3.5.ckpt \\
      --init-path "/home/jwguo/03_denoiser/reference/start_Si_216/Si_0[1-3].xyz"
""",
)
@click.option(
    "--score-model",
    "score_models",
    multiple=True,
    required=True,
    help="Score model tag(s). Can be repeated.",
)
@click.option(
    "--system",
    type=str,
    default="Si",
    show_default=True,
    help="System name (e.g. Si, MoS2).",
)
@click.option(
    "--score-data-path",
    type=str,
    required=True,
    help="Root dir containing data/{system}_{model}/ subfolders.",
)
@click.option(
    "--score-model-path",
    type=str,
    required=True,
    help="Full path to the .ckpt checkpoint file.",
)
@click.option(
    "--init-path",
    type=str,
    required=True,
    help="Initial structures: a .xyz file, a glob pattern, or a directory (all {system}_*.xyz inside).",
)
@click.option(
    "--outdir",
    type=str,
    default="denoise_logs/unconditional",
    show_default=True,
    help="Output root directory.",
)
@click.option(
    "--device",
    type=str,
    default="cuda:0",
    show_default=True,
    help="Torch device (e.g. cuda:0, cuda:1, cpu).",
)
@click.option(
    "--tmin",
    type=float,
    default=0.001,
    show_default=True,
    help="Start time for reverse SDE.",
)
@click.option(
    "--tmax",
    type=float,
    default=1.0,
    show_default=True,
    help="End time for reverse SDE.",
)
@click.option(
    "--tstep",
    type=int,
    default=256,
    show_default=True,
    help="Number of SDE time steps.",
)
@click.option(
    "--cutoff",
    type=float,
    default=5.0,
    show_default=True,
    help="Graph cutoff radius (Å).",
)
@click.option(
    "--n-runs",
    type=int,
    default=10,
    show_default=True,
    help="Number of independent runs per structure.",
)
@click.option(
    "--save-traj/--no-save-traj",
    default=True,
    show_default=True,
    help="Save full trajectory or only final frame.",
)
def uncond_denoise(
    score_models,
    system,
    score_data_path,
    score_model_path,
    init_path,
    outdir,
    device,
    tmin,
    tmax,
    tstep,
    cutoff,
    n_runs,
    save_traj,
):
    """GPU unconditional denoising over one or more score models."""
    import copy
    import ase
    from ase.io import read

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    click.echo(f"Using device: {device}")
    if torch.cuda.is_available():
        click.echo(f"GPU: {torch.cuda.get_device_name(device.index or 0)}")

    from glass.nn import periodic_radius_graph
    from glass.lit.functions.get_atoms import initialize_atoms

    def _to_device(atoms):
        _, species, pos, cell = initialize_atoms(atoms)
        return (
            species.to(device),
            pos.to(device=device, dtype=torch.float32),
            cell.to(device=device, dtype=torch.float32),
        )

    def _prior_score(species, pos, cell, t, cut, score_net, diffuser):
        edge_index, edge_vec = periodic_radius_graph(pos, cut, cell)
        edge_attr = torch.hstack([edge_vec, edge_vec.norm(dim=-1, keepdim=True)])
        return score_net.ema_model(species, edge_index, edge_attr, t, diffuser.sigma(t))

    def _denoise_by_sde(species, pos, cell, cut, score_fn, ts, diffuser, save_traj):
        ts = ts.to(pos.device).view(-1, 1)
        f, g, g2 = diffuser.f, diffuser.g, diffuser.g2
        traj = [pos.detach().cpu().clone()] if save_traj else None
        pos = pos.detach()

        for i, t in enumerate(ts[1:]):
            dt = ts[i + 1] - ts[i]
            eps = dt.abs().sqrt() * torch.randn_like(pos)
            score = score_fn(species, pos, cell, t, cut)
            disp = (f(t) * pos - g2(t) * score) * dt + g(t) * eps
            pos = (pos + disp).detach()
            if save_traj:
                traj.append(pos.cpu().clone())

        return traj if save_traj else pos

    for score_model in score_models:
        main_id = f"{system}-{score_model}"
        click.echo(f"\n=== Score model: {main_id} ===")

        resolved_data_path = score_data_path.format(
            score_model=score_model, system=system
        )
        resolved_model_path = score_model_path.format(
            score_model=score_model, system=system
        )

        click.echo(f"  Data path:  {resolved_data_path}")
        click.echo(f"  Model path: {resolved_model_path}")

        datamodule = StructureSpecDataModule(
            data_dir=resolved_data_path,
            cutoff=cutoff,
            train_prior=True,
            k=0.8,
            train_size=0.9,
            scale_y=1.0,
            dup=128,
            batch_size=32,
            num_workers=8,
        )
        datamodule.setup()
        diffuser = datamodule.train_set.diffuser

        ckpt_path = resolved_model_path
        if not os.path.exists(ckpt_path):
            raise click.ClickException(f"Checkpoint not found: {ckpt_path}")
        score_net = LitScoreNet.load_from_checkpoint(ckpt_path, map_location=device)
        score_net.eval()
        score_net.ema_model.to(device)
        score_net.ema_model.eval()

        ts_torch = torch.linspace(tmax, tmin, tstep, device=device)

        resolved_init_path = init_path.format(system=system, score_model=score_model)
        if os.path.isfile(resolved_init_path):
            xyz_files = [resolved_init_path]
        elif os.path.isdir(resolved_init_path):
            xyz_files = sorted(
                glob.glob(os.path.join(resolved_init_path, f"{system}_*.xyz"))
            )
        else:
            xyz_files = sorted(glob.glob(resolved_init_path))  # treat as glob pattern
        if not xyz_files:
            click.echo(
                f"  Warning: no .xyz files found for {resolved_init_path}, skipping."
            )
            continue

        for sample_tag in xyz_files:
            init_atoms = read(sample_tag, "-1")
            sub_id = os.path.basename(sample_tag).replace(".xyz", "")
            click.echo(sub_id)

            run_outdir = os.path.join(outdir, main_id, sub_id)
            os.makedirs(run_outdir, exist_ok=True)

            for i in range(n_runs):
                species, pos, cell = _to_device(copy.deepcopy(init_atoms))
                cell_np = cell.detach().cpu().numpy()

                def score_fn(sp, p, c, t, co, _sn=score_net, _df=diffuser):
                    return _prior_score(sp, p, c, t, co, _sn, _df)

                with torch.no_grad():
                    result = _denoise_by_sde(
                        species,
                        pos,
                        cell,
                        cutoff,
                        score_fn,
                        ts_torch,
                        diffuser,
                        save_traj,
                    )

                if save_traj:
                    traj = []
                    for p in result:
                        a = ase.Atoms(
                            numbers=init_atoms.numbers,
                            positions=p.numpy(),
                            cell=cell_np,
                            pbc=[True] * 3,
                        )
                        a.wrap()
                        traj.append(a)
                    ase.io.write(f"{run_outdir}/{i:02}_traj.xyz", traj)
                    ase.io.write(f"{run_outdir}/{i:02}_final.xyz", traj[-1])
                else:
                    final = ase.Atoms(
                        numbers=init_atoms.numbers,
                        positions=result.cpu().numpy(),
                        cell=cell_np,
                        pbc=[True] * 3,
                    )
                    final.wrap()
                    ase.io.write(f"{run_outdir}/{i:02}_final.xyz", final)

                click.echo(f"  {sub_id} run #{i} done")

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


@click.command(
    "cond_denoise",
    help="""
Run conditional (guided) denoising with spectral/structural guidance.

Supported guidance types:
  pdf    -- pair distribution function (DifferentiableRDF, CPU)
  adf    -- angular distribution function (DifferentiableADF, CPU)
  xrd    -- X-ray diffraction I(q) (DifferentiableXRD, GPU)
  nd     -- neutron diffraction I(q) (DifferentiableND, GPU)
  exafs  -- EXAFS spectrum via LitSpecNet (GPU)
  xanes  -- XANES spectrum via LitSpecNet (GPU)

EXAMPLES:

  # PDF guidance (computational ref)
  glass cond_denoise \\
      --score-model "1.5_2.5_3.5" --system Si \\
      --score-data-path "./data/{system}_{score_model}" \\
      --score-model-path "./models/pre_trained/{system}_{score_model}/model.ckpt" \\
      --init-path "/home/jwguo/03_denoiser/reference/start_{system}_216/{system}_2.0_*.xyz" \\
      --ref-path /home/jwguo/03_denoiser/reference/amorph_Si_216 \\
      --guidance-type pdf --rho 1000 --device cuda:0 --n-runs 10

  # XRD guidance
  glass cond_denoise \\
      --score-model "1.5_2.5_3.5" --system Si \\
      --score-data-path "./data/{system}_{score_model}" \\
      --score-model-path "./models/pre_trained/{system}_{score_model}/model.ckpt" \\
      --init-path "/home/jwguo/03_denoiser/reference/start_{system}_216/{system}_2.0_*.xyz" \\
      --ref-path /home/jwguo/03_denoiser/reference/amorph_Si_216 \\
      --guidance-type xrd --element-names Si --rho 5 --device cuda:0

  # EXAFS/XANES guidance
  glass cond_denoise \\
      --score-model "1.5_2.5_3.5" --system Si \\
      --score-data-path "./data/{system}_{score_model}" \\
      --score-model-path "./models/pre_trained/{system}_{score_model}/model.ckpt" \\
      --init-path "/home/jwguo/03_denoiser/reference/start_{system}_216/{system}_2.0_*.xyz" \\
      --ref-path /home/jwguo/03_denoiser/reference/amorph_Si_216 \\
      --guidance-type exafs --spec-model-path ./models/Si_exafs.ckpt --rho 1e8 --device cuda:0

  # PDF guidance (experimental data)
  glass cond_denoise \\
      --score-model "1.5_2.5_3.5" --system Si \\
      --score-data-path "./data/{system}_{score_model}" \\
      --score-model-path "./models/pre_trained/{system}_{score_model}/model.ckpt" \\
      --init-path "/home/jwguo/03_denoiser/reference/start_{system}_216/{system}_2.0_*.xyz" \\
      --exp-data ./data/exp_gr_si.json \\
      --guidance-type pdf --rho 1000 --device cuda:0
""",
)
@click.option(
    "--score-model",
    "score_models",
    multiple=True,
    required=True,
    help="Score model tag(s). Can be repeated.",
)
@click.option(
    "--system", type=str, default="Si", show_default=True, help="System name."
)
@click.option(
    "--score-data-path",
    type=str,
    required=True,
    help="Root dir containing data/{system}_{model}/ subfolders.",
)
@click.option(
    "--score-model-path",
    type=str,
    required=True,
    help="Full path to .ckpt checkpoint file (supports {system}, {score_model} placeholders).",
)
@click.option(
    "--init-path",
    type=str,
    required=True,
    help="Initial structures: a .xyz file, glob pattern, or directory.",
)
@click.option(
    "--ref-path",
    type=str,
    default=None,
    help="Directory of reference .xyz files (computational guidance). Matched by filename.",
)
@click.option(
    "--exp-data",
    type=str,
    default=None,
    help="JSON file with experimental data (keys: x, y or r, g). Used as global guidance target.",
)
@click.option(
    "--guidance-type",
    type=click.Choice(["pdf", "adf", "xrd", "nd", "exafs", "xanes"]),
    default="pdf",
    show_default=True,
    help="Guidance type.",
)
@click.option(
    "--rho",
    "rho_list",
    multiple=True,
    type=float,
    default=[1000.0],
    show_default=True,
    help="Guidance strength(s). Can be repeated.",
)
@click.option(
    "--outdir",
    type=str,
    default="denoise_logs/guided",
    show_default=True,
    help="Output root directory.",
)
@click.option(
    "--device", type=str, default="cuda:0", show_default=True, help="Torch device."
)
@click.option(
    "--tmin",
    type=float,
    default=0.001,
    show_default=True,
    help="Start time for reverse SDE.",
)
@click.option(
    "--tmax",
    type=float,
    default=1.0,
    show_default=True,
    help="End time for reverse SDE.",
)
@click.option(
    "--tstep",
    type=int,
    default=256,
    show_default=True,
    help="Number of SDE time steps.",
)
@click.option(
    "--cutoff",
    type=float,
    default=5.0,
    show_default=True,
    help="Graph cutoff radius (Å).",
)
@click.option(
    "--n-runs",
    type=int,
    default=10,
    show_default=True,
    help="Number of independent runs per structure.",
)
@click.option(
    "--save-traj/--no-save-traj",
    default=False,
    show_default=True,
    help="Save full trajectory or only final frame.",
)
# --- pdf options ---
@click.option(
    "--bin-size",
    type=int,
    default=100,
    show_default=True,
    help="[pdf] Number of RDF bins.",
)
# --- adf options ---
@click.option(
    "--angle-bins",
    type=int,
    default=100,
    show_default=True,
    help="[adf] Number of angle bins.",
)
@click.option(
    "--adf-sigma",
    type=float,
    default=0.1,
    show_default=True,
    help="[adf] Gaussian kernel width for ADF.",
)
@click.option(
    "--adf-cutoff",
    type=float,
    default=3.5,
    show_default=True,
    help="[adf] Cutoff radius for ADF triplet search (Å).",
)
# --- xrd/nd options ---
@click.option(
    "--element-names",
    "element_names",
    multiple=True,
    default=(),
    help="[xrd/nd] Element name(s) in order (e.g. --element-names Si). Can be repeated.",
)
@click.option(
    "--qmin",
    type=float,
    default=1.0,
    show_default=True,
    help="[xrd/nd] Minimum q value (Å⁻¹).",
)
@click.option(
    "--qmax",
    type=float,
    default=20.0,
    show_default=True,
    help="[xrd/nd] Maximum q value (Å⁻¹).",
)
@click.option(
    "--qstep",
    type=float,
    default=0.1,
    show_default=True,
    help="[xrd/nd] Q step size (Å⁻¹).",
)
@click.option(
    "--biso",
    type=float,
    default=1.5,
    show_default=True,
    help="[xrd/nd] Debye-Waller B factor.",
)
# --- exafs/xanes options ---
@click.option(
    "--spec-model-path",
    type=str,
    default=None,
    help="[exafs/xanes] Path to LitSpecNet checkpoint.",
)
def cond_denoise(
    score_models,
    system,
    score_data_path,
    score_model_path,
    init_path,
    ref_path,
    exp_data,
    guidance_type,
    rho_list,
    outdir,
    device,
    tmin,
    tmax,
    tstep,
    cutoff,
    n_runs,
    save_traj,
    bin_size,
    angle_bins,
    adf_sigma,
    adf_cutoff,
    element_names,
    qmin,
    qmax,
    qstep,
    biso,
    spec_model_path,
):
    """GPU conditional denoising with spectral/structural guidance."""
    import copy
    import math
    import numpy as np
    import ase
    from ase.io import read
    from torch import nn
    from glass.nn import periodic_radius_graph
    from glass.lit.functions.get_atoms import initialize_atoms
    from glass.lit.modules import (
        DifferentiableRDF,
        DifferentiableADF,
        DifferentiableXRD,
        DifferentiableND,
        LitSpecNet,
    )

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    click.echo(f"Using device: {device}")
    if torch.cuda.is_available():
        click.echo(f"GPU: {torch.cuda.get_device_name(device.index or 0)}")

    # --- validate inputs ---
    if not ref_path and not exp_data:
        raise click.ClickException(
            "Provide either --ref-path (computational) or --exp-data (experimental)."
        )
    if ref_path and exp_data:
        raise click.ClickException("--ref-path and --exp-data are mutually exclusive.")
    if guidance_type in ("xrd", "nd") and not element_names:
        raise click.ClickException(
            f"--element-names required for guidance-type '{guidance_type}'."
        )
    if guidance_type in ("exafs", "xanes") and not spec_model_path:
        raise click.ClickException(
            f"--spec-model-path required for guidance-type '{guidance_type}'."
        )

    # --- build guidance model ---
    if guidance_type == "pdf":
        guidance_model = DifferentiableRDF(cutoff=cutoff, bin_size=bin_size, sigma=0.15)
        guidance_model.eval()
    elif guidance_type == "adf":
        guidance_model = DifferentiableADF(
            cutoff=adf_cutoff,
            angle_bins=angle_bins,
            angle_range=[0, math.pi],
            sigma=adf_sigma,
            normalize=False,
        )
        guidance_model.eval()
    elif guidance_type == "xrd":
        guidance_model = DifferentiableXRD(
            q_vals=[qmin, qmax, qstep],
            element_names=list(element_names),
            biso=biso,
        )
        guidance_model.to(device).eval()
    elif guidance_type == "nd":
        guidance_model = DifferentiableND(
            q_vals=[qmin, qmax, qstep],
            element_names=list(element_names),
            biso=biso,
        )
        guidance_model.to(device).eval()
    elif guidance_type in ("exafs", "xanes"):
        spec_net_g = LitSpecNet.load_from_checkpoint(
            spec_model_path, map_location=device
        )
        spec_net_g.eval()
        spec_net_g.ema_model.to(device).eval()
        guidance_model = spec_net_g.ema_model
    click.echo(f"Guidance type: {guidance_type}")

    # --- load experimental target (global, interpolated once) ---
    exp_target_y = None
    if exp_data:
        import json as _json

        with open(exp_data) as _f:
            _d = _json.load(_f)
        x_exp = np.array(_d.get("x", _d.get("r")))
        y_exp = np.array(_d.get("y", _d.get("g")))

        if guidance_type == "pdf":
            bin_edges = np.linspace(0, cutoff, bin_size + 1)
            x_grid = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        elif guidance_type == "adf":
            x_grid = np.linspace(0, math.pi, angle_bins)
        elif guidance_type in ("xrd", "nd"):
            x_grid = np.arange(qmin, qmax, qstep)
        else:
            raise click.ClickException(
                f"--exp-data not yet supported for guidance-type '{guidance_type}'."
            )

        y_interp = np.interp(x_grid, x_exp, y_exp)
        exp_target_y = torch.from_numpy(y_interp).float().unsqueeze(0).to(device)
        click.echo(f"Loaded experimental target from {exp_data} (n_pts={len(x_grid)})")

    def _to_device(atoms):
        _, species, pos, cell = initialize_atoms(atoms)
        return (
            species.to(device),
            pos.to(device=device, dtype=torch.float32),
            cell.to(device=device, dtype=torch.float32),
        )

    def _prior_score(species, pos, cell, t, cut, score_net, diffuser):
        edge_index, edge_vec = periodic_radius_graph(pos, cut, cell)
        edge_attr = torch.hstack([edge_vec, edge_vec.norm(dim=-1, keepdim=True)])
        return score_net.ema_model(species, edge_index, edge_attr, t, diffuser.sigma(t))

    class LikelihoodScore(nn.Module):
        def __init__(
            self, score_net, guidance_model, target_y, rho, diffuser, guidance_type, cut
        ):
            super().__init__()
            self.score_net = score_net
            self.guidance_model = guidance_model
            self.target_y = target_y
            self.rho = rho
            self.diffuser = diffuser
            self.guidance_type = guidance_type
            self.cut = cut

        def forward(self, species, pos, cell, t, cut):
            with torch.enable_grad():
                pos = pos.detach().clone().requires_grad_(True)
                edge_index, edge_vec = periodic_radius_graph(pos, cut, cell)
                edge_attr = torch.hstack(
                    [edge_vec, edge_vec.norm(dim=-1, keepdim=True)]
                )
                sigma = self.diffuser.sigma(t)
                with torch.no_grad():
                    score = self.score_net(species, edge_index, edge_attr, t, sigma)
                est_clean_pos = pos + sigma.pow(2) * score

                if self.guidance_type in ("pdf", "adf"):
                    pred_y = self.guidance_model(
                        est_clean_pos.cpu(), species.cpu(), cell.cpu()
                    )[1].to(pos.device)
                    norm = torch.linalg.norm(
                        self.target_y - pred_y, dim=1, keepdim=True
                    )
                elif self.guidance_type in ("xrd", "nd"):
                    pred_y = self.guidance_model(est_clean_pos, species)
                    norm = torch.linalg.norm(self.target_y - pred_y)
                elif self.guidance_type in ("exafs", "xanes"):
                    ei2, ev2 = periodic_radius_graph(est_clean_pos, self.cut, cell)
                    ea2 = torch.hstack([ev2, ev2.norm(dim=-1, keepdim=True)])
                    pred_y = self.guidance_model(species, ei2, ea2)
                    norm = torch.linalg.norm(
                        self.target_y - pred_y, dim=1, keepdim=True
                    )

                loss = norm.square().mean()
                grad = torch.autograd.grad(loss, est_clean_pos)[0]
            return -(self.rho / (norm.sum() + 1e-12)) * grad.detach(), norm.detach()

    def _denoise_by_sde(
        species, pos, cell, cut, score_fn, likelihood_fn, ts, diffuser, save_traj
    ):
        ts = ts.to(pos.device).view(-1, 1)
        f, g, g2 = diffuser.f, diffuser.g, diffuser.g2
        traj = [pos.detach().cpu().clone()] if save_traj else None
        pos = pos.detach()

        for i, t in enumerate(ts[1:]):
            dt = ts[i + 1] - ts[i]
            eps = dt.abs().sqrt() * torch.randn_like(pos)
            with torch.no_grad():
                p_score = score_fn(species, pos, cell, t, cut)
            l_score, norm = likelihood_fn(species, pos, cell, t, cut)
            click.echo(
                f"    p={p_score.norm().item():.3f}  "
                f"l={l_score.norm().item():.3f}  "
                f"tgt={norm.sum().item():.4f}",
                err=True,
            )
            disp = (f(t) * pos - g2(t) * (p_score + l_score)) * dt + g(t) * eps
            pos = (pos + disp).detach()
            if save_traj:
                traj.append(pos.cpu().clone())

        return traj if save_traj else pos

    # --- collect init xyz files ---
    resolved_init_path = init_path.format(system=system)
    if os.path.isfile(resolved_init_path):
        xyz_files = [resolved_init_path]
    elif os.path.isdir(resolved_init_path):
        xyz_files = sorted(
            glob.glob(os.path.join(resolved_init_path, f"{system}_*.xyz"))
        )
    else:
        xyz_files = sorted(glob.glob(resolved_init_path))
    if not xyz_files:
        raise click.ClickException(f"No .xyz files found for {resolved_init_path}")

    for score_model in score_models:
        main_id = f"{system}-{score_model}"
        click.echo(f"\n=== Score model: {main_id} ===")

        resolved_data_path = score_data_path.format(
            score_model=score_model, system=system
        )
        resolved_model_path = score_model_path.format(
            score_model=score_model, system=system
        )

        datamodule = StructureSpecDataModule(
            data_dir=resolved_data_path,
            cutoff=cutoff,
            train_prior=True,
            k=0.8,
            train_size=0.9,
            scale_y=1.0,
            dup=128,
            batch_size=32,
            num_workers=8,
        )
        datamodule.setup()
        diffuser = datamodule.train_set.diffuser

        if not os.path.exists(resolved_model_path):
            raise click.ClickException(f"Checkpoint not found: {resolved_model_path}")
        score_net = LitScoreNet.load_from_checkpoint(
            resolved_model_path, map_location=device
        )
        score_net.eval()
        score_net.ema_model.to(device)
        score_net.ema_model.eval()

        ts_torch = torch.linspace(tmax, tmin, tstep, device=device)

        if ref_path:
            resolved_ref_path = ref_path.format(system=system)
            if os.path.isfile(resolved_ref_path):
                resolved_ref_path = os.path.dirname(resolved_ref_path)
            if not os.path.isdir(resolved_ref_path):
                raise click.ClickException(
                    f"--ref-path is not a valid directory: {resolved_ref_path}"
                )
            click.echo(f"  Ref dir: {resolved_ref_path}")

        for sample_tag in xyz_files:
            sub_id = os.path.basename(sample_tag).replace(".xyz", "")
            init_atoms = read(sample_tag, "-1")

            if exp_data:
                target_y = exp_target_y
            else:
                ref_file = os.path.join(resolved_ref_path, f"{sub_id}.xyz")
                if not os.path.exists(ref_file):
                    click.echo(f"  Warning: {ref_file} not found, skipping.")
                    continue
                ref_atoms = read(ref_file, "-1")
                if not (np.all(init_atoms.pbc) and np.all(ref_atoms.pbc)):
                    raise click.ClickException(
                        f"PBC must be True for both init and ref atoms ({sub_id})."
                    )
                if not np.allclose(
                    init_atoms.get_cell(), ref_atoms.get_cell(), atol=1e-5
                ):
                    raise click.ClickException(
                        f"Init and ref cells must match ({sub_id})."
                    )
                _, ref_species, ref_pos, ref_cell = initialize_atoms(ref_atoms)

                if guidance_type in ("pdf", "adf"):
                    target_y = guidance_model(
                        ref_pos.cpu(), ref_species.cpu(), ref_cell.cpu()
                    )[1].to(device)
                elif guidance_type in ("xrd", "nd"):
                    target_y = guidance_model(
                        ref_pos.to(device), ref_species.to(device)
                    )
                elif guidance_type in ("exafs", "xanes"):
                    ei_r, ev_r = periodic_radius_graph(
                        ref_pos.to(device), cutoff, ref_cell.to(device)
                    )
                    ea_r = torch.hstack([ev_r, ev_r.norm(dim=-1, keepdim=True)])
                    with torch.no_grad():
                        target_y = guidance_model(ref_species.to(device), ei_r, ea_r)

            click.echo(f"  {sub_id}")

            for rho in rho_list:
                rho_str = f"{int(rho)}" if rho == int(rho) else f"{rho}"
                run_tag = f"{guidance_type}_rho{rho_str}_tmax{tmax}_nsteps{tstep}"
                run_outdir = os.path.join(outdir, main_id, sub_id, run_tag)
                os.makedirs(run_outdir, exist_ok=True)

                likelihood_fn = LikelihoodScore(
                    score_net.ema_model,
                    guidance_model,
                    target_y,
                    rho,
                    diffuser,
                    guidance_type,
                    cutoff,
                )

                def prior_fn(sp, p, c, t, co, _sn=score_net, _df=diffuser):
                    return _prior_score(sp, p, c, t, co, _sn, _df)

                for i in range(n_runs):
                    species, pos, cell = _to_device(copy.deepcopy(init_atoms))
                    cell_np = cell.detach().cpu().numpy()

                    result = _denoise_by_sde(
                        species,
                        pos,
                        cell,
                        cutoff,
                        prior_fn,
                        likelihood_fn,
                        ts_torch,
                        diffuser,
                        save_traj,
                    )

                    if save_traj:
                        traj = []
                        for p in result:
                            a = ase.Atoms(
                                numbers=init_atoms.numbers,
                                positions=p.numpy(),
                                cell=cell_np,
                                pbc=[True] * 3,
                            )
                            a.wrap()
                            traj.append(a)
                        ase.io.write(f"{run_outdir}/{i:02}_traj.xyz", traj)
                        ase.io.write(f"{run_outdir}/{i:02}_final.xyz", traj[-1])
                    else:
                        final = ase.Atoms(
                            numbers=init_atoms.numbers,
                            positions=result.cpu().numpy(),
                            cell=cell_np,
                            pbc=[True] * 3,
                        )
                        final.wrap()
                        ase.io.write(f"{run_outdir}/{i:02}_final.xyz", final)

                    click.echo(f"    {sub_id} [{run_tag}] run #{i} done")

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()