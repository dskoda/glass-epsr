#!/usr/bin/env python
"""RLPF fine-tuning script for the glass score-based diffusion model.

Loads a pre-trained LitScoreNet checkpoint and fine-tunes it using
RL with Physical Feedback (RLPF): a PPO-style policy gradient update
driven by a combined Tersoff energy + PDF similarity reward.

Usage
-----
python scripts/rlpf_finetune.py \\
    --experiment ./my_experiment \\
    --density 2.5 \\
    --ref-pdf ./data/average_pdf_2.5.json \\
    --init-glob "init_Si_2.5_*.xyz" \\
    --output-dir ./rlpf_output \\
    --n-updates 200 \\
    --device cuda:0
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import click
import numpy as np
import torch


@click.command()
@click.option(
    "--experiment",
    "experiment_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to the experiment directory (must contain config.yaml + checkpoints/).",
)
@click.option(
    "--density",
    type=float,
    default=2.5,
    show_default=True,
    help="Target density in g/cm³ (1.5, 2.5, or 3.5).",
)
@click.option(
    "--ref-pdf",
    "ref_pdf_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to JSON file with reference PDF (keys: 'r', 'g_r').",
)
@click.option(
    "--init-glob",
    "init_glob",
    type=str,
    default="*.xyz",
    show_default=True,
    help="Glob pattern relative to EXPERIMENT/inits/ to select initial structures.",
)
@click.option(
    "--n-updates",
    type=int,
    default=200,
    show_default=True,
    help="Total number of PPO update steps.",
)
@click.option(
    "--n-rollouts",
    type=int,
    default=8,
    show_default=True,
    help="Number of trajectories per PPO update.",
)
@click.option(
    "--w-energy",
    type=float,
    default=1.0,
    show_default=True,
    help="Weight for the Tersoff energy term in the reward.",
)
@click.option(
    "--w-pdf",
    type=float,
    default=1.0,
    show_default=True,
    help="Weight for the PDF RMSE term in the reward.",
)
@click.option(
    "--kl-beta",
    type=float,
    default=0.01,
    show_default=True,
    help="KL divergence penalty coefficient.",
)
@click.option(
    "--kl-warmup",
    type=int,
    default=10,
    show_default=True,
    help="Number of updates before enabling KL penalty.",
)
@click.option(
    "--lr",
    type=float,
    default=1e-5,
    show_default=True,
    help="Learning rate for Adam optimiser.",
)
@click.option(
    "--device",
    type=str,
    default="cuda:0",
    show_default=True,
    help="Torch device (e.g. 'cuda:0', 'cpu').",
)
@click.option(
    "--checkpoint-every",
    type=int,
    default=25,
    show_default=True,
    help="Save a checkpoint every N updates.",
)
@click.option(
    "--output-dir",
    "output_dir",
    required=True,
    type=click.Path(),
    help="Directory to save checkpoints, metrics, and config.",
)
@click.option(
    "--subsample-steps",
    type=int,
    default=4,
    show_default=True,
    help="Store every Nth SDE step for PPO update (reduces memory).",
)
@click.option(
    "--tstep",
    type=int,
    default=64,
    show_default=True,
    help="Number of SDE time steps per rollout (64 is faster than 512 for fine-tuning).",
)
@click.option(
    "--checkpoint",
    type=str,
    default="best",
    show_default=True,
    help="Checkpoint to load: 'best', 'last', or a specific filename.",
)
@click.option(
    "--guidance-rho",
    "guidance_rho",
    type=float,
    default=None,
    show_default=True,
    help=(
        "PDF guidance strength during rollout collection (LikelihoodScore rho). "
        "If not set, rollouts use only the score prior (Phase H-A behaviour). "
        "Set to 737.0 to match the v3_ood HPO default."
    ),
)
def main(
    experiment_path,
    density,
    ref_pdf_path,
    init_glob,
    n_updates,
    n_rollouts,
    w_energy,
    w_pdf,
    kl_beta,
    kl_warmup,
    lr,
    device,
    checkpoint_every,
    output_dir,
    subsample_steps,
    tstep,
    checkpoint,
    guidance_rho,
):
    """Fine-tune a glass score model using RLPF."""
    import ase.io

    from glass.diffusion import VarianceExplodingDiffuser
    from glass.diffusion.rewards import TersoffPDFReward
    from glass.diffusion.rlpf import RLPFConfig, RLPFTrainer
    from glass.diffusion.schedules import power_law_ts
    from glass.experiment import Experiment
    from glass.lit.modules.prior import LitScoreNet
    from glass.potentials.tersoff.ase_calc import silicon_calculator
    from glass.utils.atoms_utils import atoms_to_device

    # ------------------------------------------------------------------
    # Setup output directory
    # ------------------------------------------------------------------
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_path / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    # Save CLI args as config.
    cli_config = {
        "experiment_path": str(experiment_path),
        "density": density,
        "ref_pdf_path": str(ref_pdf_path),
        "init_glob": init_glob,
        "n_updates": n_updates,
        "n_rollouts": n_rollouts,
        "w_energy": w_energy,
        "w_pdf": w_pdf,
        "kl_beta": kl_beta,
        "kl_warmup": kl_warmup,
        "lr": lr,
        "device": device,
        "checkpoint_every": checkpoint_every,
        "output_dir": str(output_dir),
        "subsample_steps": subsample_steps,
        "tstep": tstep,
        "checkpoint": checkpoint,
        "guidance_rho": guidance_rho,
    }
    with open(out_path / "config.json", "w") as f:
        json.dump(cli_config, f, indent=2)

    # ------------------------------------------------------------------
    # Load experiment config and checkpoint
    # ------------------------------------------------------------------
    exp = Experiment(experiment_path)
    config = exp.load_config()
    ckpt_path = exp.find_checkpoint(checkpoint)
    print(f"[RLPF] Loading checkpoint: {ckpt_path}", flush=True)

    dev = torch.device(device)
    score_net = LitScoreNet.load_from_checkpoint(str(ckpt_path), map_location=dev)
    score_net.to(dev)
    score_net.model.train()
    score_net.ema_model.eval()

    # ------------------------------------------------------------------
    # Build diffuser
    # ------------------------------------------------------------------
    diffuser = VarianceExplodingDiffuser(k=config.k)

    # ------------------------------------------------------------------
    # Load reference PDF
    # ------------------------------------------------------------------
    with open(ref_pdf_path, "r") as f:
        ref_data = json.load(f)
    target_r = np.asarray(ref_data["r"], dtype=np.float64)
    target_g_r = np.asarray(ref_data["g_r"], dtype=np.float64)
    print(
        f"[RLPF] Loaded reference PDF: r in [{target_r.min():.2f}, {target_r.max():.2f}] Å",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Build Tersoff potential
    # ------------------------------------------------------------------
    si_calc = silicon_calculator()
    tersoff_calc = si_calc._torch_calc  # TorchTersoff instance

    # ------------------------------------------------------------------
    # Build reward function
    # ------------------------------------------------------------------
    reward_fn = TersoffPDFReward(
        tersoff_calc=tersoff_calc,
        target_g_r=target_g_r,
        target_r=target_r,
        w_energy=w_energy,
        w_pdf=w_pdf,
        device=device,
    )

    # ------------------------------------------------------------------
    # Build PDF guidance for rollout collection (Phase H-B when guidance_rho set)
    # ------------------------------------------------------------------
    likelihood_fn = None
    if guidance_rho is not None:
        from glass.lit.modules.guidance import create_guidance_model
        from glass.lit.modules.likelihood import LikelihoodScore

        guidance_model = create_guidance_model(
            guidance_type="pdf",
            device=dev,
            cutoff=config.cutoff,
            bin_size=config.bin_size,
        )

        # Build target_y on the DifferentiableRDF bin grid.
        bin_edges = np.linspace(0, config.cutoff, config.bin_size + 1)
        x_grid = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        y_interp = np.interp(x_grid, target_r, target_g_r)
        target_y = torch.from_numpy(y_interp).float().unsqueeze(0).to(dev)

        likelihood_fn = LikelihoodScore(
            score_net=score_net.ema_model,
            guidance_model=guidance_model,
            target_y=target_y,
            rho=guidance_rho,
            diffuser=diffuser,
            guidance_type="pdf",
            cutoff=config.cutoff,
        )
        print(
            f"[RLPF] PDF guidance during rollouts: rho={guidance_rho}",
            flush=True,
        )
    else:
        print("[RLPF] No PDF guidance during rollouts (prior only).", flush=True)

    # ------------------------------------------------------------------
    # Build RLPF trainer
    # ------------------------------------------------------------------
    rlpf_cfg = RLPFConfig(
        kl_beta=kl_beta,
        kl_warmup_updates=kl_warmup,
        lr=lr,
        n_rollouts_per_update=n_rollouts,
        subsample_steps=subsample_steps,
    )
    trainer = RLPFTrainer(score_net, diffuser, reward_fn, rlpf_cfg, likelihood_fn=likelihood_fn)

    # ------------------------------------------------------------------
    # Load initial structures
    # ------------------------------------------------------------------
    inits_dir = Path(experiment_path) / "inits"
    init_paths = sorted(glob.glob(str(inits_dir / init_glob)))
    if len(init_paths) == 0:
        raise click.ClickException(
            f"No init structures found matching: {inits_dir / init_glob}"
        )
    print(f"[RLPF] Found {len(init_paths)} init structures.", flush=True)

    # Build time schedule
    ts = power_law_ts(config.tmin, config.tmax, tstep).to(dev)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    metrics_log = []
    print(
        f"[RLPF] Starting {n_updates} updates "
        f"({n_rollouts} rollouts/update, {tstep} SDE steps/rollout)",
        flush=True,
    )

    for update_idx in range(n_updates):
        # Cycle through init structures.
        init_path = init_paths[update_idx % len(init_paths)]
        atoms = ase.io.read(init_path)
        species, pos, cell = atoms_to_device(atoms, dev)

        # Add noise at tmax level to get a starting point.
        sigma0 = diffuser.sigma(ts[0])
        torch.manual_seed(update_idx)
        pos_noisy = pos + sigma0 * torch.randn_like(pos)

        # Collect rollouts.
        trajs = trainer.collect_rollouts(
            species, pos_noisy, cell,
            cutoff=config.cutoff,
            ts=ts,
            n_rollouts=n_rollouts,
        )

        # PPO update.
        metrics = trainer.ppo_update(trajs, species, cell, config.cutoff, ts)
        metrics["update"] = update_idx
        metrics["init"] = str(init_path)
        metrics_log.append(metrics)

        if (update_idx + 1) % 10 == 0 or update_idx == 0:
            print(
                f"[RLPF] update {update_idx+1:4d}/{n_updates} "
                f"| reward={metrics['reward_mean']:.4f} "
                f"| ppo={metrics['ppo_loss']:.4f} "
                f"| kl={metrics['kl_loss']:.4f}",
                flush=True,
            )

        # Checkpoint.
        if (update_idx + 1) % checkpoint_every == 0:
            ckpt_file = ckpt_dir / f"model_update_{update_idx+1:05d}.pt"
            torch.save(score_net.model.state_dict(), str(ckpt_file))
            print(f"[RLPF] Saved checkpoint: {ckpt_file}", flush=True)

    # Save final checkpoint.
    final_ckpt = ckpt_dir / "model_final.pt"
    torch.save(score_net.model.state_dict(), str(final_ckpt))
    print(f"[RLPF] Saved final checkpoint: {final_ckpt}", flush=True)

    # Save metrics log.
    metrics_path = out_path / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_log, f, indent=2)
    print(f"[RLPF] Saved metrics: {metrics_path}", flush=True)

    # ------------------------------------------------------------------
    # Quick evaluation: generate 5 structures using fine-tuned model
    # ------------------------------------------------------------------
    print("[RLPF] Running quick eval (5 structures from first init)…", flush=True)
    score_net.model.eval()
    eval_rewards = []
    structures_dir = out_path / "structures"
    structures_dir.mkdir(exist_ok=True)

    atoms_eval = ase.io.read(init_paths[0])
    species_eval, pos_eval, cell_eval = atoms_to_device(atoms_eval, dev)
    cell_np = cell_eval.detach().cpu().numpy()
    n_atoms = pos_eval.shape[0]

    for seed in range(5):
        torch.manual_seed(seed + 9999)
        sigma0 = diffuser.sigma(ts[0])
        pos_noisy = pos_eval + sigma0 * torch.randn_like(pos_eval)

        # Single rollout with fine-tuned model (EMA not updated in RLPF loop).
        pos_cur = pos_noisy.clone()
        with torch.no_grad():
            from glass.nn import periodic_radius_graph

            f, g, g2 = diffuser.f, diffuser.g, diffuser.g2
            for i in range(len(ts) - 1):
                t_val = ts[i]
                t_next = ts[i + 1]
                dt = float((t_next - t_val).item())
                t_tensor = t_val.reshape(1)

                edge_index, edge_vec = periodic_radius_graph(
                    pos_cur, config.cutoff, cell_eval
                )
                edge_attr = torch.hstack(
                    [edge_vec, edge_vec.norm(dim=-1, keepdim=True)]
                )
                score = score_net.model(
                    species_eval,
                    edge_index,
                    edge_attr,
                    t_tensor,
                    diffuser.sigma(t_tensor),
                )
                g2_t = float(g2(t_val))
                g_t = float(g(t_val))
                mu = pos_cur - g2_t * score * dt
                noise = abs(dt) ** 0.5 * torch.randn_like(pos_cur)
                pos_cur = (mu + g_t * noise).detach()

        r, info = reward_fn(pos_cur, cell_eval, species_eval)
        eval_rewards.append(r)
        print(
            f"  eval seed {seed}: reward={r:.4f} "
            f"(energy/atom={info['energy']:.4f} eV, pdf_rmse={info['pdf']:.4f})",
            flush=True,
        )

        # Save generated structure as xyz.
        import ase
        pos_np = pos_cur.detach().cpu().numpy().astype(np.float64)
        gen_atoms = ase.Atoms(
            symbols=["Si"] * n_atoms,
            positions=pos_np,
            cell=cell_np,
            pbc=True,
        )
        xyz_path = structures_dir / f"eval_seed_{seed:03d}.xyz"
        ase.io.write(str(xyz_path), gen_atoms)

    eval_summary = {
        "reward_mean": float(np.mean(eval_rewards)),
        "reward_std": float(np.std(eval_rewards)),
        "rewards": eval_rewards,
    }
    eval_path = out_path / "eval_summary.json"
    with open(eval_path, "w") as f:
        json.dump(eval_summary, f, indent=2)
    print(
        f"[RLPF] Eval done. Mean reward: {eval_summary['reward_mean']:.4f} "
        f"± {eval_summary['reward_std']:.4f}",
        flush=True,
    )
    print(f"[RLPF] Results written to: {out_path}", flush=True)


if __name__ == "__main__":
    main()
