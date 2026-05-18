#!/usr/bin/env python3
"""Optuna-based hyperparameter search for diffusion-sampling knobs.

The study minimises a weighted structural error against a reference:

    obj = 1.0 * coordination_emd + 0.5 * pdf_rmse + 0.25 * adf_rmse

Search space covers: time-step count, tmin, tmax, power-law schedule ``rho``,
Tersoff guidance (lambda, schedule, t_gate), Langevin corrector (n_corr,
step_size), and simulated-annealing tail (N_anneal, T0, lr).

Usage:
    pip install -e ".[hpo]"
    python scripts/hpo_generate.py <experiment_path> \\
        --ref-metrics research/ref-metrics.json \\
        --init-xyz research/tersoff_eval/inits_match/init_random_Si_25.xyz \\
        --n-trials 200 --n-jobs 4 --n-seeds 3 \\
        --devices cuda:0,cuda:1,cuda:2,cuda:3 \\
        --study-name glass_coord_v1 \\
        --storage research/hpo/glass_coord_v1.db

Resume by re-running the same command; the SQLite storage + ``load_if_exists``
picks up where it left off. Use ``--replay-best`` to re-run the best trial on
a larger seed count.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from ase import Atoms
from ase.io import read

from glass.diffusion.annealing import make_anneal_fn
from glass.diffusion.sampling import denoise_by_sde
from glass.diffusion.schedules import power_law_ts
from glass.experiment import Experiment
from glass.lit.datamodules import StructureSpecDataModule
from glass.lit.modules import LitScoreNet
from glass.lit.modules.tersoff_guidance import (
    TersoffEnergyGuidance,
    TersoffSchedule,
)
from glass.metrics import compute_all_metrics
from glass.metrics.errors import compute_all_errors
from glass.metrics.utils import load_metrics_from_json
from glass.utils.atoms_utils import atoms_to_device, compute_prior_score


# --------------------------------------------------------------------------
# Per-device cache (Optuna n_jobs uses threads, so one process shares GPUs).
# --------------------------------------------------------------------------

class DeviceCtx:
    __slots__ = ("device", "score_net", "diffuser")

    def __init__(self, device, score_net, diffuser):
        self.device = device
        self.score_net = score_net
        self.diffuser = diffuser


_device_cache: Dict[str, DeviceCtx] = {}
_device_cache_lock = threading.Lock()
_device_pool: List[str] = []
_device_counter = {"i": 0}
_device_counter_lock = threading.Lock()


def _acquire_device() -> str:
    """Round-robin device assignment, one device per worker thread."""
    with _device_counter_lock:
        dev = _device_pool[_device_counter["i"] % len(_device_pool)]
        _device_counter["i"] += 1
    return dev


_thread_local = threading.local()


def _get_device_ctx(experiment_path: str) -> DeviceCtx:
    """Load (or fetch cached) score_net + diffuser for this thread's device."""
    if getattr(_thread_local, "device", None) is None:
        _thread_local.device = _acquire_device()
    dev = _thread_local.device

    with _device_cache_lock:
        if dev in _device_cache:
            return _device_cache[dev]

        experiment = Experiment(experiment_path)
        cfg = experiment.load_config()
        ckpt = experiment.find_checkpoint("best")
        device = torch.device(dev if torch.cuda.is_available() else "cpu")

        score_net = LitScoreNet.load_from_checkpoint(ckpt, map_location=device)
        score_net.eval()
        score_net.ema_model.to(device)
        score_net.ema_model.eval()

        dm = StructureSpecDataModule(
            data_dir=experiment.get_data_dir_for_datamodule(),
            cutoff=cfg.cutoff,
            train_prior=True,
            k=cfg.k,
            train_size=0.9,
            scale_y=1.0,
            dup=128,
            batch_size=32,
            num_workers=0,
        )
        dm.setup()
        diffuser = dm.train_set.diffuser

        ctx = DeviceCtx(device=device, score_net=score_net, diffuser=diffuser)
        _device_cache[dev] = ctx
        return ctx


# --------------------------------------------------------------------------
# One denoising run.
# --------------------------------------------------------------------------

def _run_single(
    params: Dict,
    init_atoms: Atoms,
    cfg_cutoff: float,
    ctx: DeviceCtx,
) -> Atoms:
    device = ctx.device
    species, pos, cell = atoms_to_device(copy.deepcopy(init_atoms), device)
    cell_np = cell.detach().cpu().numpy()

    tersoff_guide = TersoffEnergyGuidance(clamp_norm=10.0)
    tersoff_sched = TersoffSchedule(
        schedule=params["tersoff_schedule"],
        lambda_0=params["tersoff_lambda"],
        tmax=params["tmax"],
        t_gate=params["tersoff_t_gate"],
    )

    anneal_fn = None
    if params["N_anneal"] > 0:
        anneal_fn = make_anneal_fn(
            tersoff_guidance=tersoff_guide,
            n_steps=int(params["N_anneal"]),
            T0=float(params["T0"]),
            T_end=max(float(params["T0"]) * 1e-4, 1e-8),
            lr=float(params["anneal_lr"]),
            lr_clamp=0.2,
        )

    ts_torch = power_law_ts(
        params["tmin"], params["tmax"], int(params["tstep"]),
        rho=params["rho"], device=device,
    )

    score_net = ctx.score_net
    diffuser = ctx.diffuser

    def prior_fn(sp, p, c, t, co, _sn=score_net, _df=diffuser):
        return compute_prior_score(sp, p, c, t, co, _sn, _df)

    _, final_pos = denoise_by_sde(
        species=species,
        pos=pos,
        cell=cell,
        cutoff=cfg_cutoff,
        score_fn=prior_fn,
        likelihood_fn=None,
        ts=ts_torch,
        diffuser=diffuser,
        save_traj=False,
        tersoff_guidance=tersoff_guide,
        tersoff_schedule=tersoff_sched,
        n_corr=int(params["n_corr"]),
        corr_step_size=float(params["corr_step_size"]),
        corr_use_tersoff=True,
        corr_t_gate=0.6,
        anneal_fn=anneal_fn,
    )

    atoms_out = Atoms(
        numbers=init_atoms.numbers,
        positions=final_pos.cpu().numpy(),
        cell=cell_np,
        pbc=[True, True, True],
    )
    atoms_out.wrap()
    return atoms_out


# --------------------------------------------------------------------------
# Search space + objective.
# --------------------------------------------------------------------------

W_COORD, W_PDF, W_ADF = 1.0, 0.5, 0.25


def _objective_value(errors: Dict[str, float]) -> float:
    return (
        W_COORD * float(errors["coordination_emd"])
        + W_PDF * float(errors["pdf_rmse"])
        + W_ADF * float(errors["adf_rmse"])
    )


def _sample_params(trial) -> Dict:
    return {
        "tstep": trial.suggest_categorical("tstep", [64, 128, 256, 512]),
        "tmin": trial.suggest_float("tmin", 1e-4, 5e-2, log=True),
        "tmax": trial.suggest_float("tmax", 0.5, 1.0),
        "rho": trial.suggest_float("rho", 0.5, 3.0),
        "tersoff_lambda": trial.suggest_float("tersoff_lambda", 0.0, 0.3),
        "tersoff_schedule": trial.suggest_categorical(
            "tersoff_schedule", ["constant", "linear", "sigmoid"]
        ),
        "tersoff_t_gate": trial.suggest_float("tersoff_t_gate", 0.1, 0.8),
        "n_corr": trial.suggest_categorical("n_corr", [0, 1, 2, 3]),
        "corr_step_size": trial.suggest_float("corr_step_size", 0.05, 0.5),
        "N_anneal": trial.suggest_categorical("N_anneal", [0, 50, 100, 200]),
        "T0": trial.suggest_float("T0", 1e-3, 1e-1, log=True),
        "anneal_lr": trial.suggest_float("anneal_lr", 1e-4, 1e-2, log=True),
    }


def _evaluate(
    params: Dict,
    init_atoms: Atoms,
    ref_metrics,
    ctx: DeviceCtx,
    cfg_cutoff: float,
    seed: int,
) -> Tuple[float, Dict[str, float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    atoms_out = _run_single(params, init_atoms, cfg_cutoff, ctx)
    metrics = compute_all_metrics(
        atoms_out,
        include_dihedrals=False,
        include_sq=False,
        include_voronoi=False,
    )
    errors = compute_all_errors(ref_metrics, metrics)
    return _objective_value(errors), errors


def _build_objective(
    experiment_path: str,
    ref_metrics_path: str,
    init_xyz: str,
    n_seeds: int,
    cutoff: float,
):
    ref_metrics = load_metrics_from_json(ref_metrics_path)
    init_atoms = read(init_xyz)

    def objective(trial):
        ctx = _get_device_ctx(experiment_path)
        params = _sample_params(trial)

        scores: List[float] = []
        all_err: List[Dict[str, float]] = []
        for seed_idx in range(n_seeds):
            seed = abs(1000 * int(trial.number) + seed_idx)
            t0 = time.time()
            try:
                score, err = _evaluate(
                    params, init_atoms, ref_metrics, ctx, cutoff, seed,
                )
            except Exception as e:
                trial.set_user_attr("error", f"{type(e).__name__}: {e}")
                return float("inf")
            scores.append(score)
            all_err.append(err)
            trial.set_user_attr(f"seed{seed_idx}_time_s", time.time() - t0)
            trial.report(float(np.mean(scores)), step=seed_idx)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()

        mean_err = {k: float(np.mean([e[k] for e in all_err])) for k in all_err[0]}
        trial.set_user_attr("device", str(ctx.device))
        trial.set_user_attr("mean_errors", json.dumps(mean_err))
        return float(np.mean(scores))

    return objective


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------

def _parse_devices(s: str) -> List[str]:
    return [d.strip() for d in s.split(",") if d.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_path", type=str)
    parser.add_argument("--ref-metrics", type=str, required=True)
    parser.add_argument("--init-xyz", type=str, required=True)
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=None,
                        help="Seconds.")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--devices", type=str, default="cuda:0",
                        help="Comma-separated CUDA device list.")
    parser.add_argument("--study-name", type=str, default="glass_hpo")
    parser.add_argument("--storage", type=str,
                        default="research/hpo/glass_hpo.db")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replay-best", action="store_true",
                        help="Skip search; just re-run the current best trial "
                             "at --n-seeds and report the aggregated metrics.")
    args = parser.parse_args()

    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler

    global _device_pool
    _device_pool = _parse_devices(args.devices) or ["cpu"]
    print(f"Devices: {_device_pool}")

    storage_path = Path(args.storage)
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{storage_path}"

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_url,
        direction="minimize",
        sampler=TPESampler(seed=args.seed),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=0, interval_steps=1),
        load_if_exists=True,
    )

    experiment = Experiment(args.experiment_path)
    cfg = experiment.load_config()
    objective = _build_objective(
        args.experiment_path, args.ref_metrics, args.init_xyz,
        n_seeds=args.n_seeds, cutoff=cfg.cutoff,
    )

    if args.replay_best:
        try:
            best = study.best_trial
        except ValueError:
            raise SystemExit("No completed trials in this study yet.")
        print("Replaying best trial:")
        print(json.dumps(best.params, indent=2))

        # optuna.trial.FixedTrial handles suggest_* exactly like a real Trial
        # (including keyword args like ``log=True`` and ``step``), so we reuse
        # it rather than hand-rolling a fake.
        fake = optuna.trial.FixedTrial(best.params, number=-1)
        val = objective(fake)
        print(f"\nReplay objective (n_seeds={args.n_seeds}): {val:.6f}")
        # FixedTrial records user_attrs in a private dict; expose safely.
        ua = getattr(fake, "user_attrs", None) or getattr(fake, "_user_attrs", {})
        if ua:
            if "error" in ua:
                print(f"Error: {ua['error']}")
            times = [f"{k}={v:.1f}s" for k, v in ua.items() if "time" in k]
            if times:
                print("Per-seed times: " + ", ".join(times))
            if "mean_errors" in ua:
                print("Mean errors:")
                print(json.dumps(json.loads(ua["mean_errors"]), indent=2))
        return

    print(f"\nRunning {args.n_trials} trials (n_jobs={args.n_jobs}, "
          f"n_seeds={args.n_seeds}) against {args.ref_metrics}")
    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
        n_jobs=args.n_jobs,
        gc_after_trial=True,
    )

    try:
        best = study.best_trial
    except ValueError:
        print("No successful trials.")
        return

    print("\n=== BEST TRIAL ===")
    print(f"Value: {best.value:.6f}")
    print("Params:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    out_dir = storage_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / f"{args.study_name}_best.json"
    with open(best_path, "w") as f:
        json.dump(
            {
                "value": float(best.value),
                "params": best.params,
                "user_attrs": best.user_attrs,
            },
            f, indent=2,
        )
    print(f"\nBest params written to {best_path}")

    # Dump top-N summary
    all_trials = [t for t in study.trials if t.value is not None]
    all_trials.sort(key=lambda t: t.value)
    print("\nTop 5 trials:")
    for i, t in enumerate(all_trials[:5]):
        print(f"  #{t.number}: value={t.value:.4f}  n_corr={t.params.get('n_corr')}  "
              f"N_anneal={t.params.get('N_anneal')}  "
              f"rho={t.params.get('rho'):.2f}  tstep={t.params.get('tstep')}")


if __name__ == "__main__":
    main()
