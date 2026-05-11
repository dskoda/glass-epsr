#!/usr/bin/env python3
"""Joint Optuna HPO over unconditional + PDF-conditional + SA denoising.

Each trial draws ONE parameter vector and evaluates it in TWO modes:

    unconditional:  prior + Tersoff + corrector + SA tail
    conditional:    prior + Tersoff + corrector + PDF-likelihood + SA tail

Per mode we compute structural errors against a reference (PDF, ADF,
coordination) with ``compute_all_errors`` — same metric as ``glass compare``.

The per-mode scalar error is

    obj_mode = W_PDF * pdf_rmse + W_COORD * coord_emd + W_ADF * adf_rmse

and the total objective is a weighted sum of the two modes:

    objective = W_UNCOND * obj_uncond + W_COND * obj_cond

Each trial uses ``--n-inits`` init/ref pairs × ``--n-seeds`` RNG seeds,
averaging the per-mode scalar across all runs. Parameters searched:

    tstep (cat), tmin, tmax, t_rho (power-law schedule exponent),
    tersoff_lambda, tersoff_schedule, tersoff_t_gate,
    n_corr, corr_step_size, corr_t_gate,
    N_anneal, T0, anneal_lr,
    rho (guidance strength; log-scale)

Usage
-----

Install optuna (one-time)::

    pip install -e ".[hpo]"

Quick smoke (single device, 4 trials)::

    python scripts/hpo_unified.py research/test/ \\
        --ref-dir research/test/data/ \\
        --init-dir research/test/data/ \\
        --init-glob "Si_2.5_0[0-1].xyz" \\
        --n-trials 4 --n-seeds 1 --n-inits 1 \\
        --devices cuda:0 \\
        --study-name glass_unified_smoke \\
        --storage /tmp/glass_unified_smoke.db

Full run (4 GPUs, 200 trials, overnight)::

    python scripts/hpo_unified.py research/test/ \\
        --ref-dir research/test/data/ \\
        --init-dir research/test/data/ \\
        --init-glob "Si_2.5_*.xyz" \\
        --n-trials 200 --n-seeds 2 --n-inits 2 \\
        --n-jobs 4 --devices cuda:0,cuda:1,cuda:2,cuda:3 \\
        --study-name glass_unified_v1 \\
        --storage research/hpo/glass_unified_v1.db

Resume simply by re-running the same command (SQLite storage +
``load_if_exists=True``).

Replay best trial against a larger seed count::

    python scripts/hpo_unified.py research/test/ \\
        --ref-dir ... --init-dir ... --init-glob "Si_2.5_*.xyz" \\
        --study-name glass_unified_v1 \\
        --storage research/hpo/glass_unified_v1.db \\
        --replay-best --n-seeds 5 --n-inits 5
"""

from __future__ import annotations

import argparse
import copy
import glob
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
from glass.lit.modules.guidance import create_guidance_model
from glass.lit.modules.likelihood import LikelihoodScore
from glass.lit.modules.tersoff_guidance import (
    TersoffEnergyGuidance,
    TersoffSchedule,
)
from glass.metrics import compute_all_metrics
from glass.metrics.errors import compute_all_errors
from glass.utils.atoms_utils import (
    atoms_to_device,
    compute_prior_score,
    compute_target_from_reference,
)


# --------------------------------------------------------------------------
# Objective weights.
# PDF and coord are the primary figures of merit (equal), ADF is secondary.
# Unconditional and conditional modes are weighted equally — the winning
# param set has to perform in BOTH.
# --------------------------------------------------------------------------

W_PDF = 1.0
W_COORD = 1.0
W_ADF = 0.25

W_UNCOND = 0.5
W_COND = 0.5


def _mode_obj(errors: Dict[str, float]) -> float:
    return (
        W_PDF * float(errors["pdf_rmse"])
        + W_COORD * float(errors["coordination_emd"])
        + W_ADF * float(errors["adf_rmse"])
    )


# --------------------------------------------------------------------------
# Per-device cache. Optuna n_jobs uses threads, so one process shares GPUs.
# --------------------------------------------------------------------------

class DeviceCtx:
    __slots__ = (
        "device", "score_net", "diffuser", "guidance_model",
        "ref_atoms_by_id", "target_by_id", "init_atoms_by_id",
        "ref_metrics_by_id", "cutoff",
    )

    def __init__(self, device, score_net, diffuser, guidance_model, cutoff):
        self.device = device
        self.score_net = score_net
        self.diffuser = diffuser
        self.guidance_model = guidance_model
        self.cutoff = cutoff
        # Filled on first trial that touches each (init, ref) pair.
        self.ref_atoms_by_id: Dict[str, Atoms] = {}
        self.init_atoms_by_id: Dict[str, Atoms] = {}
        self.target_by_id: Dict[str, torch.Tensor] = {}
        self.ref_metrics_by_id: Dict[str, object] = {}


_device_cache: Dict[str, DeviceCtx] = {}
_device_cache_lock = threading.Lock()
_device_pool: List[str] = []
_device_counter = {"i": 0}
_device_counter_lock = threading.Lock()
_thread_local = threading.local()


def _acquire_device() -> str:
    with _device_counter_lock:
        dev = _device_pool[_device_counter["i"] % len(_device_pool)]
        _device_counter["i"] += 1
    return dev


def _get_device_ctx(
    experiment_path: str,
    init_paths: List[Path],
    ref_dir: Path,
) -> DeviceCtx:
    """Load (or fetch cached) score_net + diffuser + guidance model for this
    thread's device, plus per-id init/ref atoms, target features, and ref
    metrics (computed once)."""
    if getattr(_thread_local, "device", None) is None:
        _thread_local.device = _acquire_device()
    dev = _thread_local.device

    with _device_cache_lock:
        if dev in _device_cache:
            ctx = _device_cache[dev]
        else:
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

            guidance_model = create_guidance_model(
                guidance_type="pdf",
                device=device,
                cutoff=cfg.cutoff,
                bin_size=cfg.bin_size,
            )

            ctx = DeviceCtx(
                device=device,
                score_net=score_net,
                diffuser=diffuser,
                guidance_model=guidance_model,
                cutoff=cfg.cutoff,
            )
            _device_cache[dev] = ctx

        # Lazily fill the init/ref/target caches for any new init paths.
        for init_path in init_paths:
            sub_id = init_path.stem
            if sub_id in ctx.init_atoms_by_id:
                continue
            init_atoms = read(str(init_path))
            ref_path = ref_dir / f"{sub_id}.xyz"
            if not ref_path.exists():
                raise FileNotFoundError(
                    f"Reference for init {sub_id} not found at {ref_path}"
                )
            ref_atoms = read(str(ref_path))

            # PDF target (tensor) on this device — for LikelihoodScore.
            target = compute_target_from_reference(
                ref_atoms, ctx.guidance_model, "pdf", ctx.cutoff, ctx.device,
            )

            # Reference structural metrics — for compute_all_errors at the end.
            ref_metrics = compute_all_metrics(
                ref_atoms,
                include_dihedrals=False,
                include_sq=False,
                include_voronoi=False,
            )

            ctx.init_atoms_by_id[sub_id] = init_atoms
            ctx.ref_atoms_by_id[sub_id] = ref_atoms
            ctx.target_by_id[sub_id] = target
            ctx.ref_metrics_by_id[sub_id] = ref_metrics

        return ctx


# --------------------------------------------------------------------------
# One denoising run (mode-agnostic).
# --------------------------------------------------------------------------

def _run_single(
    params: Dict,
    sub_id: str,
    mode: str,  # "uncond" or "cond"
    ctx: DeviceCtx,
) -> Atoms:
    device = ctx.device
    init_atoms = ctx.init_atoms_by_id[sub_id]
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
        rho=params["t_rho"], device=device,
    )

    score_net = ctx.score_net
    diffuser = ctx.diffuser

    def prior_fn(sp, p, c, t, co, _sn=score_net, _df=diffuser):
        return compute_prior_score(sp, p, c, t, co, _sn, _df)

    likelihood_fn = None
    if mode == "cond":
        likelihood_fn = LikelihoodScore(
            score_net.ema_model,
            ctx.guidance_model,
            ctx.target_by_id[sub_id],
            float(params["rho"]),
            diffuser,
            "pdf",
            ctx.cutoff,
        )

    _, final_pos = denoise_by_sde(
        species=species,
        pos=pos,
        cell=cell,
        cutoff=ctx.cutoff,
        score_fn=prior_fn,
        likelihood_fn=likelihood_fn,
        ts=ts_torch,
        diffuser=diffuser,
        save_traj=False,
        tersoff_guidance=tersoff_guide,
        tersoff_schedule=tersoff_sched,
        n_corr=int(params["n_corr"]),
        corr_step_size=float(params["corr_step_size"]),
        corr_use_tersoff=True,
        corr_t_gate=float(params["corr_t_gate"]),
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
# Search space.
# --------------------------------------------------------------------------

def _sample_params(trial) -> Dict:
    return {
        # --- Time schedule ---
        "tstep": trial.suggest_categorical("tstep", [128, 256, 512]),
        "tmin": trial.suggest_float("tmin", 1e-4, 5e-2, log=True),
        "tmax": trial.suggest_float("tmax", 0.5, 1.0),
        "t_rho": trial.suggest_float("t_rho", 0.3, 2.0),
        # --- Tersoff guidance ---
        "tersoff_lambda": trial.suggest_float("tersoff_lambda", 0.0, 0.3),
        "tersoff_schedule": trial.suggest_categorical(
            "tersoff_schedule", ["constant", "linear", "sigmoid"]
        ),
        "tersoff_t_gate": trial.suggest_float("tersoff_t_gate", 0.1, 0.8),
        # --- Langevin corrector ---
        "n_corr": trial.suggest_categorical("n_corr", [0, 1, 2]),
        "corr_step_size": trial.suggest_float("corr_step_size", 0.05, 0.5),
        "corr_t_gate": trial.suggest_float("corr_t_gate", 0.2, 0.8),
        # --- Simulated-annealing tail ---
        "N_anneal": trial.suggest_categorical("N_anneal", [0, 50, 100, 200]),
        "T0": trial.suggest_float("T0", 1e-3, 1e-1, log=True),
        "anneal_lr": trial.suggest_float("anneal_lr", 1e-4, 1e-2, log=True),
        # --- Guidance strength (conditional mode only) ---
        "rho": trial.suggest_float("rho", 1e1, 3e3, log=True),
    }


# --------------------------------------------------------------------------
# Trial evaluation.
# --------------------------------------------------------------------------

def _evaluate_one(
    params: Dict,
    sub_id: str,
    mode: str,
    ctx: DeviceCtx,
    seed: int,
) -> Dict[str, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    atoms_out = _run_single(params, sub_id, mode, ctx)
    metrics = compute_all_metrics(
        atoms_out,
        include_dihedrals=False,
        include_sq=False,
        include_voronoi=False,
    )
    errors = compute_all_errors(ctx.ref_metrics_by_id[sub_id], metrics)
    return errors


def _build_objective(
    experiment_path: str,
    init_paths: List[Path],
    ref_dir: Path,
    n_inits: int,
    n_seeds: int,
):
    # Deterministic order so the same (sub_id, seed) pairs are evaluated
    # across replay runs.
    init_paths = sorted(init_paths)[: max(n_inits, 1)]

    def objective(trial):
        ctx = _get_device_ctx(experiment_path, init_paths, ref_dir)
        params = _sample_params(trial)

        mode_obj_samples: Dict[str, List[float]] = {"uncond": [], "cond": []}
        per_run_errors: List[Dict[str, object]] = []

        step = 0
        for seed_idx in range(n_seeds):
            for init_path in init_paths:
                sub_id = init_path.stem
                for mode in ("uncond", "cond"):
                    seed = abs(
                        1000 * int(trial.number) + 100 * seed_idx
                        + (0 if mode == "uncond" else 1)
                    ) + hash(sub_id) % 997
                    t0 = time.time()
                    try:
                        errors = _evaluate_one(
                            params, sub_id, mode, ctx, int(seed),
                        )
                    except Exception as e:
                        trial.set_user_attr("error", f"{type(e).__name__}: {e}")
                        return float("inf")
                    mode_obj_samples[mode].append(_mode_obj(errors))
                    per_run_errors.append(
                        {"sub_id": sub_id, "mode": mode, "seed": int(seed),
                         "time_s": time.time() - t0, **errors}
                    )
            # Report a running estimate per seed so the pruner has
            # something to chew on.
            running = (
                W_UNCOND * float(np.mean(mode_obj_samples["uncond"]))
                + W_COND * float(np.mean(mode_obj_samples["cond"]))
            )
            trial.report(running, step=step)
            step += 1
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()

        obj_uncond = float(np.mean(mode_obj_samples["uncond"]))
        obj_cond = float(np.mean(mode_obj_samples["cond"]))
        total = W_UNCOND * obj_uncond + W_COND * obj_cond

        # Pre-digest summary: per-metric means, per-mode objective, device.
        mean_err_uncond = {
            k: float(np.mean([e[k] for e in per_run_errors
                              if e["mode"] == "uncond" and isinstance(e.get(k), (int, float))]))
            for k in ("pdf_rmse", "coordination_emd", "adf_rmse")
        }
        mean_err_cond = {
            k: float(np.mean([e[k] for e in per_run_errors
                              if e["mode"] == "cond" and isinstance(e.get(k), (int, float))]))
            for k in ("pdf_rmse", "coordination_emd", "adf_rmse")
        }
        trial.set_user_attr("device", str(ctx.device))
        trial.set_user_attr("obj_uncond", obj_uncond)
        trial.set_user_attr("obj_cond", obj_cond)
        trial.set_user_attr("mean_errors_uncond", json.dumps(mean_err_uncond))
        trial.set_user_attr("mean_errors_cond", json.dumps(mean_err_cond))

        return total

    return objective


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------

def _parse_devices(s: str) -> List[str]:
    return [d.strip() for d in s.split(",") if d.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("experiment_path", type=str,
                        help="Experiment dir containing checkpoints/, config.yaml.")
    parser.add_argument("--ref-dir", type=str, required=True,
                        help="Directory of reference *.xyz files (one per init sub_id).")
    parser.add_argument("--init-dir", type=str, required=True,
                        help="Directory of init *.xyz files (will be matched to refs by filename stem).")
    parser.add_argument("--init-glob", type=str, default="*.xyz",
                        help="Glob pattern (relative to --init-dir) filtering inits.")
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=None,
                        help="Seconds.")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--n-seeds", type=int, default=2,
                        help="Random seeds per init per trial.")
    parser.add_argument("--n-inits", type=int, default=2,
                        help="Number of init structures per trial (sorted, deterministic).")
    parser.add_argument("--devices", type=str, default="cuda:0",
                        help="Comma-separated CUDA device list.")
    parser.add_argument("--study-name", type=str, default="glass_unified")
    parser.add_argument("--storage", type=str,
                        default="research/hpo/glass_unified.db")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replay-best", action="store_true",
                        help="Skip search; re-run the current best trial with the "
                             "provided --n-seeds and --n-inits.")
    args = parser.parse_args()

    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler

    global _device_pool
    _device_pool = _parse_devices(args.devices) or ["cpu"]
    print(f"Devices: {_device_pool}")

    init_dir = Path(args.init_dir).resolve()
    ref_dir = Path(args.ref_dir).resolve()
    init_paths = sorted(init_dir.glob(args.init_glob))
    if not init_paths:
        raise SystemExit(
            f"No init files matched {args.init_glob!r} under {init_dir}"
        )
    # Validate that every init has a matching ref.
    missing = [p.stem for p in init_paths
               if not (ref_dir / f"{p.stem}.xyz").exists()]
    if missing:
        raise SystemExit(
            f"{len(missing)} inits have no matching ref in {ref_dir}: "
            f"{missing[:3]}..."
        )
    print(f"Found {len(init_paths)} init/ref pairs in {init_dir}")
    print(f"Using first {min(args.n_inits, len(init_paths))} per trial.")

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

    objective = _build_objective(
        args.experiment_path, init_paths, ref_dir,
        n_inits=args.n_inits, n_seeds=args.n_seeds,
    )

    if args.replay_best:
        try:
            best = study.best_trial
        except ValueError:
            raise SystemExit("No completed trials in this study yet.")
        print("Replaying best trial:")
        print(json.dumps(best.params, indent=2))

        fake = optuna.trial.FixedTrial(best.params, number=-1)
        val = objective(fake)
        print(f"\nReplay objective (n_inits={args.n_inits}, "
              f"n_seeds={args.n_seeds}): {val:.6f}")
        ua = getattr(fake, "user_attrs", None) or getattr(fake, "_user_attrs", {})
        if "error" in ua:
            print(f"Error: {ua['error']}")
            return
        print(f"  obj_uncond: {ua.get('obj_uncond', float('nan')):.6f}")
        print(f"  obj_cond:   {ua.get('obj_cond', float('nan')):.6f}")
        if "mean_errors_uncond" in ua:
            print("  uncond errors:", ua["mean_errors_uncond"])
        if "mean_errors_cond" in ua:
            print("  cond errors:  ", ua["mean_errors_cond"])
        return

    print(f"\nRunning {args.n_trials} trials "
          f"(n_jobs={args.n_jobs}, n_seeds={args.n_seeds}, "
          f"n_inits={args.n_inits}, 2 modes => "
          f"{args.n_inits*args.n_seeds*2} runs/trial)")

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
    print("User attrs:")
    for k, v in best.user_attrs.items():
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
                "weights": {
                    "W_PDF": W_PDF, "W_COORD": W_COORD, "W_ADF": W_ADF,
                    "W_UNCOND": W_UNCOND, "W_COND": W_COND,
                },
            },
            f, indent=2,
        )
    print(f"\nBest params written to {best_path}")

    # Dump top-N summary.
    all_trials = [t for t in study.trials if t.value is not None]
    all_trials.sort(key=lambda t: t.value)
    print("\nTop 5 trials:")
    for t in all_trials[:5]:
        ua = t.user_attrs
        print(
            f"  #{t.number}: value={t.value:.4f}  "
            f"(uncond={ua.get('obj_uncond', float('nan')):.4f} "
            f"cond={ua.get('obj_cond', float('nan')):.4f})  "
            f"tstep={t.params.get('tstep')} n_corr={t.params.get('n_corr')} "
            f"N_anneal={t.params.get('N_anneal')} "
            f"rho={t.params.get('rho'):.1f} "
            f"tl={t.params.get('tersoff_lambda'):.3f}"
        )


if __name__ == "__main__":
    main()
