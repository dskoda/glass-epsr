#!/usr/bin/env python3
"""Physics-weighted HPO v7 — coordination-focused with lambda_prior.

Forked from ``hpo_phys_v6.py``. v7 targets two things that v6 left as
unsolved at ρ=1.5:

1. **Coordination quality is the primary objective.** CN ≤ 3 is very
   heavily penalised; CN ≤ 2 is catastrophic. The Tersoff energy/forces,
   rings, and ADF terms that appeared in v5/v6 are removed so TPE can focus
   the search on coordination.

2. **lambda_prior** is a new tunable that scales the prior score network
   output at every SDE step. lambda_prior = 1.0 is the standard case;
   0.0 turns the prior off entirely, leaving only guidance terms. Values
   between 0 and 1 interpolate between pure-guidance and standard denoising.

3. **Coordination-number guidance** is now searchable via ``coord_lambda``
   and ``coord_w_low``. These activate the differentiable soft-counting loss
   (low-hinge + target pull) that is already wired into ``denoise_by_sde``.

4. **Wider Tersoff and rho ranges** — v6 fixed rho at 416; v7 re-opens the
   full PDF guidance strength range and allows stronger Tersoff.

Objective — 4-term, coordination-heavy:
    W_PDF*pdf_rmse
    + W_COORD*coord_emd
    + W_UC3*log10(1 + frac_le3/EPS3)
    + W_UC2*log10(1 + frac_le2/EPS2)

Search space (12 parameters):
    tmax            ∈ [0.4, 1.0]
    tmin            ∈ [1e-4, 2e-2]  (log-uniform)
    tstep           ∈ {128, 256, 512}
    t_rho           ∈ [0.5, 2.0]
    rho             ∈ [50, 2000]    (log-uniform, PDF guidance)
    tersoff_lambda  ∈ [0.0, 2.0]
    tersoff_schedule ∈ {constant, linear, sigmoid}
    tersoff_t_gate  ∈ [0.1, 0.9]
    n_restart       ∈ {1, 2, 3, 4}
    lambda_prior    ∈ [0.0, 5.0]    (NEW)
    coord_lambda    ∈ [0.0, 100.0]  (NEW; 0 disables coord guidance)
    coord_w_low     ∈ [0.0, 50.0]   (NEW; low-CN hinge weight)

Fixed (from v6 best #39):
    n_corr=2, corr_step_size=0.44, corr_t_gate=0.464
    N_anneal=0, T0=0.01, anneal_lr=1e-3

Usage
-----

Quick smoke (single GPU, 2 trials)::

    python scripts/hpo_phys_v7.py research/density_extrapolation/experiment/ \\
        --ref-dir research/03-paper-silicon/results/generated/cond/density_1.5/reference \\
        --init-dir research/density_extrapolation/experiment/inits \\
        --init-glob "init_Si_1.5_*.xyz" \\
        --n-trials 2 --n-seeds 1 --n-inits 1 \\
        --n-jobs 1 --devices cuda:0 \\
        --study-name glass_phys_v7_smoke \\
        --storage /tmp/glass_phys_v7_smoke.db

Full run (4 GPUs, 1000 trials)::

    python scripts/hpo_phys_v7.py research/density_extrapolation/experiment/ \\
        --ref-dir research/03-paper-silicon/results/generated/cond/density_1.5/reference \\
        --init-dir research/density_extrapolation/experiment/inits \\
        --init-glob "init_Si_1.5_*.xyz" \\
        --n-trials 1000 --n-seeds 1 --n-inits 5 \\
        --n-jobs 4 --devices cuda:0,cuda:1,cuda:2,cuda:3 \\
        --study-name glass_phys_v7_coord \\
        --storage research/hpo/glass_phys_v7_coord.db
"""

from __future__ import annotations

import argparse
import copy
import json
import math
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
from glass.lit.modules.coord_guidance import (
    CoordinationGuidance,
    CoordinationLoss,
    CoordinationSchedule,
    DifferentiableCoordinationNumber,
)
from glass.lit.modules.guidance import create_guidance_model
from glass.lit.modules.likelihood import LikelihoodScore
from glass.lit.modules.tersoff_guidance import (
    TersoffEnergyGuidance,
    TersoffSchedule,
)
from glass.metrics import compute_all_metrics
from glass.metrics.errors import compute_all_errors
from glass.utils.atoms import (
    atoms_to_device,
    compute_prior_score,
    compute_target_from_reference,
)


# --------------------------------------------------------------------------
# Objective weights — coordination-focused, very heavy undercoord penalties.
# Tersoff energy/forces, rings, and ADF are excluded to let TPE concentrate
# the search on coordination quality and PDF agreement.
# --------------------------------------------------------------------------

W_PDF = 1.0
W_COORD = 5.0
W_UC3 = 50.0      # log-scale: CN ≤ 3 very heavily penalised
W_UC2 = 80.0      # log-scale: CN ≤ 2 (dangling bonds) catastrophic
EPS3 = 0.002      # 0.2 % saturation for W_UC3 term
EPS2 = 0.0005     # 0.05 % saturation for W_UC2 term


# Fixed parameters from v6 best trial #39.
_FIXED_PARAMS: Dict = {
    "n_corr": 2,
    "corr_step_size": 0.44,
    "corr_t_gate": 0.464,
    "N_anneal": 0,
    "T0": 0.01,
    "anneal_lr": 1e-3,
}


def _uc_penalty(frac: float, weight: float, eps: float) -> float:
    """log10(1 + frac/eps) * weight. Zero at frac=0, ~weight at frac=eps."""
    return weight * math.log10(1.0 + max(float(frac), 0.0) / eps)


def _phys_obj(errors: Dict[str, float]) -> float:
    """Per-structure coordination-focused objective."""
    return (
        W_PDF * float(errors["pdf_rmse"])
        + W_COORD * float(errors["coordination_emd"])
        + _uc_penalty(errors["undercoord_frac_le3"], W_UC3, EPS3)
        + _uc_penalty(errors["undercoord_frac_le2"], W_UC2, EPS2)
    )


# --------------------------------------------------------------------------
# Per-device cache. Optuna n_jobs uses threads, one process shares GPUs.
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
            target = compute_target_from_reference(
                ref_atoms, ctx.guidance_model, "pdf", ctx.cutoff, ctx.device,
            )
            ref_metrics = compute_all_metrics(
                ref_atoms,
                include_dihedrals=False,
                include_sq=False,
                include_voronoi=False,
                include_rings=False,
                include_tersoff=False,
            )
            ctx.init_atoms_by_id[sub_id] = init_atoms
            ctx.ref_atoms_by_id[sub_id] = ref_atoms
            ctx.target_by_id[sub_id] = target
            ctx.ref_metrics_by_id[sub_id] = ref_metrics

        return ctx


# --------------------------------------------------------------------------
# One denoising run.
# --------------------------------------------------------------------------

def _run_single(
    params: Dict,
    sub_id: str,
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

    lp = float(params.get("lambda_prior", 1.0))
    def prior_fn(sp, p, c, t, co, _sn=score_net, _df=diffuser, _lp=lp):
        return _lp * compute_prior_score(sp, p, c, t, co, _sn, _df)

    likelihood_fn = LikelihoodScore(
        score_net.ema_model,
        ctx.guidance_model,
        ctx.target_by_id[sub_id],
        float(params["rho"]),
        diffuser,
        "pdf",
        ctx.cutoff,
    )

    # Coordination guidance (disabled when coord_lambda == 0).
    coord_guidance_fn = None
    coord_sched_fn = None
    coord_lambda = float(params.get("coord_lambda", 0.0))
    if coord_lambda > 0.0:
        coord_fn = DifferentiableCoordinationNumber(r_cut=2.85, smear=0.30)
        loss_fn = CoordinationLoss(
            n_target=4.0,
            sigma_target=0.5,
            w_target=1.0,
            n_low=4.0,
            w_low=float(params.get("coord_w_low", 0.0)),
            k_low=4.0,
            n_high=7.0,
            w_high=0.0,
            k_high=4.0,
        )
        coord_guidance_fn = CoordinationGuidance(
            coord_fn=coord_fn,
            loss_fn=loss_fn,
            clamp_norm=10.0,
        )
        coord_sched_fn = CoordinationSchedule(
            schedule="constant",
            lambda_0=coord_lambda,
            tmax=params["tmax"],
            t_gate=1.0,
        )

    n_restart = int(params.get("n_restart", 1))
    pos_current = pos
    for restart_idx in range(n_restart):
        last_restart = (restart_idx == n_restart - 1)
        _, pos_current = denoise_by_sde(
            species=species,
            pos=pos_current,
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
            anneal_fn=anneal_fn if last_restart else None,
            coord_guidance=coord_guidance_fn,
            coord_schedule=coord_sched_fn,
        )

    atoms_out = Atoms(
        numbers=init_atoms.numbers,
        positions=pos_current.cpu().numpy(),
        cell=cell_np,
        pbc=[True, True, True],
    )
    atoms_out.wrap()
    return atoms_out


# --------------------------------------------------------------------------
# Search space.
# --------------------------------------------------------------------------

def _sample_params(trial) -> Dict:
    searched = {
        # Time schedule
        "tmax": trial.suggest_float("tmax", 0.4, 1.0),
        "tmin": trial.suggest_float("tmin", 1e-4, 2e-2, log=True),
        "tstep": trial.suggest_categorical("tstep", [128, 256, 512]),
        "t_rho": trial.suggest_float("t_rho", 0.5, 2.0),
        # PDF guidance strength
        "rho": trial.suggest_float("rho", 50.0, 2000.0, log=True),
        # Tersoff guidance
        "tersoff_lambda": trial.suggest_float("tersoff_lambda", 0.0, 2.0),
        "tersoff_schedule": trial.suggest_categorical(
            "tersoff_schedule", ["constant", "linear", "sigmoid"]
        ),
        "tersoff_t_gate": trial.suggest_float("tersoff_t_gate", 0.1, 0.9),
        # Restart count
        "n_restart": trial.suggest_categorical("n_restart", [1, 2, 3, 4]),
        # Prior scale (new in v7)
        "lambda_prior": trial.suggest_float("lambda_prior", 0.0, 5.0),
        # Coordination guidance (new in v7)
        "coord_lambda": trial.suggest_float("coord_lambda", 0.0, 100.0),
        "coord_w_low": trial.suggest_float("coord_w_low", 0.0, 50.0),
    }
    return {**_FIXED_PARAMS, **searched}


# --------------------------------------------------------------------------
# Trial evaluation.
# --------------------------------------------------------------------------

def _evaluate_one(
    params: Dict,
    sub_id: str,
    ctx: DeviceCtx,
    seed: int,
) -> Dict[str, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    atoms_out = _run_single(params, sub_id, ctx)
    metrics = compute_all_metrics(
        atoms_out,
        include_dihedrals=False,
        include_sq=False,
        include_voronoi=False,
        include_rings=False,
        include_tersoff=False,
    )
    errors = compute_all_errors(ctx.ref_metrics_by_id[sub_id], metrics)
    coords = metrics.coordination.coordination_numbers
    errors["undercoord_frac_le3"] = float((coords <= 3).mean())
    errors["undercoord_frac_le2"] = float((coords <= 2).mean())
    errors["undercoord_frac"] = errors["undercoord_frac_le3"]
    return errors


def _ensure_pairs(ctx: DeviceCtx, init_paths: List[Path], ref_dir: Path) -> None:
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
        target = compute_target_from_reference(
            ref_atoms, ctx.guidance_model, "pdf", ctx.cutoff, ctx.device,
        )
        ref_metrics = compute_all_metrics(
            ref_atoms,
            include_dihedrals=False,
            include_sq=False,
            include_voronoi=False,
            include_rings=False,
            include_tersoff=False,
        )
        ctx.init_atoms_by_id[sub_id] = init_atoms
        ctx.ref_atoms_by_id[sub_id] = ref_atoms
        ctx.target_by_id[sub_id] = target
        ctx.ref_metrics_by_id[sub_id] = ref_metrics


def _build_objective(
    experiment_path: str,
    buckets: List[Tuple[str, List[Path], Path]],
    n_inits: int,
    n_seeds: int,
):
    buckets = [
        (label, sorted(paths)[: max(n_inits, 1)], ref_dir)
        for label, paths, ref_dir in buckets
    ]

    _ERR_KEYS = (
        "pdf_rmse",
        "coordination_emd",
        "undercoord_frac_le3",
        "undercoord_frac_le2",
    )

    def objective(trial):
        ctx = _get_device_ctx(experiment_path, [], buckets[0][2])
        for label, paths, ref_dir in buckets:
            _ensure_pairs(ctx, paths, ref_dir)

        params = _sample_params(trial)

        obj_samples_by_label: Dict[str, List[float]] = {
            label: [] for label, _, _ in buckets
        }
        per_run_errors: List[Dict[str, object]] = []

        step = 0
        for seed_idx in range(n_seeds):
            for label, paths, _ref_dir in buckets:
                for init_path in paths:
                    sub_id = init_path.stem
                    seed = abs(
                        1000 * int(trial.number) + 100 * seed_idx + 1
                    ) + hash(sub_id) % 997
                    t0 = time.time()
                    try:
                        errors = _evaluate_one(params, sub_id, ctx, int(seed))
                    except Exception as e:
                        trial.set_user_attr("error", f"{type(e).__name__}: {e}")
                        return float("inf")
                    obj_samples_by_label[label].append(_phys_obj(errors))
                    per_run_errors.append(
                        {"label": label, "sub_id": sub_id,
                         "seed": int(seed),
                         "time_s": time.time() - t0, **errors}
                    )
            running_per_bucket = [
                float(np.mean(samples))
                for samples in obj_samples_by_label.values()
                if samples
            ]
            if running_per_bucket:
                trial.report(float(np.mean(running_per_bucket)), step=step)
            step += 1
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()

        per_bucket_obj: Dict[str, float] = {
            label: float(np.mean(obj_samples_by_label[label]))
            for label, _, _ in buckets
        }
        total = float(np.mean(list(per_bucket_obj.values())))

        for label in per_bucket_obj:
            err = {
                k: float(np.mean(
                    [e[k] for e in per_run_errors
                     if e["label"] == label
                     and isinstance(e.get(k), (int, float))]
                ))
                for k in _ERR_KEYS
            }
            trial.set_user_attr(f"mean_errors_{label}", json.dumps(err))
            trial.set_user_attr(f"obj_{label}", per_bucket_obj[label])

        mean_err = {
            k: float(np.mean([e[k] for e in per_run_errors
                              if isinstance(e.get(k), (int, float))]))
            for k in _ERR_KEYS
        }
        trial.set_user_attr("device", str(ctx.device))
        trial.set_user_attr("obj", total)
        trial.set_user_attr("mean_errors", json.dumps(mean_err))
        trial.set_user_attr(
            "bucket_labels", json.dumps([l for l, _, _ in buckets])
        )

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
    parser.add_argument("--ref-dir", type=str, default=None,
                        help="Directory of reference *.xyz files (one per init sub_id).")
    parser.add_argument("--ref-dirs", type=str, default=None,
                        help="Comma-separated list of reference dirs (multi-density).")
    parser.add_argument("--init-dir", type=str, required=True,
                        help="Directory of init *.xyz files.")
    parser.add_argument("--init-glob", type=str, default=None,
                        help="Glob pattern for inits (single-density).")
    parser.add_argument("--init-globs", type=str, default=None,
                        help="Comma-separated globs, one per density bucket.")
    parser.add_argument("--bucket-labels", type=str, default=None,
                        help="Optional comma-separated labels for buckets.")
    parser.add_argument("--n-trials", type=int, default=1000)
    parser.add_argument("--timeout", type=int, default=None, help="Seconds.")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--n-seeds", type=int, default=1,
                        help="Random seeds per init per trial.")
    parser.add_argument("--n-inits", type=int, default=5,
                        help="Number of init structures per trial (sorted, deterministic).")
    parser.add_argument("--devices", type=str, default="cuda:0",
                        help="Comma-separated CUDA device list.")
    parser.add_argument("--study-name", type=str, default="glass_phys_v7_coord")
    parser.add_argument("--storage", type=str,
                        default="research/hpo/glass_phys_v7_coord.db")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replay-best", action="store_true",
                        help="Skip search; re-run the current best trial.")
    args = parser.parse_args()

    import optuna
    from optuna.pruners import NopPruner
    from optuna.samplers import TPESampler

    global _device_pool
    _device_pool = _parse_devices(args.devices) or ["cpu"]
    print(f"Devices: {_device_pool}")

    init_dir = Path(args.init_dir).resolve()

    if args.init_globs and args.ref_dirs:
        import re
        globs = [g.strip() for g in args.init_globs.split(",") if g.strip()]
        ref_dirs = [Path(d.strip()).resolve() for d in args.ref_dirs.split(",") if d.strip()]
        if len(globs) != len(ref_dirs):
            raise SystemExit(
                f"--init-globs has {len(globs)} entries but --ref-dirs has "
                f"{len(ref_dirs)}. They must match."
            )
        if args.bucket_labels:
            labels = [l.strip() for l in args.bucket_labels.split(",")]
            if len(labels) != len(globs):
                raise SystemExit("--bucket-labels must match --init-globs count.")
        else:
            labels = []
            for g in globs:
                m = re.search(r"(\d+\.\d+)", g)
                labels.append(m.group(1) if m else g)
        buckets: List[Tuple[str, List[Path], Path]] = []
        for label, g, rd in zip(labels, globs, ref_dirs):
            paths = sorted(init_dir.glob(g))
            if not paths:
                raise SystemExit(f"No init files matched {g!r} under {init_dir}")
            missing = [p.stem for p in paths if not (rd / f"{p.stem}.xyz").exists()]
            if missing:
                raise SystemExit(
                    f"Bucket {label}: {len(missing)} inits missing refs "
                    f"in {rd}: {missing[:3]}..."
                )
            buckets.append((label, paths, rd))
        print(
            f"Multi-density study: {len(buckets)} buckets — "
            + ", ".join(f"{l}({len(p)})" for l, p, _ in buckets)
        )
    elif args.init_glob and args.ref_dir:
        ref_dir = Path(args.ref_dir).resolve()
        init_paths = sorted(init_dir.glob(args.init_glob))
        if not init_paths:
            raise SystemExit(
                f"No init files matched {args.init_glob!r} under {init_dir}"
            )
        missing = [p.stem for p in init_paths
                   if not (ref_dir / f"{p.stem}.xyz").exists()]
        if missing:
            raise SystemExit(
                f"{len(missing)} inits have no matching ref in {ref_dir}: "
                f"{missing[:3]}..."
            )
        print(f"Single-density study: {len(init_paths)} init/ref pairs in {init_dir}")
        buckets = [("all", init_paths, ref_dir)]
    else:
        raise SystemExit(
            "Provide either --init-glob+--ref-dir (single-density) "
            "or --init-globs+--ref-dirs (multi-density)."
        )

    print(f"Using first {args.n_inits} per bucket per trial.")

    storage_path = Path(args.storage)
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{storage_path}"

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_url,
        direction="minimize",
        sampler=TPESampler(seed=args.seed),
        pruner=NopPruner(),
        load_if_exists=True,
    )

    objective = _build_objective(
        args.experiment_path, buckets,
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
        print(f"  obj: {ua.get('obj', float('nan')):.6f}")
        if "mean_errors" in ua:
            print("  errors: ", ua["mean_errors"])
        return

    print(f"\nRunning {args.n_trials} trials "
          f"(n_jobs={args.n_jobs}, n_seeds={args.n_seeds}, "
          f"n_inits={args.n_inits}, cond-only => "
          f"{args.n_inits*args.n_seeds} runs/trial/bucket)")

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
                    "W_PDF": W_PDF,
                    "W_COORD": W_COORD,
                    "W_UC3": W_UC3,
                    "W_UC2": W_UC2,
                    "EPS3": EPS3,
                    "EPS2": EPS2,
                },
            },
            f, indent=2,
        )
    print(f"\nBest params written to {best_path}")

    all_trials = [t for t in study.trials if t.value is not None]
    all_trials.sort(key=lambda t: t.value)
    print("\nTop 5 trials:")
    for t in all_trials[:5]:
        ua = t.user_attrs
        print(
            f"  #{t.number}: value={t.value:.4f}  "
            f"n_restart={t.params.get('n_restart')} "
            f"tmax={t.params.get('tmax'):.3f} "
            f"rho={t.params.get('rho'):.1f} "
            f"tl={t.params.get('tersoff_lambda'):.3f} "
            f"lp={t.params.get('lambda_prior'):.3f} "
            f"cl={t.params.get('coord_lambda'):.2f}"
        )


if __name__ == "__main__":
    main()
