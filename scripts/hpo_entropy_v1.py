#!/usr/bin/env python3
"""Entropy-guidance HPO v1 — search for optimal ACSF variance guidance params.

Forked from ``hpo_phys_v6.py``. v6 found the best objective at 37.91 with
n_restart=3, but still leaves undercoordinated atoms and ring-topology defects.
This study adds the structural entropy term (ACSF descriptor variance, Cliffe
et al. PRB 95 224108 2017) as an additional guidance signal and searches its
four hyperparameters while keeping all v6 parameters fixed.

The objective is identical to v5/v6 (8-term, log-scale undercoord penalties).
Focus: ρ=1.5 g/cm³ (OOD), PDF-conditional.

Fixed parameters (v6 best trial #39, glass_phys_v6_restart, 2026-05-21):
    tmin=9.267e-3, tmax=0.938, tstep=256, t_rho=1.01, n_restart=3
    tersoff_lambda=0.30 (user preference; HPO best was 0.128)
    tersoff_schedule=sigmoid, tersoff_t_gate=0.490
    n_corr=2, corr_step_size=0.44, corr_t_gate=0.464
    rho=416.0, N_anneal=0

Searched parameters (4):
    entropy_lambda   ∈ [0.01, 100.0]  (log-uniform)
    entropy_schedule ∈ {constant, linear, sigmoid}
    entropy_t_gate   ∈ [0.1, 1.0]
    entropy_r_cut    ∈ [3.0, 6.0]  (Å — ACSF cutoff)

Warm-start trials (enqueued when study is fresh):
    Trial 0 — lambda=0 (entropy off): reproduces v6 best as baseline anchor.
    Trial 1 — lambda=1 (default on, constant, t_gate=1.0, r_cut=4.0).

Usage
-----

Smoke test (single GPU, 4 trials, ~2 min)::

    python scripts/hpo_entropy_v1.py research/density_extrapolation/experiment/ \\
        --ref-dir research/density_extrapolation/results/generated/cond/density_1.5/reference \\
        --init-dir research/density_extrapolation/experiment/inits \\
        --init-glob "init_Si_1.5_*.xyz" \\
        --n-trials 4 --n-seeds 1 --n-inits 1 \\
        --n-jobs 1 --devices cuda:0 \\
        --study-name glass_entropy_v1_smoke \\
        --storage /tmp/glass_entropy_v1_smoke.db

Full run (4 GPUs, 100 trials, ~3–5 h)::

    python scripts/hpo_entropy_v1.py research/density_extrapolation/experiment/ \\
        --ref-dir research/density_extrapolation/results/generated/cond/density_1.5/reference \\
        --init-dir research/density_extrapolation/experiment/inits \\
        --init-glob "init_Si_1.5_*.xyz" \\
        --n-trials 100 --n-seeds 1 --n-inits 5 \\
        --n-jobs 4 --devices cuda:0,cuda:1,cuda:2,cuda:3 \\
        --study-name glass_entropy_v1_15ood \\
        --storage research/hpo/glass_entropy_v1_15ood.db

Resume (re-run the same command; SQLite + load_if_exists=True).

Replay best at higher seed count::

    python scripts/hpo_entropy_v1.py research/density_extrapolation/experiment/ \\
        --ref-dir ... --init-dir ... --init-glob "init_Si_1.5_*.xyz" \\
        --study-name glass_entropy_v1_15ood \\
        --storage research/hpo/glass_entropy_v1_15ood.db \\
        --replay-best --n-seeds 5 --n-inits 5
"""

from __future__ import annotations

import argparse
import copy
import glob
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

from glass.descriptors import EntropyGuidance, EntropySchedule, TorchACSF
from glass.diffusion.annealing import make_anneal_fn
from glass.diffusion.sampling import denoise_by_sde
from glass.diffusion.schedules import power_law_ts
from glass.experiment import Experiment
from glass.lit.datamodules import StructureSpecDataModule
from glass.lit.modules import LitScoreNet
from glass.lit.modules.guidance import create_guidance_model
from glass.lit.modules.likelihood import LikelihoodScore
from glass.lit.modules.tersoff_guidance import TersoffEnergyGuidance, TersoffSchedule
from glass.metrics import compute_all_metrics
from glass.metrics.errors import compute_all_errors
from glass.utils.atoms_utils import (
    atoms_to_device,
    compute_prior_score,
    compute_target_from_reference,
)


# --------------------------------------------------------------------------
# Objective weights — identical to v5/v6.
# --------------------------------------------------------------------------

W_PDF = 1.0
W_COORD = 3.0
W_UC3 = 18.0      # log-scale weight for fraction with coord ≤ 3
W_UC2 = 24.0      # log-scale weight for fraction with coord ≤ 2 (dangling)
EPS3 = 0.005      # 0.5 % saturates the W_UC3 lower-bound noise
EPS2 = 0.001      # 0.1 % — coord-2 atoms are catastrophic, so steeper

W_TERSOFF_E = 3.0
W_TERSOFF_F = 2.0
W_RINGS = 0.5
W_ADF = 0.01


# Fixed parameters: v6 best trial #39 (glass_phys_v6_restart, 2026-05-21).
# tersoff_lambda=0.30 is the user-set value (HPO best was 0.128).
_FIXED_PARAMS: Dict = {
    "tmin": 9.267e-3,
    "tmax": 0.938,
    "tstep": 256,
    "t_rho": 1.01,
    "n_restart": 3,
    "tersoff_lambda": 0.30,
    "tersoff_schedule": "sigmoid",
    "tersoff_t_gate": 0.490,
    "n_corr": 2,
    "corr_step_size": 0.44,
    "corr_t_gate": 0.464,
    "rho": 416.0,
    "N_anneal": 0,
    "T0": 0.01,
    "anneal_lr": 1e-3,
}


def _uc_penalty(frac: float, weight: float, eps: float) -> float:
    """log10(1 + frac/eps) · weight. Zero at frac=0, ~weight at frac=eps."""
    return weight * math.log10(1.0 + max(float(frac), 0.0) / eps)


def _phys_obj(errors: Dict[str, float]) -> float:
    return (
        W_PDF * float(errors["pdf_rmse"])
        + W_COORD * float(errors["coordination_emd"])
        + _uc_penalty(errors["undercoord_frac_le3"], W_UC3, EPS3)
        + _uc_penalty(errors["undercoord_frac_le2"], W_UC2, EPS2)
        + W_TERSOFF_E * float(errors["tersoff_energy_error"])
        + W_TERSOFF_F * float(errors["tersoff_forces_rms_error"])
        + W_RINGS * float(errors["rings_emd"])
        + W_ADF * float(errors["adf_rmse"])
    )


# --------------------------------------------------------------------------
# Per-device cache.
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
                include_rings=True,
                rings_maxlength=10,
                include_tersoff=True,
                tersoff_device=str(ctx.device),
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
    mode: str,  # always "cond" in this study
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

    # Entropy guidance: clamp_norm mirrors Tersoff; recreated per trial
    # because r_cut affects the TorchACSF neighbour graph.
    entropy_guide = None
    entropy_sched = None
    entropy_lam = float(params["entropy_lambda"])
    if entropy_lam > 1e-6:  # skip for baseline anchor (lambda=1e-9)
        entropy_guide = EntropyGuidance(
            acsf=TorchACSF.for_silicon(r_cut=float(params["entropy_r_cut"])),
            clamp_norm=10.0,
        )
        entropy_sched = EntropySchedule(
            schedule=params["entropy_schedule"],
            lambda_0=entropy_lam,
            tmax=params["tmax"],
            t_gate=float(params["entropy_t_gate"]),
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
            entropy_guidance=entropy_guide,
            entropy_schedule=entropy_sched,
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
        "entropy_lambda": trial.suggest_float("entropy_lambda", 0.01, 100.0, log=True),
        "entropy_schedule": trial.suggest_categorical(
            "entropy_schedule", ["constant", "linear", "sigmoid"]
        ),
        "entropy_t_gate": trial.suggest_float("entropy_t_gate", 0.1, 1.0),
        "entropy_r_cut": trial.suggest_float("entropy_r_cut", 3.0, 6.0),
    }
    return {**_FIXED_PARAMS, **searched}


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
        include_rings=True,
        rings_maxlength=10,
        include_tersoff=True,
        tersoff_device=str(ctx.device),
    )
    errors = compute_all_errors(ctx.ref_metrics_by_id[sub_id], metrics)
    coords = metrics.coordination.coordination_numbers
    errors["undercoord_frac_le3"] = float((coords <= 3).mean())
    errors["undercoord_frac_le2"] = float((coords <= 2).mean())
    errors["undercoord_frac"] = errors["undercoord_frac_le3"]
    return errors


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
        "rings_emd",
        "tersoff_energy_error",
        "tersoff_forces_rms_error",
        "adf_rmse",
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
                        errors = _evaluate_one(
                            params, sub_id, "cond", ctx, int(seed),
                        )
                    except Exception as e:
                        trial.set_user_attr("error", f"{type(e).__name__}: {e}")
                        return float("inf")
                    obj_samples_by_label[label].append(_phys_obj(errors))
                    per_run_errors.append(
                        {"label": label, "sub_id": sub_id, "mode": "cond",
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
            trial.set_user_attr(f"mean_errors_cond_{label}", json.dumps(err))
            trial.set_user_attr(f"obj_cond_{label}", per_bucket_obj[label])

        mean_err_cond = {
            k: float(np.mean([e[k] for e in per_run_errors
                              if isinstance(e.get(k), (int, float))]))
            for k in _ERR_KEYS
        }
        trial.set_user_attr("device", str(ctx.device))
        trial.set_user_attr("obj_cond", total)
        trial.set_user_attr("mean_errors_cond", json.dumps(mean_err_cond))
        trial.set_user_attr(
            "bucket_labels", json.dumps([l for l, _, _ in buckets])
        )

        return total

    return objective


def _ensure_pairs(ctx: "DeviceCtx", init_paths: List[Path], ref_dir: Path) -> None:
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
            include_rings=True,
            rings_maxlength=10,
            include_tersoff=True,
            tersoff_device=str(ctx.device),
        )
        ctx.init_atoms_by_id[sub_id] = init_atoms
        ctx.ref_atoms_by_id[sub_id] = ref_atoms
        ctx.target_by_id[sub_id] = target
        ctx.ref_metrics_by_id[sub_id] = ref_metrics


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------

def _parse_devices(s: str) -> List[str]:
    return [d.strip() for d in s.split(",") if d.strip()]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("experiment_path", type=str,
                        help="Experiment dir containing checkpoints/, config.yaml.")
    parser.add_argument("--ref-dir", type=str, default=None,
                        help="Directory of reference *.xyz files (single-density). "
                             "Use --ref-dirs for multi-density.")
    parser.add_argument("--ref-dirs", type=str, default=None,
                        help="Comma-separated reference dirs, one per --init-globs entry.")
    parser.add_argument("--init-dir", type=str, required=True,
                        help="Directory of init *.xyz files.")
    parser.add_argument("--init-glob", type=str, default=None,
                        help="Glob pattern (single-density). Use --init-globs for multi-density.")
    parser.add_argument("--init-globs", type=str, default=None,
                        help="Comma-separated globs, one per density bucket (multi-density).")
    parser.add_argument("--bucket-labels", type=str, default=None,
                        help="Optional comma-separated bucket labels (parallel to --init-globs).")
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=None, help="Seconds.")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--n-seeds", type=int, default=1,
                        help="Random seeds per init per trial.")
    parser.add_argument("--n-inits", type=int, default=5,
                        help="Number of init structures per trial.")
    parser.add_argument("--devices", type=str, default="cuda:0",
                        help="Comma-separated CUDA device list.")
    parser.add_argument("--study-name", type=str, default="glass_entropy_v1_15ood")
    parser.add_argument("--storage", type=str,
                        default="research/hpo/glass_entropy_v1_15ood.db")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replay-best", action="store_true",
                        help="Skip search; re-run the current best trial with "
                             "the provided --n-seeds and --n-inits.")
    args = parser.parse_args()

    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler

    global _device_pool
    _device_pool = _parse_devices(args.devices) or ["cpu"]
    print(f"Devices: {_device_pool}")

    init_dir = Path(args.init_dir).resolve()

    if args.init_globs and args.ref_dirs:
        globs = [g.strip() for g in args.init_globs.split(",") if g.strip()]
        ref_dirs = [Path(d.strip()).resolve() for d in args.ref_dirs.split(",") if d.strip()]
        if len(globs) != len(ref_dirs):
            raise SystemExit(
                f"--init-globs has {len(globs)} entries but --ref-dirs has "
                f"{len(ref_dirs)}."
            )
        if args.bucket_labels:
            labels = [l.strip() for l in args.bucket_labels.split(",")]
            if len(labels) != len(globs):
                raise SystemExit("--bucket-labels must match --init-globs count.")
        else:
            import re
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
                    f"Bucket {label}: {len(missing)} inits missing refs in {rd}: "
                    f"{missing[:3]}..."
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
        print(f"Single-density study: {len(init_paths)} init/ref pairs")
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
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=0, interval_steps=1),
        load_if_exists=True,
    )

    objective = _build_objective(
        args.experiment_path, buckets,
        n_inits=args.n_inits, n_seeds=args.n_seeds,
    )

    # Warm-start: two anchor trials so TPE starts from known-good points.
    #   Trial 0 — entropy effectively off (lambda=1e-9, below any useful threshold):
    #     reproduces v6 best, gives baseline obj. 0.0 is invalid for log-uniform.
    #   Trial 1 — entropy on (lambda=1, constant, full t_gate, r_cut=4 Å).
    if len(study.trials) == 0:
        study.enqueue_trial({
            "entropy_lambda": 1e-9,
            "entropy_schedule": "constant",
            "entropy_t_gate": 1.0,
            "entropy_r_cut": 4.0,
        })
        study.enqueue_trial({
            "entropy_lambda": 1.0,
            "entropy_schedule": "constant",
            "entropy_t_gate": 1.0,
            "entropy_r_cut": 4.0,
        })

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
        print(f"  obj_cond: {ua.get('obj_cond', float('nan')):.6f}")
        if "mean_errors_cond" in ua:
            print("  cond errors:", ua["mean_errors_cond"])
        for k, v in ua.items():
            if k.startswith("mean_errors_cond_") or k.startswith("obj_cond_"):
                print(f"  {k}: {v}")
        return

    print(
        f"\nRunning {args.n_trials} trials "
        f"(n_jobs={args.n_jobs}, n_seeds={args.n_seeds}, "
        f"n_inits={args.n_inits}, cond-only => "
        f"{args.n_inits * args.n_seeds} runs/trial/bucket)"
    )

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
                "fixed_params": _FIXED_PARAMS,
                "user_attrs": best.user_attrs,
                "weights": {
                    "W_PDF": W_PDF, "W_COORD": W_COORD,
                    "W_UC3": W_UC3, "W_UC2": W_UC2,
                    "EPS3": EPS3, "EPS2": EPS2,
                    "W_RINGS": W_RINGS,
                    "W_TERSOFF_E": W_TERSOFF_E, "W_TERSOFF_F": W_TERSOFF_F,
                    "W_ADF": W_ADF,
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
        ent_lam = t.params.get("entropy_lambda", 0)
        ent_sch = t.params.get("entropy_schedule", "—")
        ent_tg = t.params.get("entropy_t_gate", "—")
        ent_rc = t.params.get("entropy_r_cut", "—")
        print(
            f"  #{t.number}: value={t.value:.4f}  "
            f"cond={ua.get('obj_cond', float('nan')):.4f}  "
            f"ent_lam={ent_lam:.3g} ent_sch={ent_sch} "
            f"ent_tg={ent_tg:.2f} ent_rc={ent_rc:.2f}"
        )


if __name__ == "__main__":
    main()
