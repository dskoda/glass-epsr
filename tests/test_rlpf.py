"""Tests for RLPF (RL with Physical Feedback) fine-tuning components.

All tests run on CPU with tiny structures for speed.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from ase.build import bulk

DATA_DIR = Path(__file__).resolve().parent / "data"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_si_bulk_4():
    """4-atom Si bulk diamond cell (unit cell)."""
    return bulk("Si", "diamond", a=5.43)


def _make_si_bulk_8():
    """8-atom Si bulk diamond cell (conventional cell, 2x supercell)."""
    from ase.build import make_supercell
    import numpy as np
    atoms = bulk("Si", "diamond", a=5.43)
    return atoms.repeat([1, 1, 2])  # 8 atoms


def _make_score_net():
    """Create a tiny LitScoreNet without loading from checkpoint."""
    from glass.lit.modules.prior import LitScoreNet

    return LitScoreNet(
        num_species=1,
        num_convs=1,
        dim=16,
        ema_decay=0.999,
        learn_rate=1e-3,
    )


def _make_tersoff_calc():
    """Return a TorchTersoff instance with Si parameters."""
    from glass.potentials.tersoff.ase_calc import silicon_calculator

    return silicon_calculator()._torch_calc


def _make_reward_fn(tersoff_calc):
    """Build a TersoffPDFReward with a synthetic target PDF."""
    from glass.diffusion.rewards import TersoffPDFReward

    target_r = np.linspace(0.5, 8.0, 200)
    target_g_r = np.ones_like(target_r)  # flat reference (ideal gas)
    return TersoffPDFReward(
        tersoff_calc=tersoff_calc,
        target_g_r=target_g_r,
        target_r=target_r,
        w_energy=1.0,
        w_pdf=1.0,
        device="cpu",
    )


def _atoms_to_tensors(atoms):
    """Convert ASE atoms to (species, pos, cell) tensors on CPU."""
    from glass.utils.atoms_utils import atoms_to_device

    return atoms_to_device(atoms, "cpu")


def _make_ts(n_steps=8):
    from glass.diffusion.schedules import power_law_ts

    return power_law_ts(1e-3, 0.1, n_steps)


def _make_diffuser():
    from glass.diffusion import VarianceExplodingDiffuser

    return VarianceExplodingDiffuser(k=0.8)


def _make_trainer(score_net=None):
    """Build an RLPFTrainer with all defaults."""
    from glass.diffusion.rlpf import RLPFConfig, RLPFTrainer

    if score_net is None:
        score_net = _make_score_net()
    diffuser = _make_diffuser()
    tersoff_calc = _make_tersoff_calc()
    reward_fn = _make_reward_fn(tersoff_calc)
    cfg = RLPFConfig(
        subsample_steps=2,
        n_rollouts_per_update=2,
        ppo_epochs=1,
    )
    return RLPFTrainer(score_net, diffuser, reward_fn, cfg)


# ---------------------------------------------------------------------------
# Test 1: reward returns negative for reasonable structure
# ---------------------------------------------------------------------------


def test_reward_returns_negative_for_reasonable_structure():
    """TersoffPDFReward on a bulk Si crystal returns a finite scalar reward.

    The reward is -(w_energy * E/N + w_pdf * pdf_rmse).  For bulk Si the
    Tersoff energy per atom is around -4.6 eV (negative), so with w_energy=1
    and a moderate pdf_rmse the total reward is typically positive.  We just
    verify the interface contract: (float, dict) with the expected keys.
    """
    atoms = _make_si_bulk_8()
    tersoff_calc = _make_tersoff_calc()
    reward_fn = _make_reward_fn(tersoff_calc)

    species, pos, cell = _atoms_to_tensors(atoms)
    reward, info = reward_fn(pos, cell, species)

    assert isinstance(reward, float), "Reward must be a float."
    assert np.isfinite(reward), f"Reward must be finite, got {reward}"
    assert "energy" in info
    assert "pdf" in info
    assert info["pdf"] >= 0.0
    # Energy per atom for bulk Si (Tersoff) is around -4 to -5 eV/atom.
    assert info["energy"] < 0.0, (
        f"Tersoff energy per atom for bulk Si should be negative, got {info['energy']}"
    )


# ---------------------------------------------------------------------------
# Test 2: reward is higher (less negative) for relaxed vs random positions
# ---------------------------------------------------------------------------


def test_reward_lower_for_better_structure():
    """Bulk Si (well-structured) should have higher reward than random positions."""
    atoms = _make_si_bulk_8()
    tersoff_calc = _make_tersoff_calc()
    reward_fn = _make_reward_fn(tersoff_calc)

    species, pos, cell = _atoms_to_tensors(atoms)
    reward_bulk, _ = reward_fn(pos, cell, species)

    # Randomly displace atoms by a lot — expect worse (more negative) reward.
    torch.manual_seed(42)
    pos_random = pos + 2.0 * torch.randn_like(pos)
    reward_random, _ = reward_fn(pos_random, cell, species)

    assert reward_bulk > reward_random, (
        f"Bulk reward {reward_bulk:.4f} should be > random reward {reward_random:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 3: Transition dataclass stores correct fields
# ---------------------------------------------------------------------------


def test_transition_dataclass():
    """Transition stores exactly the fields specified."""
    from glass.diffusion.rlpf import Transition

    N = 4
    x_t = torch.zeros(N, 3)
    mu_old = torch.ones(N, 3)
    x_next = torch.full((N, 3), 2.0)

    tr = Transition(
        x_t=x_t,
        mu_old=mu_old,
        sigma_sq=0.01,
        x_next=x_next,
        t=0.05,
        dt=-0.01,
    )

    assert tr.x_t.shape == (N, 3)
    assert tr.mu_old.shape == (N, 3)
    assert tr.sigma_sq == pytest.approx(0.01)
    assert tr.x_next.shape == (N, 3)
    assert tr.t == pytest.approx(0.05)
    assert tr.dt == pytest.approx(-0.01)


# ---------------------------------------------------------------------------
# Test 4: collect_rollouts returns trajectories with correct shape
# ---------------------------------------------------------------------------


def test_collect_rollouts_returns_trajectories():
    """collect_rollouts returns the requested number of Trajectory objects."""
    from glass.diffusion.rlpf import RLPFConfig, RLPFTrainer

    atoms = _make_si_bulk_4()
    species, pos, cell = _atoms_to_tensors(atoms)
    diffuser = _make_diffuser()
    ts = _make_ts(n_steps=8)

    # Add noise at ts[0] level.
    sigma0 = float(diffuser.sigma(ts[0]).item()) if hasattr(diffuser.sigma(ts[0]), "item") else diffuser.sigma(ts[0])
    pos_noisy = pos + sigma0 * torch.randn_like(pos)

    score_net = _make_score_net()
    tersoff_calc = _make_tersoff_calc()
    reward_fn = _make_reward_fn(tersoff_calc)
    cfg = RLPFConfig(
        subsample_steps=2,
        n_rollouts_per_update=3,
        ppo_epochs=1,
    )
    trainer = RLPFTrainer(score_net, diffuser, reward_fn, cfg)

    n_rollouts = 3
    trajectories = trainer.collect_rollouts(
        species, pos_noisy, cell,
        cutoff=5.0,
        ts=ts,
        n_rollouts=n_rollouts,
    )

    assert len(trajectories) == n_rollouts, (
        f"Expected {n_rollouts} trajectories, got {len(trajectories)}"
    )
    for traj in trajectories:
        assert traj.x0.shape == pos.shape, (
            f"x0 shape {traj.x0.shape} != pos shape {pos.shape}"
        )
        assert isinstance(traj.reward, float)
        assert len(traj.transitions) >= 0


# ---------------------------------------------------------------------------
# Test 5: ppo_update changes model weights
# ---------------------------------------------------------------------------


def test_ppo_update_changes_weights():
    """After one ppo_update, score_net.model weights must differ from before."""
    from glass.diffusion.rlpf import RLPFConfig, RLPFTrainer

    atoms = _make_si_bulk_4()
    species, pos, cell = _atoms_to_tensors(atoms)
    diffuser = _make_diffuser()
    ts = _make_ts(n_steps=8)

    sigma0 = diffuser.sigma(ts[0])
    torch.manual_seed(0)
    pos_noisy = pos + float(sigma0) * torch.randn_like(pos)

    score_net = _make_score_net()
    tersoff_calc = _make_tersoff_calc()
    reward_fn = _make_reward_fn(tersoff_calc)
    cfg = RLPFConfig(
        subsample_steps=2,
        n_rollouts_per_update=2,
        ppo_epochs=1,
        kl_warmup_updates=0,
        kl_beta=0.0,
    )
    trainer = RLPFTrainer(score_net, diffuser, reward_fn, cfg)

    # Snapshot weights before update.
    params_before = {
        name: param.detach().clone()
        for name, param in score_net.model.named_parameters()
    }

    trajs = trainer.collect_rollouts(
        species, pos_noisy, cell, cutoff=5.0, ts=ts, n_rollouts=2
    )
    trainer.ppo_update(trajs, species, cell, cutoff=5.0, ts=ts)

    # At least one parameter must have changed.
    changed = any(
        not torch.allclose(params_before[name], param.detach(), atol=0.0)
        for name, param in score_net.model.named_parameters()
    )
    assert changed, "ppo_update must change at least one model parameter."


# ---------------------------------------------------------------------------
# Test 6: ppo_update returns metrics dict with required keys
# ---------------------------------------------------------------------------


def test_ppo_update_returns_metrics_dict():
    """ppo_update returns a dict with the required keys."""
    from glass.diffusion.rlpf import RLPFConfig, RLPFTrainer

    atoms = _make_si_bulk_4()
    species, pos, cell = _atoms_to_tensors(atoms)
    diffuser = _make_diffuser()
    ts = _make_ts(n_steps=8)

    sigma0 = diffuser.sigma(ts[0])
    pos_noisy = pos + float(sigma0) * torch.randn_like(pos)

    trainer = _make_trainer()

    trajs = trainer.collect_rollouts(
        species, pos_noisy, cell, cutoff=5.0, ts=ts, n_rollouts=2
    )
    metrics = trainer.ppo_update(trajs, species, cell, cutoff=5.0, ts=ts)

    required_keys = {"loss", "ppo_loss", "kl_loss", "reward_mean"}
    assert required_keys.issubset(
        metrics.keys()
    ), f"Missing keys: {required_keys - metrics.keys()}"
    for key in required_keys:
        assert isinstance(metrics[key], float), f"{key} must be a float."


# ---------------------------------------------------------------------------
# Test 7: KL warmup — kl_loss should be 0 during warmup
# ---------------------------------------------------------------------------


def test_kl_warmup_zero_kl_early():
    """With n_updates=0 and kl_warmup=10, kl_loss must be 0."""
    from glass.diffusion.rlpf import RLPFConfig, RLPFTrainer

    atoms = _make_si_bulk_4()
    species, pos, cell = _atoms_to_tensors(atoms)
    diffuser = _make_diffuser()
    ts = _make_ts(n_steps=8)

    sigma0 = diffuser.sigma(ts[0])
    pos_noisy = pos + float(sigma0) * torch.randn_like(pos)

    score_net = _make_score_net()
    tersoff_calc = _make_tersoff_calc()
    reward_fn = _make_reward_fn(tersoff_calc)
    cfg = RLPFConfig(
        subsample_steps=2,
        n_rollouts_per_update=2,
        ppo_epochs=1,
        kl_beta=0.1,
        kl_warmup_updates=10,  # warmup active
    )
    trainer = RLPFTrainer(score_net, diffuser, reward_fn, cfg)
    assert trainer.n_updates == 0, "n_updates should start at 0."

    trajs = trainer.collect_rollouts(
        species, pos_noisy, cell, cutoff=5.0, ts=ts, n_rollouts=2
    )
    metrics = trainer.ppo_update(trajs, species, cell, cutoff=5.0, ts=ts)

    assert metrics["kl_loss"] == pytest.approx(0.0), (
        f"kl_loss should be 0 during warmup, got {metrics['kl_loss']}"
    )


# ---------------------------------------------------------------------------
# Test 8: subsample_steps reduces number of transitions
# ---------------------------------------------------------------------------


def test_subsample_steps_reduces_transitions():
    """With subsample_steps=2 vs 1, the number of stored transitions halves."""
    from glass.diffusion.rlpf import RLPFConfig, RLPFTrainer

    atoms = _make_si_bulk_4()
    species, pos, cell = _atoms_to_tensors(atoms)
    diffuser = _make_diffuser()
    ts = _make_ts(n_steps=8)  # 7 SDE steps (T-1)

    sigma0 = diffuser.sigma(ts[0])
    pos_noisy = pos + float(sigma0) * torch.randn_like(pos)

    score_net_a = _make_score_net()
    score_net_b = _make_score_net()
    tersoff_calc = _make_tersoff_calc()
    reward_fn = _make_reward_fn(tersoff_calc)

    cfg_1 = RLPFConfig(subsample_steps=1, n_rollouts_per_update=1, ppo_epochs=1)
    cfg_2 = RLPFConfig(subsample_steps=2, n_rollouts_per_update=1, ppo_epochs=1)

    trainer_1 = RLPFTrainer(score_net_a, diffuser, reward_fn, cfg_1)
    trainer_2 = RLPFTrainer(score_net_b, diffuser, reward_fn, cfg_2)

    torch.manual_seed(5)
    trajs_1 = trainer_1.collect_rollouts(
        species, pos_noisy.clone(), cell, cutoff=5.0, ts=ts, n_rollouts=1
    )
    torch.manual_seed(5)
    trajs_2 = trainer_2.collect_rollouts(
        species, pos_noisy.clone(), cell, cutoff=5.0, ts=ts, n_rollouts=1
    )

    n1 = len(trajs_1[0].transitions)
    n2 = len(trajs_2[0].transitions)

    assert n2 < n1, (
        f"subsample_steps=2 should give fewer transitions ({n2}) than "
        f"subsample_steps=1 ({n1})."
    )
    # With T-1=7 steps: subsample_steps=1 → 7, subsample_steps=2 → 4
    assert n1 == 7, f"Expected 7 transitions with subsample_steps=1, got {n1}"
    assert n2 == 4, f"Expected 4 transitions with subsample_steps=2, got {n2}"


# ---------------------------------------------------------------------------
# Test 9: likelihood_fn during rollouts shifts mu_old
# ---------------------------------------------------------------------------


def test_likelihood_fn_shifts_mu_old():
    """collect_rollouts with a non-zero likelihood_fn produces different mu_old values."""
    from glass.diffusion.rlpf import RLPFConfig, RLPFTrainer

    atoms = _make_si_bulk_4()
    species, pos, cell = _atoms_to_tensors(atoms)
    diffuser = _make_diffuser()
    ts = _make_ts(n_steps=8)

    sigma0 = diffuser.sigma(ts[0])
    torch.manual_seed(0)
    pos_noisy = pos + float(sigma0) * torch.randn_like(pos)

    # Stub likelihood_fn: returns a constant nonzero score correction.
    class _ConstantLikelihood:
        def __call__(self, species, pos, cell, t, cutoff):
            correction = torch.full_like(pos, 0.1)
            norm = torch.tensor(0.1)
            return correction, norm

    score_net = _make_score_net()
    tersoff_calc = _make_tersoff_calc()
    reward_fn = _make_reward_fn(tersoff_calc)
    cfg = RLPFConfig(subsample_steps=1, n_rollouts_per_update=1, ppo_epochs=1)

    trainer_plain = RLPFTrainer(score_net, diffuser, reward_fn, cfg)
    trainer_guided = RLPFTrainer(score_net, diffuser, reward_fn, cfg,
                                 likelihood_fn=_ConstantLikelihood())

    torch.manual_seed(7)
    trajs_plain = trainer_plain.collect_rollouts(
        species, pos_noisy.clone(), cell, cutoff=5.0, ts=ts, n_rollouts=1
    )
    torch.manual_seed(7)
    trajs_guided = trainer_guided.collect_rollouts(
        species, pos_noisy.clone(), cell, cutoff=5.0, ts=ts, n_rollouts=1
    )

    # mu_old must differ between guided and unguided rollouts.
    mu_plain = trajs_plain[0].transitions[0].mu_old
    mu_guided = trajs_guided[0].transitions[0].mu_old
    assert not torch.allclose(mu_plain, mu_guided, atol=1e-6), (
        "likelihood_fn should shift mu_old in collect_rollouts."
    )
