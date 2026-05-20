"""RLPF (RL with Physical Feedback) fine-tuning for the score diffusion model.

Implements a PPO-style policy gradient update where:
- Rollouts are collected using the EMA model (stable inference).
- The trainable `model` is optimised with a clipped PPO objective.
- A KL-divergence penalty relative to the frozen reference policy is added.

Key design choices:
- The Gaussian log-ratio is computed analytically from the SDE step variance,
  avoiding the need to store full log-prob gradients.
- Transitions are stored on CPU and converted back on demand to avoid GPU OOM.
- KL is warmed up from 0 to avoid destabilising the policy early in training.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
from torch import Tensor


@dataclass
class RLPFConfig:
    """Hyperparameters for the RLPF trainer."""

    clip_eps: float = 0.2
    ppo_epochs: int = 4
    kl_beta: float = 0.01
    kl_warmup_updates: int = 10
    lr: float = 1e-5
    max_grad_norm: float = 1.0
    n_rollouts_per_update: int = 8
    subsample_steps: int = 4
    baseline_ema: float = 0.99
    normalize_advantages: bool = True


@dataclass
class Transition:
    """A single SDE step recorded during rollout collection.

    Attributes:
        x_t: Noisy positions *before* the step [N, 3].
        mu_old: Old policy mean = x_t - g2(t)*score_old*dt [N, 3].
        sigma_sq: Step variance g2(t)*|dt| (scalar float).
        x_next: Positions *after* the step [N, 3].
        t: Time value at this step (scalar float).
        dt: Time increment (scalar float, typically negative).
    """

    x_t: Tensor
    mu_old: Tensor
    sigma_sq: float
    x_next: Tensor
    t: float
    dt: float


@dataclass
class Trajectory:
    """A complete reverse-SDE trajectory with its terminal reward.

    Attributes:
        transitions: List of Transition objects (subsampled).
        x0: Final denoised positions [N, 3].
        reward: Scalar reward for the final structure.
    """

    transitions: List[Transition]
    x0: Tensor
    reward: float


class RLPFTrainer:
    """PPO-style trainer for RLPF fine-tuning of the score network.

    Args:
        score_net: LitScoreNet instance whose ``model`` will be trained.
        diffuser: VarianceExplodingDiffuser instance.
        reward_fn: Callable(pos, cell, species) -> (float, dict).
        cfg: RLPFConfig with training hyperparameters.
    """

    def __init__(
        self,
        score_net,
        diffuser,
        reward_fn,
        cfg: RLPFConfig = None,
        likelihood_fn=None,
    ) -> None:
        if cfg is None:
            cfg = RLPFConfig()
        self.score_net = score_net
        self.diffuser = diffuser
        self.reward_fn = reward_fn
        self.cfg = cfg
        # Optional PDF guidance applied during rollout collection (Phase H-B).
        # If set, each SDE step adds the likelihood gradient to the prior score,
        # steering trajectories toward the target structure during data collection.
        self.likelihood_fn = likelihood_fn

        # Frozen reference policy: deep copy of EMA model weights.
        # AveragedModel stores parameters averaged under the EMA scheme;
        # we keep this frozen for the KL penalty.
        self.score_ref = copy.deepcopy(score_net.ema_model)
        self.score_ref.eval()
        for p in self.score_ref.parameters():
            p.requires_grad_(False)

        # Optimiser on the trainable `model` (not EMA).
        self.optimizer = torch.optim.Adam(
            score_net.model.parameters(), lr=cfg.lr
        )

        # Running reward baseline (EMA).
        self.baseline: Optional[float] = None

        # Update counter for KL warmup.
        self.n_updates: int = 0

    # ------------------------------------------------------------------
    # Score helpers
    # ------------------------------------------------------------------

    def _build_graph(self, pos: Tensor, cell: Tensor, cutoff: float):
        from glass.nn import periodic_radius_graph

        edge_index, edge_vec = periodic_radius_graph(pos, cutoff, cell)
        edge_attr = torch.hstack([edge_vec, edge_vec.norm(dim=-1, keepdim=True)])
        return edge_index, edge_attr

    def _score_new(
        self,
        species: Tensor,
        pos: Tensor,
        cell: Tensor,
        t: Tensor,
        cutoff: float,
    ) -> Tensor:
        """Score from the *trainable* model (with grad)."""
        edge_index, edge_attr = self._build_graph(pos, cell, cutoff)
        return self.score_net.model(
            species, edge_index, edge_attr, t, self.diffuser.sigma(t)
        )

    def _score_ref(
        self,
        species: Tensor,
        pos: Tensor,
        cell: Tensor,
        t: Tensor,
        cutoff: float,
    ) -> Tensor:
        """Score from the frozen reference policy (no grad)."""
        with torch.no_grad():
            edge_index, edge_attr = self._build_graph(pos, cell, cutoff)
            return self.score_ref(
                species, edge_index, edge_attr, t, self.diffuser.sigma(t)
            )

    def _score_ema(
        self,
        species: Tensor,
        pos: Tensor,
        cell: Tensor,
        t: Tensor,
        cutoff: float,
    ) -> Tensor:
        """Score from the EMA model used for rollout collection (no grad)."""
        with torch.no_grad():
            edge_index, edge_attr = self._build_graph(pos, cell, cutoff)
            return self.score_net.ema_model(
                species, edge_index, edge_attr, t, self.diffuser.sigma(t)
            )

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def collect_rollouts(
        self,
        species: Tensor,
        pos_init: Tensor,
        cell: Tensor,
        cutoff: float,
        ts: Tensor,
        n_rollouts: Optional[int] = None,
    ) -> List[Trajectory]:
        """Run the reverse SDE and record transitions for PPO.

        The EMA model is used for stable inference. Tensors are stored on
        CPU to avoid GPU OOM.

        Args:
            species: [N, num_sp] one-hot species tensor.
            pos_init: Initial noisy positions [N, 3] at noise level ts[0].
            cell: Unit cell [3, 3].
            cutoff: Graph cutoff radius.
            ts: Descending time schedule [T], ts[0]=tmax, ts[-1]=tmin.
            n_rollouts: Number of independent trajectories; defaults to
                cfg.n_rollouts_per_update.

        Returns:
            List of Trajectory objects.
        """
        if n_rollouts is None:
            n_rollouts = self.cfg.n_rollouts_per_update

        device = pos_init.device
        f, g, g2 = self.diffuser.f, self.diffuser.g, self.diffuser.g2
        ts_dev = ts.to(device).view(-1)
        T = len(ts_dev)

        trajectories: List[Trajectory] = []

        for _ in range(n_rollouts):
            pos = pos_init.clone()
            transitions: List[Transition] = []

            for i in range(T - 1):
                t_val = ts_dev[i]
                t_next = ts_dev[i + 1]
                dt = float((t_next - t_val).item())
                t_scalar = t_val.reshape(1)

                score = self._score_ema(species, pos, cell, t_scalar, cutoff)

                if self.likelihood_fn is not None:
                    l_score, _ = self.likelihood_fn(species, pos, cell, t_scalar, cutoff)
                    score = score + l_score

                g2_t = float(g2(t_val))
                g_t = float(g(t_val))
                sigma_sq = g2_t * abs(dt)

                mu_old = pos - g2_t * score * dt  # mean of p(x_{i+1} | x_i)

                noise = abs(dt) ** 0.5 * torch.randn_like(pos)
                x_next = mu_old + g_t * noise

                # Store every subsample_steps-th step.
                if i % self.cfg.subsample_steps == 0:
                    transitions.append(
                        Transition(
                            x_t=pos.detach().cpu(),
                            mu_old=mu_old.detach().cpu(),
                            sigma_sq=sigma_sq,
                            x_next=x_next.detach().cpu(),
                            t=float(t_val.item()),
                            dt=dt,
                        )
                    )

                pos = x_next.detach()

            # Terminal reward
            reward, _ = self.reward_fn(pos, cell, species)

            trajectories.append(
                Trajectory(
                    transitions=transitions,
                    x0=pos.detach().cpu(),
                    reward=float(reward),
                )
            )

        return trajectories

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def ppo_update(
        self,
        trajectories: List[Trajectory],
        species: Tensor,
        cell: Tensor,
        cutoff: float,
        ts: Tensor,
    ) -> dict:
        """Run PPO update on collected trajectories.

        Returns:
            dict with keys: loss, ppo_loss, kl_loss, reward_mean.
        """
        device = next(self.score_net.model.parameters()).device
        rewards = [traj.reward for traj in trajectories]
        reward_mean = float(sum(rewards) / len(rewards))

        # Update running baseline (EMA).
        if self.baseline is None:
            self.baseline = reward_mean
        else:
            self.baseline = (
                self.cfg.baseline_ema * self.baseline
                + (1.0 - self.cfg.baseline_ema) * reward_mean
            )

        # Compute advantages (reward - baseline), same for all transitions in a traj.
        advantages_raw = [r - self.baseline for r in rewards]

        if self.cfg.normalize_advantages and len(advantages_raw) > 1:
            adv_mean = sum(advantages_raw) / len(advantages_raw)
            adv_std = (
                sum((a - adv_mean) ** 2 for a in advantages_raw) / len(advantages_raw)
            ) ** 0.5
            advantages = [
                (a - adv_mean) / (adv_std + 1e-8) for a in advantages_raw
            ]
        else:
            advantages = advantages_raw

        # KL warmup: zero KL for the first kl_warmup_updates updates.
        kl_coef = 0.0 if self.n_updates < self.cfg.kl_warmup_updates else self.cfg.kl_beta

        total_loss_accum = 0.0
        ppo_loss_accum = 0.0
        kl_loss_accum = 0.0
        n_steps_total = 0

        for epoch in range(self.cfg.ppo_epochs):
            for traj_idx, traj in enumerate(trajectories):
                adv = advantages[traj_idx]
                if len(traj.transitions) == 0:
                    continue

                for trans in traj.transitions:
                    t_val = trans.t
                    t_tensor = torch.tensor([t_val], dtype=torch.float32, device=device)

                    x_t = trans.x_t.to(device)
                    mu_old = trans.mu_old.to(device)
                    x_next = trans.x_next.to(device)
                    sigma_sq = trans.sigma_sq
                    dt = trans.dt

                    if sigma_sq <= 0.0:
                        continue

                    # Recompute score with trainable model (with grad).
                    score_new = self._score_new(
                        species, x_t, cell, t_tensor, cutoff
                    )
                    g2_t = float(self.diffuser.g2(t_val))
                    mu_new = x_t - g2_t * score_new * dt

                    # Gaussian log-ratio (summed over atoms and dims, then mean).
                    # log π_new/π_old = (||x_next - mu_old||² - ||x_next - mu_new||²)
                    #                    / (2 * sigma_sq)
                    diff_old = x_next - mu_old  # [N, 3]
                    diff_new = x_next - mu_new  # [N, 3]

                    log_ratio = (
                        (diff_old.pow(2).sum() - diff_new.pow(2).sum())
                        / (2.0 * sigma_sq)
                    )

                    ratio = torch.exp(log_ratio)

                    # PPO clipped objective.
                    adv_t = torch.tensor(adv, dtype=torch.float32, device=device)
                    ppo_unclipped = ratio * adv_t
                    ppo_clipped = (
                        torch.clamp(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps)
                        * adv_t
                    )
                    ppo_loss = -torch.min(ppo_unclipped, ppo_clipped)

                    # KL penalty: KL(new || ref) ≈ (1/2) * ||score_new - score_ref||²
                    # (from Gaussian score-matching perspective).
                    kl_loss = torch.tensor(0.0, device=device)
                    if kl_coef > 0.0:
                        score_ref = self._score_ref(
                            species, x_t, cell, t_tensor, cutoff
                        )
                        kl_loss = 0.5 * (score_new - score_ref).pow(2).mean()

                    step_loss = ppo_loss + kl_coef * kl_loss

                    self.optimizer.zero_grad()
                    step_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.score_net.model.parameters(), self.cfg.max_grad_norm
                    )
                    self.optimizer.step()

                    total_loss_accum += float(step_loss.item())
                    ppo_loss_accum += float(ppo_loss.item())
                    kl_loss_accum += float(kl_loss.item())
                    n_steps_total += 1

        if n_steps_total > 0:
            total_loss = total_loss_accum / n_steps_total
            ppo_loss_avg = ppo_loss_accum / n_steps_total
            kl_loss_avg = kl_loss_accum / n_steps_total
        else:
            total_loss = ppo_loss_avg = kl_loss_avg = 0.0

        self.n_updates += 1

        return {
            "loss": total_loss,
            "ppo_loss": ppo_loss_avg,
            "kl_loss": kl_loss_avg,
            "reward_mean": reward_mean,
        }
