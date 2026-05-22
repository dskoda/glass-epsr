# RLPF: Removed (Phase H)

## What was removed

- `glass/diffusion/rlpf.py` — `RLPFConfig`, `Transition`, `Trajectory`, `RLPFTrainer` (PPO-style policy gradient fine-tuning)
- `glass/diffusion/rewards.py` — `RewardConfig`, `TersoffPDFReward` (Tersoff energy + PDF RMSE reward signal)
- `scripts/rlpf_finetune.py` — CLI script driving RLPF training loops
- `tests/test_rlpf.py` — 9 unit tests covering rollout collection, PPO updates, KL warmup, and subsample logic

The research experiment directory `research/density_extrapolation/experiments/phase_h_rlpf/`
is retained as an archive.

## Why RLPF was tried

RLPF (RL with Physical Feedback) was a PPO-style fine-tuning approach: collect
reverse-SDE rollout trajectories with the EMA model as reference policy, compute
a terminal reward combining Tersoff energy and PDF RMSE, then apply clipped-ratio
PPO updates with a KL divergence penalty against the frozen reference.

The goal was to improve out-of-distribution (ρ = 1.5) generation quality after the
standard HPO-tuned sampler still showed poor coordination numbers at low density.

## Why it was removed (Phase H verdict)

Both Phase H-A (plain rollouts) and Phase H-B (PDF-guided rollouts) were run at
ρ = 1.5 with 50–55 structures across four reward arms. Results:

| | pdf_rmse | coord_emd |
|---|---|---|
| RLPF (any arm) | ~0.66 | ~0.86 |
| v3_ood baseline | ~0.86 | ~0.64 |

RLPF significantly **worsened** coordination (coord_emd 0.64 → 0.86) while
modestly improving PDF RMSE. The reward did not include a coordination term, so
the policy optimised the wrong objective. The KL penalty bound the policy within
~1.5 KL units of the ρ=2.5 prior — enough to prevent catastrophic forgetting but
not enough to shift coordination away from the in-distribution 4-coordinated
geometry.

Adding PDF guidance during rollout collection (Phase H-B, rho=737) changed all
metrics by ≤0.01 — the KL constraint remained binding regardless of rollout quality.

## Root cause

The pretrained score model is anchored to ρ = 2.5 (4-coordinated Si). At ρ = 1.5
the target has mostly 3-coordinated atoms. The KL penalty prevents the model from
moving far enough from the ρ=2.5 prior to shift coordination. Short rollouts
(tstep=64) provide insufficient structural exploration. The reward signal (energy +
PDF) is correlated at ρ=1.5 and neither term provides independent coordination signal.

## Paths not taken

The following variants were not explored and may still be worth attempting in future:

- RLPF with a coordination-EMD reward term
- RLPF with longer rollouts (tstep=512)
- A much weaker KL penalty (with warmup from a density-adapted checkpoint)
- RLPF at in-distribution density (ρ=2.5) where the prior is already well-calibrated

Full results and mechanistic discussion: `research/density_extrapolation/experiments/phase_h_rlpf/README.md`
