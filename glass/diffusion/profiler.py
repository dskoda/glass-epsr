"""Guidance contribution profiler for denoise_by_sde.

Records per-step, per-component score vectors so callers can audit how much
each guidance term (prior, Tersoff, likelihood, entropy, coordination) is
contributing to the atomic displacements over time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from torch import Tensor


@dataclass
class StepRecord:
    """Snapshot of score contributions at one predictor step."""

    step: int
    t: float
    # Effective displacement = g2(t) * score * |dt|  (Å, approximately)
    prior_score: Tensor               # [N, 3]  raw score_fn output
    tersoff_score: Optional[Tensor]   # [N, 3]  lam * guidance_vec, or None
    likelihood_score: Optional[Tensor]  # [N, 3]  l_score, or None
    entropy_score: Optional[Tensor]   # [N, 3]  lam_e * e_vec, or None
    coord_score: Optional[Tensor]     # [N, 3]  lam_co * c_vec, or None
    total_disp: Tensor                # [N, 3]  actual displacement applied


@dataclass
class GuidanceProfiler:
    """Accumulates StepRecords from denoise_by_sde and provides analysis helpers."""

    records: List[StepRecord] = field(default_factory=list)

    def record(
        self,
        step: int,
        t: float,
        prior_score: Tensor,
        total_disp: Tensor,
        tersoff_score: Optional[Tensor] = None,
        likelihood_score: Optional[Tensor] = None,
        entropy_score: Optional[Tensor] = None,
        coord_score: Optional[Tensor] = None,
    ) -> None:
        def _clone(x):
            return x.detach().cpu().clone() if x is not None else None

        self.records.append(
            StepRecord(
                step=step,
                t=t,
                prior_score=_clone(prior_score),
                tersoff_score=_clone(tersoff_score),
                likelihood_score=_clone(likelihood_score),
                entropy_score=_clone(entropy_score),
                coord_score=_clone(coord_score),
                total_disp=_clone(total_disp),
            )
        )

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def ts(self) -> List[float]:
        return [r.t for r in self.records]

    def _per_step_rms(self, attr: str) -> List[Optional[float]]:
        """RMS norm of the score tensor (over atoms and xyz) at each step."""
        out = []
        for r in self.records:
            v = getattr(r, attr)
            if v is None:
                out.append(None)
            else:
                out.append(float(v.norm() / (v.numel() ** 0.5)))
        return out

    def score_rms(self) -> Dict[str, List[Optional[float]]]:
        """Per-step RMS of each score component (not weighted by g2*dt)."""
        return {
            "prior": self._per_step_rms("prior_score"),
            "tersoff": self._per_step_rms("tersoff_score"),
            "likelihood": self._per_step_rms("likelihood_score"),
            "entropy": self._per_step_rms("entropy_score"),
            "coord": self._per_step_rms("coord_score"),
            "total_disp": self._per_step_rms("total_disp"),
        }

    def per_atom_disp_norms(self) -> Dict[str, Tensor]:
        """Per-atom L2 norm of total displacement summed over all steps.

        Returns dict mapping component name → Tensor [N] of cumulative
        displacement magnitudes in the same units as total_disp (Å).

        Components whose score is always None return a zero tensor.
        """
        if not self.records:
            return {}
        N = self.records[0].total_disp.shape[0]

        def _accumulate(attr):
            acc = torch.zeros(N)
            for r in self.records:
                v = getattr(r, attr)
                if v is not None:
                    acc += v.norm(dim=-1)
            return acc

        return {
            "prior": _accumulate("prior_score"),
            "tersoff": _accumulate("tersoff_score"),
            "likelihood": _accumulate("likelihood_score"),
            "entropy": _accumulate("entropy_score"),
            "coord": _accumulate("coord_score"),
            "total_disp": _accumulate("total_disp"),
        }

    def summary_table(self) -> str:
        """Human-readable table of per-step RMS scores."""
        rms = self.score_rms()
        ts = self.ts()
        components = ["prior", "tersoff", "likelihood", "entropy", "coord", "total_disp"]

        header = f"{'step':>5}  {'t':>8}  " + "  ".join(f"{c:>12}" for c in components)
        lines = [header, "-" * len(header)]
        for idx, (step_rec, t_val) in enumerate(zip(self.records, ts)):
            row = f"{step_rec.step:>5}  {t_val:>8.5f}  "
            cells = []
            for c in components:
                v = rms[c][idx]
                cells.append(f"{v:>12.5f}" if v is not None else f"{'—':>12}")
            lines.append(row + "  ".join(cells))
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialise to a JSON-serialisable dict."""
        rms = self.score_rms()
        return {
            "ts": self.ts(),
            "score_rms": {k: [v if v is not None else None for v in vs] for k, vs in rms.items()},
            "per_atom_disp_norms": {
                k: v.tolist() for k, v in self.per_atom_disp_norms().items()
            },
            "n_steps": len(self.records),
        }

    def save_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
