"""Unified experiment management for glass training and generation.

This module provides a simplified interface for managing machine learning experiments,
organizing data, checkpoints, logs, and outputs in a single directory structure.
"""

import os
import glob
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, asdict, field


@dataclass
class ExperimentConfig:
    """Configuration for a glass experiment.
    
    This dataclass stores all parameters needed for training or generation,
    with sensible defaults for most options.
    """
    # Experiment metadata
    name: str = "experiment"
    model_type: str = "score"  # "score" or "spec"
    
    # Model architecture
    num_species: int = 1
    num_convs: int = 5
    dim: int = 200
    ema_decay: float = 0.9999
    
    # Training parameters
    max_epochs: int = 12000
    batch_size: int = 1
    learning_rate: float = 1e-3
    cutoff: float = 5.0
    k: float = 0.8
    dup: int = 4
    train_size: float = 0.9
    scale_y: float = 1.0
    num_workers: int = 8
    
    # Spec model specific
    spec_type: Optional[str] = None  # "exafs" or "xanes"
    out_dim: Optional[int] = None  # 400 for exafs, 100 for xanes
    
    # Data paths (relative to experiment root)
    data_dir: str = "data"
    
    # Checkpoint settings
    save_top_k: int = 3
    
    # Generation parameters (for inference).
    # Defaults come from the multi-density HPO ``glass_unified_v3_ood``
    # (2026-05-18, scripts/hpo_unified.py), run AFTER the Phase E Tersoff
    # implementation bug was fixed. Top-5 consensus across 200 trials.
    # Best trial replay (5 inits × 5 seeds × 3 densities):
    #   cond pdf_rmse=0.028, coord_emd=0.173, adf_rmse=0.060
    # The PDF is 33 % better than v2_ood (0.042) at ρ=1.5/3.5; coord is
    # essentially unchanged.
    checkpoint: str = "best"  # "best", "last", or specific filename
    n_runs: int = 10
    tmin: float = 4e-4
    tmax: float = 0.834
    tstep: int = 512
    save_traj: bool = False
    device: str = "cuda:0"

    # Guidance parameters (for conditional generation).
    # v3_ood: with the Tersoff fix in place, the optimizer converges on
    # rho ≈ 600-1100 (median 737) — ~3× higher than v2_ood's 240. The fix
    # made Tersoff produce ~3-180 meV/atom corrections that the prior had
    # been silently absorbing; with that absorbed, the likelihood term can
    # safely run hotter. At in-distribution (ρ=2.5) cond pdf=0.012, in
    # line with v1's 0.013.
    guidance_type: Optional[str] = None  # "pdf", "adf", "xrd", "nd", "exafs", "xanes"
    rho: float = 737.0
    ref_path: Optional[str] = None
    exp_data: Optional[str] = None
    spec_model_path: Optional[str] = None

    # Guidance-specific parameters
    bin_size: int = 100  # PDF
    angle_bins: int = 100  # ADF
    adf_sigma: float = 0.1  # ADF
    adf_cutoff: float = 3.5  # ADF
    element_names: list = field(default_factory=list)  # XRD/ND
    qmin: float = 1.0  # XRD/ND
    qmax: float = 20.0  # XRD/ND
    qstep: float = 0.1  # XRD/ND
    biso: float = 1.5  # XRD/ND

    # Tersoff-guidance defaults (v3_ood top-5 consensus).
    # tersoff_lambda 0.20 → 0.23 (slight increase). Schedule converges on
    # sigmoid (mode 2/5 in top-5; linear is competitive). t_gate dropped
    # from 0.45 to 0.15 — Tersoff now activates earlier in the trajectory
    # rather than late. With the fixed potential the angular term carries
    # more weight, so it makes sense to apply it during structural
    # decisions rather than just at low noise.
    tersoff_guidance: bool = True
    tersoff_lambda: float = 0.23
    tersoff_schedule: str = "sigmoid"
    tersoff_t_gate: float = 0.15
    tersoff_clamp: float = 10.0

    # Sampler refinements (v3_ood top-5 consensus).
    # n_corr 2 → 1 (top-5 was unanimous on n_corr=1). With the fixed
    # Tersoff and stronger rho, the predictor step alone is well-behaved
    # and the corrector's polishing role is no longer load-bearing.
    # corr_step_size 0.12 → 0.30 (much larger; the corrector still runs
    # but does fewer, larger steps). corr_t_gate 0.58 → 0.79 (corrector
    # active over a wider t range, including high noise).
    # t_schedule_rho 1.35 → 0.98 (back near 1.0 = uniform). The fixed
    # Tersoff term makes the high-noise regime more informative, so
    # concentrating steps near t=0 is no longer needed.
    n_corr: int = 1
    corr_step_size: float = 0.30
    corr_use_tersoff: bool = True
    corr_t_gate: float = 0.79
    t_schedule_rho: float = 0.98

    # Simulated-annealing post-relaxation. The HPO study converged on
    # N_anneal=0 — the Langevin corrector already captures what SA would do,
    # so SA is disabled by default but the knobs remain tunable.
    sa_n_steps: int = 0
    sa_T0: float = 1e-2
    sa_T_end: float = 1e-5
    sa_lr: float = 1e-3
    sa_lr_clamp: float = 0.2

    # Infrastructure
    accelerator: str = "gpu"
    strategy: str = "ddp_find_unused_parameters_true"
    matmul_precision: str = "medium"
    refresh_rate: int = 10
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ExperimentConfig":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)
    
    def update(self, **kwargs) -> "ExperimentConfig":
        """Update configuration with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


class Experiment:
    """Manages experiment directory structure and resources.
    
    An experiment is organized as:
        experiment_root/
        ├── config.yaml          # Experiment configuration
        ├── data/                # Training data (.xyz files)
        ├── checkpoints/         # Model checkpoints
        │   ├── best.ckpt
        │   ├── last.ckpt
        │   └── epoch_*.ckpt
        ├── inits/               # Initial structures for generation
        ├── outputs/             # Generated structures
        └── logs/                # TensorBoard logs
            └── version_*/
    
    Args:
        root: Path to experiment directory (created if doesn't exist)
        create: If True, create directory structure on init
    """
    
    def __init__(self, root: Union[str, Path], create: bool = False):
        self.root = Path(root).resolve()
        self.config_path = self.root / "config.yaml"
        
        # Define subdirectories
        self.data_dir = self.root / "data"
        self.checkpoints_dir = self.root / "checkpoints"
        self.inits_dir = self.root / "inits"
        self.outputs_dir = self.root / "outputs"
        self.logs_dir = self.root / "logs"
        
        # Config cache
        self._config: Optional[ExperimentConfig] = None
        
        if create:
            self._create_structure()
    
    def _create_structure(self) -> None:
        """Create experiment directory structure."""
        self.root.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.inits_dir.mkdir(exist_ok=True)
        self.outputs_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
    
    @property
    def exists(self) -> bool:
        """Check if experiment directory exists."""
        return self.root.exists()
    
    def load_config(self) -> ExperimentConfig:
        """Load configuration from config.yaml."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        self._config = ExperimentConfig.from_yaml(self.config_path)
        return self._config
    
    def save_config(self, config: Optional[ExperimentConfig] = None) -> None:
        """Save configuration to config.yaml."""
        if config is None:
            config = self._config
        if config is None:
            raise ValueError("No config to save")
        self.root.mkdir(parents=True, exist_ok=True)
        config.to_yaml(self.config_path)
        self._config = config
    
    def get_config(self) -> ExperimentConfig:
        """Get cached or loaded config."""
        if self._config is None:
            return self.load_config()
        return self._config
    
    def find_checkpoint(self, checkpoint: str = "best") -> Path:
        """Find checkpoint file.
        
        Args:
            checkpoint: "best", "last", or specific filename/pattern
            
        Returns:
            Path to checkpoint file
            
        Raises:
            FileNotFoundError: If no matching checkpoint found
        """
        if checkpoint == "best":
            path = self.checkpoints_dir / "best.ckpt"
            if path.exists():
                return path
            # Fallback: find checkpoint with "best" in name
            candidates = sorted(self.checkpoints_dir.glob("*best*.ckpt"))
            if candidates:
                return candidates[-1]
            # Final fallback: any checkpoint sorted by name
            candidates = sorted(self.checkpoints_dir.glob("*.ckpt"))
            if candidates:
                return candidates[-1]
                
        elif checkpoint == "last":
            path = self.checkpoints_dir / "last.ckpt"
            if path.exists():
                return path
            # Fallback: find checkpoint with "last" in name
            candidates = sorted(self.checkpoints_dir.glob("*last*.ckpt"))
            if candidates:
                return candidates[-1]
            # Final fallback: any checkpoint sorted by name
            candidates = sorted(self.checkpoints_dir.glob("*.ckpt"))
            if candidates:
                return candidates[-1]
        else:
            # Specific filename or pattern
            path = self.checkpoints_dir / checkpoint
            if path.exists():
                return path
            # Try as glob pattern
            candidates = sorted(self.checkpoints_dir.glob(checkpoint))
            if candidates:
                return candidates[-1]
        
        raise FileNotFoundError(
            f"No checkpoint found for '{checkpoint}' in {self.checkpoints_dir}"
        )
    
    def list_checkpoints(self) -> list:
        """List all available checkpoints."""
        return sorted(self.checkpoints_dir.glob("*.ckpt"))
    
    def get_init_files(self, init_dir: Optional[Union[str, Path]] = None) -> list:
        """Get list of initialization .xyz files.
        
        Args:
            init_dir: Override directory (default: self.inits_dir)
            
        Returns:
            List of Path objects to .xyz files
        """
        search_dir = Path(init_dir) if init_dir else self.inits_dir
        if not search_dir.exists():
            return []
        return sorted(search_dir.glob("*.xyz"))
    
    def get_data_files(self) -> list:
        """Get list of training data .xyz files.
        
        Searches in order:
        1. data_dir/*.xyz (flat structure)
        2. data_dir/structures/*.xyz (old structure)
        3. data_dir/structures/train/*.xyz (old structure)
        
        Returns:
            List of Path objects to .xyz files
        """
        # Flat structure (preferred)
        files = sorted(self.data_dir.glob("*.xyz"))
        if files:
            return files
        
        # Old structure fallbacks
        files = sorted(self.data_dir.glob("structures/*.xyz"))
        if files:
            return files
            
        files = sorted(self.data_dir.glob("structures/train/*.xyz"))
        return files
    
    def get_data_dir_for_datamodule(self) -> str:
        """Get data directory path suitable for StructureSpecDataModule.
        
        The datamodule expects a directory that may contain subdirectories
        for different data types. We return the data_dir with trailing slash.
        """
        return str(self.data_dir) + "/"
    
    def __repr__(self) -> str:
        return f"Experiment('{self.root}')"
