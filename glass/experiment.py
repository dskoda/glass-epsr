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
    # Defaults come from the restart-capable HPO ``glass_phys_v6_restart``
    # (2026-05-21, scripts/hpo_phys_v6.py), 100-trial study, cond-only,
    # ρ=1.5 (hardest OOD case). Best trial (#39, n_restart=3) at 5 inits × 1 seed:
    #   cond pdf_rmse=0.086, coord_emd=0.443, undercoord_le3=8.6%,
    #   undercoord_le2=0.46%, tersoff_energy_error=0.083
    # vs v5 (glass_phys_v5_15ood, 1200 trials):
    #   cond pdf_rmse=0.131, coord_emd=0.522, undercoord_le3=9.9%
    # Key changes vs v5:
    #   tmax 0.595 → 0.938; tersoff_lambda 0.281 → 0.30 (user-set);
    #   tersoff_t_gate 0.276 → 0.490; n_restart 1 → 3
    # All other params fixed at v5-best values.
    checkpoint: str = "best"  # "best", "last", or specific filename
    n_runs: int = 10
    tmin: float = 9.267e-3
    tmax: float = 0.938
    tstep: int = 256
    save_traj: bool = False
    device: str = "cuda:0"

    # Guidance parameters (for conditional generation).
    # rho=416 carried over from v5 (fixed in v6 search).
    guidance_type: Optional[str] = None  # "pdf", "adf", "xrd", "nd", "exafs", "xanes"
    rho: float = 416.0
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

    # Tersoff-guidance defaults (v6 best trial #39 + user override).
    # tersoff_lambda set to 0.30 (user preference; HPO best was 0.128).
    # t_gate 0.276 → 0.490: Tersoff active in mid-trajectory only.
    # Schedule remains sigmoid.
    tersoff_guidance: bool = True
    tersoff_lambda: float = 0.30
    tersoff_schedule: str = "sigmoid"
    tersoff_t_gate: float = 0.490
    tersoff_clamp: float = 10.0

    # Sampler refinements (v6 fixed params, carried from v5 best).
    # n_corr=2, corr_step_size=0.44, corr_t_gate=0.464, t_schedule_rho=1.01
    n_corr: int = 2
    corr_step_size: float = 0.44
    corr_use_tersoff: bool = True
    corr_t_gate: float = 0.464
    t_schedule_rho: float = 1.01

    # Simulated-annealing post-relaxation. The HPO study converged on
    # N_anneal=0 — the Langevin corrector already captures what SA would do,
    # so SA is disabled by default but the knobs remain tunable.
    # Restart: number of full denoising passes per structure.
    # n_restart=3 (default from glass_phys_v6_restart, 2026-05-21).
    # Each pass starts from the previous output (same cell/species/guidance).
    # SA tail runs only on the final pass.
    n_restart: int = 3

    # Structural-entropy guidance (ACSF variance, Cliffe et al. 2017).
    # Disabled by default; enable per-run via CLI for ablation studies.
    entropy_guidance: bool = False
    entropy_lambda: float = 1.0
    entropy_schedule: str = "constant"
    entropy_t_gate: float = 1.0
    entropy_r_cut: float = 4.0

    # Differentiable coordination-number guidance.
    # Disabled by default. Three composable penalty modes are evaluated
    # on a smooth (cosine-switched) per-atom coord:
    #   low  hinge: softplus(k_low  * (n_low  - c)) / k_low      (set w_low > 0)
    #   target:    sigma^2 * (sqrt(1 + ((c - n_target)/sigma)^2) - 1)
    #   high hinge: softplus(k_high * (c - n_high)) / k_high     (set w_high > 0)
    coord_guidance: bool = False
    coord_lambda: float = 1.0
    coord_schedule: str = "constant"
    coord_t_gate: float = 1.0
    coord_r_cut: float = 2.85
    coord_smear: float = 0.30
    coord_clamp: float = 10.0
    coord_n_target: float = 4.0
    coord_sigma_target: float = 0.5
    coord_w_target: float = 1.0
    coord_n_low: float = 4.0
    coord_w_low: float = 0.0
    coord_k_low: float = 4.0
    coord_n_high: float = 7.0
    coord_w_high: float = 0.0
    coord_k_high: float = 4.0

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
