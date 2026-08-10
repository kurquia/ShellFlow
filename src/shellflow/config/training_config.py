"""
Configuration dataclasses for PyTorch Lightning training.

The RFM pipeline (RiemannianDiTsModule) consumes only OptimizerConfig,
SchedulerConfig, and TrainingConfig — constructed from plain dicts built by
scripts/train_riemannian_DiTs.py, and re-instantiated from checkpoint hparams
on load. Fields that the RFM pipeline no longer reads are retained so that
`TrainingConfig(**old_hparams_dict)` keeps working for existing checkpoints.

The remaining dataclasses (ModelConfig, ExperimentConfig, ...) are legacy
YAML/CLI tooling from earlier model families and have no current consumers.
"""

import copy
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Literal, Optional

import yaml

# =============================================================================
# Sub-configuration dataclasses
# =============================================================================


@dataclass
class ModelConfig:
    """Legacy model-architecture configuration (pre-RFM model families).

    Not used by the RFM pipeline — the training script passes a plain
    model-config dict to RiemannianDiTsModule instead.
    """

    d_model: int = 128
    num_heads: int = 8
    num_layers: int = 6
    max_feature_types: int = 73
    max_particles: int = 30
    mlp_ratio: float = 4.0

    # BERT-specific
    n_bins: int = 1
    use_cls_token: bool = False

    # Diffusion-specific
    num_timesteps: int = 1000
    dropout: float = 0.1
    # Particle-vector diffusion
    architecture: str = "token"  # "token" (legacy) or "particle_vector"
    cartesian_stats: str | None = None  # path to cartesian_stats.yaml
    prediction_type: str = "epsilon"  # "epsilon" or "v_prediction"
    min_snr_gamma: float | None = None  # None=disabled, 5.0=recommended
    ema_decay: float | None = None  # None=disabled, 0.9999=recommended
    component_loss_weights: list[float] | str | None = (
        "auto"  # "auto"=derive from cartesian_stats, null=uniform, [1,1,2.9,2.6]=manual
    )
    met_embed_dim: int = 64  # MET conditioning projection dimension
    num_met_tokens: int = 0  # 0 = additive MET conditioning, >0 = cross-attention with N learned context tokens
    num_extra_features: int = 5  # Number of per-particle auxiliary conditioning features (5=data_05, 13=data_06)
    cross_attn_gate: bool = False  # Tanh gate on MET cross-attention (False=fully open, True=zero-init ramp-up)
    timestep_sampling: str = "logit_normal"  # "uniform" or "logit_normal"
    multiplicative_attention: bool = False  # q⊙(attn@v) branch for pairwise products
    pair_bias_type: str = "factored"  # "factored" (rank-1 outer product) or "mlp" (MLP on concatenated pairs)
    phi_augmentation: bool = (
        False  # Random φ-rotation augmentation (exact symmetry in transverse plane)
    )
    minkowski_loss_weight: float | None = (
        None  # Pairwise Minkowski dot product loss (null to disable)
    )
    minkowski_balanced: bool = (
        True  # Category-balanced normalization (by particle type + charge)
    )
    onshell_loss_weight: float | None = (
        None  # Per-particle mass-shell loss (null to disable)
    )
    onshell_loss_type: str = "huber"  # "huber" (smooth L1, recommended) or "mse"
    schedule_lr_multiplier: float = 1.0  # LR multiplier for noise schedule parameters
    noise_correlation: float = (
        0.0  # Inter-particle noise correlation ρ (0 = independent)
    )
    noise_correlation_type: str = "global"  # "global" or "pairwise"
    self_cond_prob: float = (
        0.8  # Self-conditioning probability during training (0 = disabled)
    )
    cond_drop_prob: float = (
        0.0  # CFG: conditioning dropout probability (0.1 = 10% unconditional)
    )
    guidance_scale: float = 1.0  # CFG: guidance scale at inference (1.0 = no guidance)
    ot_cfm: bool = False  # OT-CFM: optimal transport conditional flow matching
    # Generic auxiliary pairwise losses (alternative/supplement to Minkowski)
    auxiliary_loss_type: str | None = (
        None  # "swd" | "random_gram" | "covariance" | null
    )
    auxiliary_loss_weight: float = 0.5  # Weight for the auxiliary loss term
    swd_n_projections: int = 32  # Number of random projections (SWD + random_gram)
    swd_alpha_threshold: float = 0.3  # Min alpha_bar for SWD pair inclusion
    # End-to-end DDIM fine-tuning (backprop through full sampling chain)
    e2e_loss_weight: float = 0.0  # 0 = disabled, >0 = weight for e2e loss
    e2e_loss_type: str = "mse"  # "mse" (per-element) or "minkowski" (gram matrix)
    e2e_ddim_steps: int = 25  # DDIM steps in differentiable chain
    e2e_every_k_steps: int = 10  # Run e2e every K training steps
    e2e_batch_size: int = 64  # Sub-batch for e2e (sampled from main B=1024)
    e2e_gradient_checkpointing: bool = (
        True  # Recompute per-step activations during backward
    )
    e2e_only: bool = False  # True = drop recon loss, train purely on e2e

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""

    name: Literal["adamw", "adam", "sgd"] = "adamw"
    lr: float = 1e-3
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    momentum: float = 0.9  # SGD-specific
    nesterov: bool = False  # SGD-specific


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""

    name: Literal["onecycle", "cosine", "linear", "constant", "reduce_on_plateau"] = (
        "onecycle"
    )
    warmup_pct: float = 0.05
    warmup_steps: Optional[int] = None
    anneal_strategy: Literal["linear", "cos"] = "cos"
    min_lr: float = 1e-6
    patience: int = 5  # ReduceLROnPlateau
    factor: float = 0.5  # ReduceLROnPlateau


@dataclass
class DataConfig:
    """Data loading and processing configuration."""

    config_path: str = "src/shellflow/config/dataloader_config.yaml"
    train_split: float = 0.9
    batch_size: int = 512
    num_workers: int = 8
    lazy_loading: bool = True
    normalize: bool = True
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 3
    mask_fraction: Optional[float] = None
    masking_probability: Optional[float] = None


@dataclass
class TrainingConfig:
    """Training loop configuration.

    RiemannianDiTsModule stores this in its hparams for provenance but does
    not read individual fields (epochs/grad_clip/seed are consumed by the
    training script directly from YAML). The loss-specific fields below are
    legacy but retained so old checkpoint hparams still construct (see
    module docstring).
    """

    epochs: int = 50
    grad_clip: float = 1.0
    seed: int = 42
    log_every_n_steps: int = 50

    # Loss function
    loss_type: Literal["mse", "huber", "mmd", "cross_entropy"] = "mse"
    huber_delta: float = 1.0
    bin_min: float = -1.0
    bin_max: float = 1.0

    # MMD loss parameters
    mmd_weight: float = 0.1
    mmd_mse_weight: float = 1.0
    mmd_sigma: float | list[float] | None = None
    mmd_n_rff_features: int = 256
    mmd_use_huber: bool = False
    mmd_adaptive_sigma: bool = True
    mmd_multiscale: bool = True

    # torch.compile
    compile_model: bool = True
    compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "default"

    # Test mode
    test_mode: bool = False
    test_mode_samples: int = 1000


@dataclass
class WandbConfig:
    """Weights & Biases configuration."""

    enabled: bool = True
    project: str = "shellflow"
    name: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    run_id: Optional[str] = None
    save_code: bool = True
    log_model: bool = False


@dataclass
class FinetuneConfig:
    """Finetuning-specific configuration."""

    pretrained_checkpoint: Optional[str] = None
    freeze_encoder: bool = False


@dataclass
class ResumeConfig:
    """Resume training configuration."""

    checkpoint_path: Optional[str] = None


@dataclass
class CheckpointConfig:
    """Checkpointing configuration."""

    dir: str = "checkpoints/bert"
    save_every_n_epochs: int = 5
    save_top_k: int = 3


@dataclass
class EarlyStoppingConfig:
    """Early stopping configuration."""

    enabled: bool = False
    patience: int = 10
    min_delta: float = 0.0001


# =============================================================================
# CLI argument to config field mapping
# =============================================================================

# Maps CLI argument names to (section, field_name) tuples
# Arguments not in this map but matching a field name are auto-mapped
ARG_MAPPING = {
    # Renamed/special mappings only
    "checkpoint_dir": ("checkpoint", "dir"),
    "early_stopping": ("early_stopping", "enabled"),
    "early_stopping_patience": ("early_stopping", "patience"),
    "early_stopping_min_delta": ("early_stopping", "min_delta"),
    "wandb_project": ("wandb", "project"),
    "wandb_name": ("wandb", "name"),
    "wandb_id": ("wandb", "run_id"),
    "wandb_tags": ("wandb", "tags"),
    "no_wandb": ("wandb", "enabled"),  # Inverted flag
    "resume_from": ("resume", "checkpoint_path"),
}

# Flags that should be inverted when applied
INVERTED_FLAGS = {"no_wandb"}


# =============================================================================
# Main experiment configuration
# =============================================================================


@dataclass
class ExperimentConfig:
    """Complete experiment configuration combining all sub-configs."""

    mode: Literal["pretrain", "finetune"] = "pretrain"
    device: Literal["cuda", "cpu", "auto"] = "auto"
    precision: Literal["32", "16-mixed", "bf16-mixed"] = "bf16-mixed"

    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)
    resume: ResumeConfig = field(default_factory=ResumeConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)

    # Section name to config class mapping for introspection
    _SECTIONS = {
        "model": ModelConfig,
        "optimizer": OptimizerConfig,
        "scheduler": SchedulerConfig,
        "data": DataConfig,
        "training": TrainingConfig,
        "wandb": WandbConfig,
        "finetune": FinetuneConfig,
        "resume": ResumeConfig,
        "checkpoint": CheckpointConfig,
        "early_stopping": EarlyStoppingConfig,
    }

    def to_dict(self) -> dict:
        """Convert to flat dictionary for W&B logging."""
        result = {"mode": self.mode, "device": self.device, "precision": self.precision}
        for section_name in self._SECTIONS:
            for key, value in asdict(getattr(self, section_name)).items():
                result[f"{section_name}/{key}"] = value
        return result

    def to_nested_dict(self) -> dict:
        """Convert to nested dictionary (YAML-safe)."""

        def convert_tuples(obj):
            if isinstance(obj, dict):
                return {k: convert_tuples(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_tuples(item) for item in obj]
            return obj

        result = {"mode": self.mode, "device": self.device, "precision": self.precision}
        for section_name in self._SECTIONS:
            result[section_name] = convert_tuples(asdict(getattr(self, section_name)))
        return result

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentConfig":
        """Create config from dictionary."""
        # Handle optimizer betas conversion from list to tuple
        optimizer_data = data.get("optimizer", {})
        if "betas" in optimizer_data and isinstance(optimizer_data["betas"], list):
            optimizer_data = {**optimizer_data, "betas": tuple(optimizer_data["betas"])}

        return cls(
            mode=data.get("mode", "pretrain"),
            device=data.get("device", "auto"),
            precision=data.get("precision", "32"),
            model=ModelConfig(**data.get("model", {})),
            optimizer=OptimizerConfig(**optimizer_data),
            scheduler=SchedulerConfig(**data.get("scheduler", {})),
            data=DataConfig(**data.get("data", {})),
            training=TrainingConfig(**data.get("training", {})),
            wandb=WandbConfig(**data.get("wandb", {})),
            finetune=FinetuneConfig(**data.get("finetune", {})),
            resume=ResumeConfig(**data.get("resume", {})),
            checkpoint=CheckpointConfig(**data.get("checkpoint", {})),
            early_stopping=EarlyStoppingConfig(**data.get("early_stopping", {})),
        )

    def save_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        with open(path, "w") as f:
            yaml.dump(
                self.to_nested_dict(), f, default_flow_style=False, sort_keys=False
            )

    def override_from_args(self, args) -> "ExperimentConfig":
        """Override config values from argparse namespace."""
        config = copy.deepcopy(self)

        # Build reverse lookup: field_name -> section_name
        field_to_section = {}
        for section_name, section_cls in self._SECTIONS.items():
            for f in fields(section_cls):
                field_to_section[f.name] = section_name

        # Process each argument
        for arg_name in dir(args):
            if arg_name.startswith("_"):
                continue
            value = getattr(args, arg_name, None)
            if value is None:
                continue

            # Check explicit mapping first
            if arg_name in ARG_MAPPING:
                section_name, field_name = ARG_MAPPING[arg_name]
                if arg_name in INVERTED_FLAGS:
                    value = not value
            # Try to auto-map by field name
            elif arg_name in field_to_section:
                section_name, field_name = field_to_section[arg_name], arg_name
            # Check top-level attributes
            elif arg_name in ("mode", "device", "precision"):
                setattr(config, arg_name, value)
                continue
            else:
                continue

            setattr(getattr(config, section_name), field_name, value)

        return config


def create_default_config_yaml(path: str | Path) -> None:
    """Create a default configuration YAML file."""
    ExperimentConfig().save_yaml(path)
