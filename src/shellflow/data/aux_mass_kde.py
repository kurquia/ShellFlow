"""K-body invariant-mass KDE helpers for the density-aware aux multiplier.

Used by:
  - The lightning module's _build_density_aware_buffers (inline at __init__).
  - scripts/compute_aux_mass_kde.py (CLI for ad-hoc inspection / .pt export).

Both call the same helpers so the offline plot and the in-training KDE are
bit-identical given identical hyperparameters.
"""

import numpy as np
import torch
from scipy.stats import gaussian_kde
from torch.utils.data import DataLoader, Subset

from shellflow.data.particle_dataset import ParticleDataset
from shellflow.data.particle_vector_collator import ParticleVectorCollator
from shellflow.data.tuple_categories import tuple_keys
from shellflow.utils.logger_config import get_logger

logger = get_logger(__name__)

# Density floor for the KDE grid: floor log p at this many orders of magnitude
# below the peak. Bounds the per-category log-density dynamic range (and hence
# |log p_truth − log p_gen|) and guarantees a finite floor even for narrow,
# peaked categories whose median grid density underflows to 0.
_LOG_FLOOR_DECADES = 6.0


def all_kbody_log_masses(
    kin: torch.Tensor,
    mask: torch.Tensor,
    K: int,
    max_pairs_per_event: int | None = None,
    types: torch.Tensor | None = None,
    charges: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Vectorized K-body log invariant mass for all active K-tuples in a batch.

    Args:
        kin:  (B, N, 4) float — (px, py, pz, E)
        mask: (B, N)    float — 1 for active, 0 for padding
        K:    int — tuple size
        max_pairs_per_event: cap on tuples kept per event (None = all)
        types:   (B, N) int — particle type codes (for category keys; optional)
        charges: (B, N) int — collator charge indices (for category keys)

    Returns:
        (log_m, key) where
          log_m: (M_total,) float64 — log invariant mass for active K-tuples
          key:   (M_total,) int64 category key per tuple, or None when
                 types/charges are not provided.
    """
    _, N, _ = kin.shape
    if K > N:
        empty = torch.empty(0, dtype=torch.float64)
        return empty, (None if types is None else torch.empty(0, dtype=torch.int64))

    combs = torch.combinations(torch.arange(N, device=kin.device), r=K)
    if max_pairs_per_event is not None and combs.shape[0] > max_pairs_per_event:
        idx = torch.randperm(combs.shape[0], device=kin.device)[:max_pairs_per_event]
        combs = combs[idx]

    active = mask > 0.5
    tuple_active = active[:, combs].all(dim=-1)

    kin_d = kin.double()
    px_sum = kin_d[..., 0][:, combs].sum(dim=-1)
    py_sum = kin_d[..., 1][:, combs].sum(dim=-1)
    pz_sum = kin_d[..., 2][:, combs].sum(dim=-1)
    E_sum = kin_d[..., 3][:, combs].sum(dim=-1)

    m2 = E_sum.pow(2) - px_sum.pow(2) - py_sum.pow(2) - pz_sum.pow(2)
    m2 = m2.clamp(min=1e-8)
    log_m = 0.5 * torch.log(m2)

    key = None
    if types is not None and charges is not None:
        # (B, C_K, K) → (B, C_K) permutation-invariant category key
        key_full = tuple_keys(types[:, combs], charges[:, combs], K)
        key = key_full[tuple_active].cpu()

    return log_m[tuple_active].cpu(), key


def collect_log_masses(
    config_path: str,
    K_max: int,
    max_events: int,
    batch_size: int = 4096,
    num_workers: int = 4,
    max_particles: int = 10,
    max_pairs_per_event: int | None = None,
) -> dict[int, np.ndarray]:
    """Walk the training dataset, collect log m for all K ∈ {2, ..., K_max}."""
    dataset = ParticleDataset(
        config_path=config_path,
        normalize=False,
        lazy_loading=True,
    )
    N_total = len(dataset)
    logger.info(f"Dataset has {N_total:,} events")

    n_use = min(N_total, max_events)
    if n_use < N_total:
        dataset = Subset(dataset, list(range(n_use)))
        logger.info(f"  Capped to first {n_use:,} events for speed")

    collator = ParticleVectorCollator(max_particles=max_particles)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=False,
    )

    log_m_per_K = {K: [] for K in range(2, K_max + 1)}
    n_events = 0
    for i, batch in enumerate(loader):
        kin = batch["kinematics"]
        mask = batch["particle_mask"]
        for K in range(2, K_max + 1):
            log_m, _ = all_kbody_log_masses(kin, mask, K, max_pairs_per_event)
            if log_m.numel() > 0:
                log_m_per_K[K].append(log_m)
        n_events += kin.shape[0]
        if (i + 1) % 50 == 0:
            logger.info(
                f"  {i + 1} batches ({n_events:,} events); "
                + ", ".join(
                    f"K={K}: {sum(t.numel() for t in log_m_per_K[K]):,} tuples"
                    for K in range(2, K_max + 1)
                )
            )

    out = {}
    for K in range(2, K_max + 1):
        if log_m_per_K[K]:
            out[K] = torch.cat(log_m_per_K[K]).numpy().astype(np.float64)
        else:
            out[K] = np.empty(0, dtype=np.float64)
        logger.info(f"K={K}: collected {len(out[K]):,} log-mass values")
    return out


def fit_kde_grid(
    log_m_values: np.ndarray,
    bandwidth: float,
    grid_size: int,
    grid_lo: float,
    grid_hi: float,
    max_kde_samples: int,
    seed: int = 42,
) -> dict | None:
    """Fit a Gaussian KDE on log m and evaluate on a uniform grid in log m.

    Returns a dict with `log_m_grid` and `log_p_grid` as float32 tensors plus
    `log_p_floor` and `log_p_ref` as plain floats. None if no samples.
    """
    if len(log_m_values) == 0:
        return None

    # gaussian_kde inverts a covariance via Cholesky; a (near-)constant input
    # (a category whose log-masses are all equal — e.g. a collapsed generated
    # category, or all values pinned at the m² clamp floor) is singular and
    # raises LinAlgError/ValueError. Guard it so neither the truth-KDE build nor
    # the gen-KDE refresh can crash a rank on a degenerate category.
    if float(np.std(log_m_values)) < 1e-6 or np.unique(log_m_values).size < 2:
        logger.warning(
            f"  KDE input near-constant (n={len(log_m_values)}, "
            f"std={float(np.std(log_m_values)):.2e}); skipping this category"
        )
        return None

    if len(log_m_values) > max_kde_samples:
        rng = np.random.default_rng(seed=seed)
        idx = rng.choice(len(log_m_values), size=max_kde_samples, replace=False)
        log_m_values = log_m_values[idx]
        logger.info(f"  Subsampled KDE input → {max_kde_samples:,}")

    try:
        kde = gaussian_kde(log_m_values, bw_method=bandwidth)
    except (np.linalg.LinAlgError, ValueError) as e:
        logger.warning(f"  gaussian_kde failed (n={len(log_m_values)}): {e}; skipping")
        return None
    log_m_grid = np.linspace(grid_lo, grid_hi, grid_size, dtype=np.float64)
    log_p_grid = kde.logpdf(log_m_grid)

    # PEAK-relative floor (robust to narrow / peaked per-category densities).
    # A median-relative floor (log(median)-log(100)) underflows to -inf for
    # narrow categories where >half the grid has p≈0 (e.g. a dilepton or
    # jet-pair KDE concentrated in a sub-range of the −3..6 log-mass grid).
    # A -inf floor disables flooring → raw kde.logpdf keeps -inf at out-of-band
    # masses → w* = exp(logp_truth − logp_gen) = exp(-inf − (-inf)) = NaN.
    # Flooring `_LOG_FLOOR_DECADES` orders of magnitude below the (always
    # finite) peak guarantees a finite floor and a bounded log-density range.
    log_p_max = float(np.max(log_p_grid))
    if not np.isfinite(log_p_max):
        # Degenerate KDE (all grid points underflowed); fall back to a
        # flat tiny density so w* ≡ 1 for this category.
        log_p_max = 0.0
    log_p_floor = log_p_max - _LOG_FLOOR_DECADES * np.log(10.0)
    log_p_ref = log_p_max

    log_p_grid_clamped = np.maximum(log_p_grid, log_p_floor)

    return {
        "log_m_grid": torch.from_numpy(log_m_grid).float(),
        "log_p_grid": torch.from_numpy(log_p_grid_clamped).float(),
        "log_p_floor": log_p_floor,
        "log_p_ref": log_p_ref,
    }


def build_per_k_kde(
    config_path: str,
    K_max: int,
    bandwidth: float,
    grid_size: int,
    grid_lo: float,
    grid_hi: float,
    max_events: int,
    max_kde_samples: int,
    batch_size: int = 4096,
    num_workers: int = 4,
    max_particles: int = 10,
) -> dict[int, dict]:
    """End-to-end: walk dataset → fit KDE per K → return per-K result dicts."""
    log_m_per_K = collect_log_masses(
        config_path=config_path,
        K_max=K_max,
        max_events=max_events,
        batch_size=batch_size,
        num_workers=num_workers,
        max_particles=max_particles,
    )
    out: dict[int, dict] = {}
    for K in range(2, K_max + 1):
        logger.info(f"Fitting KDE for K={K} ({len(log_m_per_K[K]):,} samples)")
        result = fit_kde_grid(
            log_m_values=log_m_per_K[K],
            bandwidth=bandwidth,
            grid_size=grid_size,
            grid_lo=grid_lo,
            grid_hi=grid_hi,
            max_kde_samples=max_kde_samples,
        )
        if result is None:
            logger.warning(f"  K={K}: no samples, skipping")
            continue
        out[K] = result
        logger.info(
            f"  K={K}: log_p_floor={result['log_p_floor']:.3f}, "
            f"log_p_ref={result['log_p_ref']:.3f}"
        )
    return out


# ---------------------------------------------------------------------------
# Category-conditioned KDE build
# ---------------------------------------------------------------------------
def collect_log_masses_by_category(
    config_path: str,
    K_max: int,
    max_events: int,
    batch_size: int = 4096,
    num_workers: int = 4,
    max_particles: int = 10,
    max_pairs_per_event: int | None = None,
) -> dict[int, dict[int, np.ndarray]]:
    """Walk the dataset, collect log m bucketed by (K, category key).

    Category key = type-composition + charge-sign of the tuple (see
    `tuple_categories.tuple_keys`).

    Returns:
        {K: {key (int): np.ndarray of log-masses for that category}}
    """
    dataset = ParticleDataset(
        config_path=config_path,
        normalize=False,
        lazy_loading=True,
    )
    N_total = len(dataset)
    logger.info(f"Dataset has {N_total:,} events")

    n_use = min(N_total, max_events)
    if n_use < N_total:
        dataset = Subset(dataset, list(range(n_use)))
        logger.info(f"  Capped to first {n_use:,} events for speed")

    collator = ParticleVectorCollator(max_particles=max_particles)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=False,
    )

    # Accumulate flat (log_m, key) tensors per K; bucket by key at the end.
    log_m_acc = {K: [] for K in range(2, K_max + 1)}
    key_acc = {K: [] for K in range(2, K_max + 1)}
    n_events = 0
    for i, batch in enumerate(loader):
        kin = batch["kinematics"]
        mask = batch["particle_mask"]
        p_type = batch["particle_type"]
        p_charge = batch["particle_charge"]
        for K in range(2, K_max + 1):
            log_m, key = all_kbody_log_masses(
                kin, mask, K, max_pairs_per_event, types=p_type, charges=p_charge
            )
            if log_m.numel() > 0:
                log_m_acc[K].append(log_m)
                key_acc[K].append(key)
        n_events += kin.shape[0]
        if (i + 1) % 50 == 0:
            logger.info(
                f"  {i + 1} batches ({n_events:,} events); "
                + ", ".join(
                    f"K={K}: {sum(t.numel() for t in log_m_acc[K]):,} tuples"
                    for K in range(2, K_max + 1)
                )
            )

    out: dict[int, dict[int, np.ndarray]] = {}
    for K in range(2, K_max + 1):
        out[K] = {}
        if not log_m_acc[K]:
            logger.info(f"K={K}: no tuples collected")
            continue
        log_m_all = torch.cat(log_m_acc[K]).numpy().astype(np.float64)
        key_all = torch.cat(key_acc[K]).numpy()
        uniq_keys, counts = np.unique(key_all, return_counts=True)
        for ky in uniq_keys:
            out[K][int(ky)] = log_m_all[key_all == ky]
        logger.info(
            f"K={K}: {len(log_m_all):,} tuples across {len(uniq_keys)} categories"
        )
    return out


def build_per_k_category_kde(
    config_path: str,
    K_max: int,
    bandwidth: float,
    grid_size: int,
    grid_lo: float,
    grid_hi: float,
    max_events: int,
    max_kde_samples: int,
    category_min_samples: int,
    batch_size: int = 4096,
    num_workers: int = 4,
    max_particles: int = 10,
) -> dict[int, dict]:
    """End-to-end: walk dataset → bucket by category → fit a KDE per
    (K, category) whose sample count clears `category_min_samples`.

    Returns:
        {K: {
            "log_m_grid": (grid_size,) float32 tensor (shared across categories),
            "categories": [  # sorted by key ascending
                {"key": int, "log_p_grid": tensor, "log_p_floor": float,
                 "n_samples": int},
                ...
            ],
        }}
    Categories below the threshold are dropped (tuples with those keys fall
    back to w*=1 at train time).
    """
    from shellflow.data.tuple_categories import key_to_name

    by_cat = collect_log_masses_by_category(
        config_path=config_path,
        K_max=K_max,
        max_events=max_events,
        batch_size=batch_size,
        num_workers=num_workers,
        max_particles=max_particles,
    )

    out: dict[int, dict] = {}
    for K in range(2, K_max + 1):
        cats = by_cat.get(K, {})
        kept: list[dict] = []
        log_m_grid = None
        # Deterministic order: ascending key.
        for ky in sorted(cats.keys()):
            samples = cats[ky]
            if len(samples) < category_min_samples:
                continue
            result = fit_kde_grid(
                log_m_values=samples,
                bandwidth=bandwidth,
                grid_size=grid_size,
                grid_lo=grid_lo,
                grid_hi=grid_hi,
                max_kde_samples=max_kde_samples,
            )
            if result is None:
                continue
            if log_m_grid is None:
                log_m_grid = result["log_m_grid"]
            kept.append(
                {
                    "key": int(ky),
                    "log_p_grid": result["log_p_grid"],
                    "log_p_floor": result["log_p_floor"],
                    "n_samples": int(len(samples)),
                }
            )
            logger.info(
                f"  K={K} cat key={ky} [{key_to_name(ky, K)}]: "
                f"{len(samples):,} samples, "
                f"log_p_floor={result['log_p_floor']:.3f}"
            )
        if log_m_grid is None:
            # No category cleared the threshold — build a shared grid anyway so
            # the lookup has a valid (empty-category) grid to register.
            log_m_grid = torch.linspace(grid_lo, grid_hi, grid_size).float()
            logger.warning(
                f"  K={K}: no category cleared category_min_samples="
                f"{category_min_samples}; all tuples fall back to w*=1"
            )
        out[K] = {"log_m_grid": log_m_grid, "categories": kept}
        logger.info(f"K={K}: kept {len(kept)} categories ≥ {category_min_samples}")
    return out
