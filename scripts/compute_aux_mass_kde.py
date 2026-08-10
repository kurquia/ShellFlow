#!/usr/bin/env python
"""CLI wrapper to precompute K-body log-mass KDEs to a .pt file.

The lightning module's `_build_density_aware_buffers` calls the same helpers
inline at training start, so this script is NOT required for training. It's
useful for:
  - Inspection: load the .pt and plot log_p_grid vs exp(log_m_grid) in a notebook
    to visually confirm peaks at known resonances before launching training.
  - Reproducibility: archive a deterministic KDE artifact alongside checkpoints.

Usage:
    uv run python scripts/compute_aux_mass_kde.py \
        --config src/shellflow/config/dataloader_config_06_2to4lep_reduced.yaml \
        --output /tmp/kde_inspect.pt \
        --max-events 100000
"""

import argparse
from pathlib import Path

import torch

from shellflow.data.aux_mass_kde import build_per_k_kde
from shellflow.utils.logger_config import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Precompute K-body log-mass KDEs (CLI wrapper around shellflow.data.aux_mass_kde)"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--k-max", type=int, default=4)
    parser.add_argument(
        "--bandwidth",
        type=float,
        default=0.01,
        help="KDE bandwidth in log-m units (~1%% relative mass resolution)",
    )
    parser.add_argument("--grid-size", type=int, default=4096)
    parser.add_argument(
        "--grid-lo",
        type=float,
        default=-3.0,
        help="Lower bound of log-m grid (default -3 → m=0.05 GeV)",
    )
    parser.add_argument(
        "--grid-hi",
        type=float,
        default=6.0,
        help="Upper bound of log-m grid (default 6.0 → m≈403 GeV)",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=1_000_000,
        help="Cap on training events to walk (1M is plenty for KDE)",
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-particles", type=int, default=10)
    parser.add_argument(
        "--max-kde-samples",
        type=int,
        default=200_000,
        help="Subsample to this many log-m values for KDE fitting (eval cost scales)",
    )
    args = parser.parse_args()

    per_k = build_per_k_kde(
        config_path=args.config,
        K_max=args.k_max,
        bandwidth=args.bandwidth,
        grid_size=args.grid_size,
        grid_lo=args.grid_lo,
        grid_hi=args.grid_hi,
        max_events=args.max_events,
        max_kde_samples=args.max_kde_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_particles=args.max_particles,
    )

    out = {
        "metadata": {
            "config": args.config,
            "bandwidth": args.bandwidth,
            "grid_size": args.grid_size,
            "grid_lo": args.grid_lo,
            "grid_hi": args.grid_hi,
            "max_events": args.max_events,
        }
    }
    for K, result in per_k.items():
        out[f"K_{K}"] = result

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, output_path)
    logger.info(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
