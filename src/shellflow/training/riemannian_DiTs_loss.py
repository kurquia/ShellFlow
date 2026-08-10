"""OSSF lepton-pair selection helper for diagnostic metrics.

This module retains only the OSSF lepton-pair selector used by the
validation-time monitoring metrics (`_compute_ossf_metrics` in
`riemannian_DiTs_lightning.py`). Training-time invariant-mass losses no
longer live here: the current objective is the hierarchical K-body
invariant-mass loss on the separate auxiliary head, implemented by
`_compute_aux_head_loss` / `_compute_kbody_loss` in the lightning module.
"""

from typing import Optional

import torch


@torch.no_grad()
def _select_all_ossf_leptons(
    p_type: torch.Tensor,
    p_charge: torch.Tensor,
    p_mask: torch.Tensor,
) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Select ALL valid OSSF lepton pairs across the batch.

    Matches plot-level selection: same flavor, opposite charge, both active.
    No quality cuts or collinear guard — the full mass spectrum is visible.

    Returns:
        (b_idx, i_idx, j_idx) flat 1-D tensors for all valid pairs (sorted by
        b_idx via nonzero ordering), or None if empty.
    """
    B, N = p_type.shape
    device = p_type.device
    active = p_mask > 0.5

    is_lep = ((p_type == 11) | (p_type == 13)) & active
    q_signed = p_charge.float() - 1.0  # {0,1,2} → {-1,0,+1}

    same_flav = p_type.unsqueeze(2) == p_type.unsqueeze(1)
    opp_q = (q_signed.unsqueeze(2) * q_signed.unsqueeze(1)) < -0.5
    both_lep = is_lep.unsqueeze(2) & is_lep.unsqueeze(1)
    triu = torch.triu(torch.ones(N, N, dtype=torch.bool, device=device), diagonal=1)

    valid = same_flav & opp_q & both_lep & triu

    if not valid.any():
        return None

    b_idx, i_idx, j_idx = valid.nonzero(as_tuple=True)
    return b_idx, i_idx, j_idx
