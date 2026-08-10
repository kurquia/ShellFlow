"""Set-wise tuple categorization for category-conditioned distribution matching.

A K-tuple's *category* is a permutation-invariant function of its particles'
(type, charge-sign). It is used to look up the per-category invariant-mass KDE
in the distribution-matching weight w* (see riemannian_DiTs_lightning.py
`_distribution_matching_weight_categorized`). The motivation: reweighting a
specific pair (e.g. an OSSF dilepton) against the *all-pairs* mass density is
the wrong reference — resonances get diluted by the combinatorial mixture.
Categorizing by composition lets each kind of tuple be matched against its own
reference density.

The category KEY is an int64 encoding of the sorted multiset of per-particle
codes:
    code  = type_idx * NUM_CHARGE_CLASSES + charge_idx
    key   = Σ_i  sorted_code[i] · BASE**i        (BASE = NUM_PARTICLE_TYPES·3)

  - type_idx: PARTICLE_TYPE_CODES order (jet=0, largeRjet=1, e=2, mu=3, tau=4,
    photon=5); any other code (MET=0, padding, unknown) → NUM_PARTICLE_TYPES-1.
  - charge_idx: collator charge index in {0,1,2} ≙ physical charge {−1,0,+1}.

Sorting the per-particle codes before encoding makes the key invariant to the
order of particles within the tuple — the K-body invariant mass is likewise
permutation-invariant — so the train-time `combs` ordering and the KDE-build
`torch.combinations` ordering produce identical keys for the same tuple.

Categories are NOT a fixed enumeration: they are discovered dynamically at
KDE-build time (keys with ≥ category_min_samples occurrences get their own KDE;
rarer keys fall back to w*=1). This module only provides the key function and a
human-readable decoder for logging.
"""

from __future__ import annotations

import torch

# These MUST match shellflow.models.riemannian_DiTs_model (asserted by unit test
# test_tuple_categories.py::test_constants_match_model). Replicated here to keep
# this module dependency-free (importable from data/ without pulling in the
# model module).
PARTICLE_TYPE_CODES = [1, 10, 11, 13, 15, 22]
NUM_PARTICLE_TYPES = len(PARTICLE_TYPE_CODES) + 1  # +1 unknown/padding/MET
NUM_CHARGE_CLASSES = 3  # {-1, 0, +1} stored as {0, 1, 2}
BASE = NUM_PARTICLE_TYPES * NUM_CHARGE_CLASSES  # per-particle code range = 21

_UNKNOWN_IDX = NUM_PARTICLE_TYPES - 1
_LUT_SIZE = 100

# Human-readable names for the key decoder (logging / metrics only).
_TYPE_NAMES = {0: "jet", 1: "lRjet", 2: "e", 3: "mu", 4: "tau", 5: "gamma", 6: "X"}
_SIGN_NAMES = {0: "-", 1: "0", 2: "+"}
_CHARGED_TYPE_IDX = {2, 3, 4}  # e, mu, tau — show charge sign in the name


def _type_lut(device: torch.device) -> torch.Tensor:
    """type code → type index LUT (size _LUT_SIZE), default = unknown index."""
    lut = torch.full((_LUT_SIZE,), _UNKNOWN_IDX, dtype=torch.long, device=device)
    for idx, code in enumerate(PARTICLE_TYPE_CODES):
        lut[code] = idx
    return lut


def tuple_codes(types: torch.Tensor, charges: torch.Tensor) -> torch.Tensor:
    """Per-particle code = type_idx · NUM_CHARGE_CLASSES + charge_idx.

    Args:
        types:   (...) int — particle type codes (1, 11, 13, …)
        charges: (...) int — collator charge index in {0, 1, 2}
    Returns:
        codes: (...) long — code in [0, BASE)
    """
    lut = _type_lut(types.device)
    safe = types.clamp(0, _LUT_SIZE - 1).long()
    type_idx = lut[safe]
    charge_idx = charges.clamp(0, NUM_CHARGE_CLASSES - 1).long()
    return type_idx * NUM_CHARGE_CLASSES + charge_idx


def tuple_keys(
    types_tuple: torch.Tensor,
    charges_tuple: torch.Tensor,
    K: int,
) -> torch.Tensor:
    """Permutation-invariant int64 category key per tuple.

    Args:
        types_tuple:   (..., K) int — types of the K particles in each tuple
        charges_tuple: (..., K) int — charge indices of the K particles
        K:             tuple size (must equal the last dim)
    Returns:
        key: (...) int64 — base-BASE encoding of the sorted per-particle codes
    """
    if types_tuple.shape[-1] != K:
        raise ValueError(f"tuple_keys: last dim {types_tuple.shape[-1]} != K={K}")
    codes = tuple_codes(types_tuple, charges_tuple)  # (..., K)
    codes_sorted, _ = torch.sort(codes, dim=-1)  # canonical (multiset) order
    powers = BASE ** torch.arange(K, device=codes.device, dtype=torch.int64)
    return (codes_sorted.to(torch.int64) * powers).sum(dim=-1)


def code_to_str(code: int) -> str:
    """Decode a single per-particle code to a short string, e.g. 'e+', 'jet'."""
    t = code // NUM_CHARGE_CLASSES
    c = code % NUM_CHARGE_CLASSES
    name = _TYPE_NAMES.get(int(t), "X")
    if int(t) in _CHARGED_TYPE_IDX:
        return f"{name}{_SIGN_NAMES[int(c)]}"
    return name


def key_to_name(key: int, K: int) -> str:
    """Decode an int64 key to a human-readable composition, e.g. 'e- e+'.

    For logging / metric naming only. Codes come out in ascending (sorted)
    order, matching the encoding in `tuple_keys`.
    """
    k = int(key)
    parts = []
    for _ in range(K):
        parts.append(code_to_str(k % BASE))
        k //= BASE
    return " ".join(parts)
