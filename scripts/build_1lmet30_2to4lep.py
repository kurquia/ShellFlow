"""Build the merged, de-duplicated 1LMET30 ∪ 2to4lep converted dataset.

Two outputs under data/:
  run2_1LMET30_2to4lep_converted : full events,        seq_len = global max
  run2_1LMET30_2to4lep_reduced   : events with <=8 particles, seq_len = max over <=8

De-duplication
--------------
2to4lep and 1LMET30 are different event-level skims of the SAME Run-2 data, so
events with >=2 leptons (pT>7) AND MET>30 appear in both. Verified empirically
that a shared (runNumber, eventNumber) has byte-identical stored objects across
the two skims, so dedup is lossless and source-agnostic — we keep the 2to4lep
copy (per requirement) and drop the 1LMET30 duplicate.

Lossless global_id
------------------
The in-band metadata token in transformer_input is float32, which cannot hold
the 64-bit (runNumber<<40)|eventNumber id (collapses ~1e10 distinct events to
one value). So we additionally write an int64 sidecar `global_ids_XXXX.npy`
alongside each `transformer_input_XXXX.npy` (same row order), preserving the
exact id. Recover (run, event) via
shellflow.utils.metadata.MetadataHandler.recover_from_id.

Resource constraints
--------------------
- RAM <= 200 GB: true streaming (never load a whole file); modest chunk sizes.
- Gentle on CPU (training runs concurrently): single process, no DataLoader
  workers; launch under `nice`.

Run:
  nice -n 19 uv run python scripts/build_1lmet30_2to4lep.py
"""

from __future__ import annotations

import argparse
import gc
import glob
import os
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

from shellflow.compute_higgs_der_features import add_der_features_to_tree
from shellflow.rootfile_to_numpy import (
    DATA_FEATURE_PATH,
    convert_tree_to_flatsequence_fixed,
    load_feature_config,
)
from shellflow.utils.logger_config import get_logger

logger = get_logger(__name__)

DIR_2TO4LEP = "data/root_datasets/run2_data_2to4lep"
DIR_1LMET30 = "data/root_datasets/run2_data_1LMET30"
OUT_BASE = "data"
OUT_FULL = "run2_1LMET30_2to4lep_converted"
OUT_RED = "run2_1LMET30_2to4lep_reduced"

# Per-type particle-count branches (also used by the <=8 total filter, matching
# rootfile_to_numpy._filter_by_particle_count).
PARTICLE_COUNT_BRANCHES = ["lep_n", "jet_n", "photon_n", "tau_n", "largeRJet_n"]
# feature_config type -> its count branch (metadata/met are always count 1).
TYPE_COUNT_BRANCH = {
    "lepton": "lep_n",
    "photon": "photon_n",
    "tau": "tau_n",
    "jet": "jet_n",
    "largeRjet": "largeRJet_n",
}
SHIFT_BITS = 40  # matches shellflow.utils.metadata.MetadataHandler
MAX_PARTICLES = 8


def _global_ids(tree, start, stop) -> np.ndarray:
    """Exact int64 (runNumber<<40)|eventNumber for events [start, stop)."""
    run = np.asarray(
        tree["runNumber"].array(entry_start=start, entry_stop=stop, library="np")
    ).astype(np.uint64)
    ev_raw = tree["eventNumber"].array(entry_start=start, entry_stop=stop, library="np")
    # eventNumber is stored as a string branch in ATLAS Open Data ntuples.
    ev = np.array([int(x) for x in ev_raw], dtype=np.uint64)
    gid = (run << np.uint64(SHIFT_BITS)) | ev
    return gid.astype(np.int64)


def scan_exclusion_and_seqlens(
    files_2to4lep, files_1lmet30, feature_config, batch: int
):
    """One lean streaming pass over both skims, reading ONLY count + id branches.

    Returns:
        excl_ids      : sorted unique int64 global ids present in 2to4lep
        full_seq_len  : seq_len from per-type max over ALL union events
        red_seq_len   : seq_len from per-type max over <=8-particle union events
        stats         : dict of per-type maxima + event counts
    Dedup is irrelevant for seq_len (duplicate events carry identical
    multiplicities already counted from 2to4lep), so the scan uses the naive
    union — its per-type maxima equal the deduped union's.
    """
    types = list(feature_config.keys())
    nfields = {t: len(feature_config[t]["fields"]) for t in types}
    max_full = {t: 0 for t in types}
    max_red = {t: 0 for t in types}
    n_events = {"2to4lep": 0, "1LMET30": 0}
    excl_chunks = []

    for tag, files in (("2to4lep", files_2to4lep), ("1LMET30", files_1lmet30)):
        for fi, f in enumerate(files):
            t = uproot.open(f)["analysis;1"]
            n = t.num_entries
            n_events[tag] += n
            for s in range(0, n, batch):
                e = min(s + batch, n)
                cnt = {
                    b: np.asarray(t[b].array(entry_start=s, entry_stop=e, library="np"))
                    for b in PARTICLE_COUNT_BRANCHES
                }
                total = sum(cnt.values())  # (B,) total particles per event
                le8 = total <= MAX_PARTICLES
                per_type = {
                    "lepton": cnt["lep_n"],
                    "photon": cnt["photon_n"],
                    "tau": cnt["tau_n"],
                    "jet": cnt["jet_n"],
                    "largeRjet": cnt["largeRJet_n"],
                }
                for ty in types:
                    if ty in ("metadata", "met"):
                        max_full[ty] = max(max_full[ty], 1)
                        max_red[ty] = max(max_red[ty], 1)
                        continue
                    c = per_type[ty]
                    if len(c):
                        max_full[ty] = max(max_full[ty], int(c.max()))
                    if le8.any():
                        max_red[ty] = max(max_red[ty], int(c[le8].max()))
                if tag == "2to4lep":
                    excl_chunks.append(_global_ids(t, s, e))
            del t
            gc.collect()
            logger.info(
                f"[scan] {tag} file {fi + 1}/{len(files)} done "
                f"({n:,} events); full_seq so far="
                f"{sum((nfields[x] + 1) * max_full[x] for x in types)}"
            )

    excl_ids = (
        np.unique(np.concatenate(excl_chunks))
        if excl_chunks
        else np.empty(0, dtype=np.int64)
    )
    full_seq = int(sum((nfields[t] + 1) * max_full[t] for t in types))
    red_seq = int(sum((nfields[t] + 1) * max_red[t] for t in types))
    stats = {
        "max_full": max_full,
        "max_red": red_seq and max_red,
        "n_events": n_events,
        "n_excl": int(len(excl_ids)),
    }
    return excl_ids, full_seq, red_seq, stats


class _ChunkWriter:
    """Accumulates (data, gid) and flushes transformer_input + global_ids
    sidecar chunks once the event count crosses `max_per_chunk`."""

    def __init__(self, out_dir: str, max_per_chunk: int):
        self.out_dir = out_dir
        self.max_per_chunk = max_per_chunk
        os.makedirs(out_dir, exist_ok=True)
        self.data = []
        self.gids = []
        self.n = 0
        self.idx = 0
        self.total = 0

    def add(self, data: np.ndarray, gid: np.ndarray):
        self.data.append(data)
        self.gids.append(gid)
        self.n += data.shape[0]
        self.total += data.shape[0]
        if self.n >= self.max_per_chunk:
            self.flush()

    def flush(self):
        if self.n == 0:
            return
        data = np.concatenate(self.data, axis=0)
        gid = np.concatenate(self.gids, axis=0)
        np.save(
            os.path.join(self.out_dir, f"transformer_input_{self.idx:04d}.npy"), data
        )
        np.save(os.path.join(self.out_dir, f"global_ids_{self.idx:04d}.npy"), gid)
        logger.info(
            f"[{Path(self.out_dir).name}] saved chunk {self.idx} "
            f"({data.shape[0]:,} events, shape {data.shape})"
        )
        self.idx += 1
        self.data = []
        self.gids = []
        self.n = 0
        del data, gid
        gc.collect()


def convert(
    files_2to4lep,
    files_1lmet30,
    feature_config,
    type_encoding,
    excl_ids: np.ndarray,
    full_seq: int,
    red_seq: int,
    batch: int,
    max_per_chunk: int,
):
    """Single I/O pass producing both outputs with dedup + int64 sidecar."""
    full_w = _ChunkWriter(os.path.join(OUT_BASE, OUT_FULL), max_per_chunk)
    red_w = _ChunkWriter(os.path.join(OUT_BASE, OUT_RED), max_per_chunk)
    n_dropped_dup = 0

    for tag, files in (("2to4lep", files_2to4lep), ("1LMET30", files_1lmet30)):
        for fi, f in enumerate(files):
            t = uproot.open(f)["analysis;1"]
            n = t.num_entries
            for s in range(0, n, batch):
                e = min(s + batch, n)
                arr = t.arrays(library="ak", entry_start=s, entry_stop=e)
                gid = _global_ids(t, s, e)  # (B,) int64

                # Dedup: drop 1LMET30 events whose id is in the 2to4lep set
                # (keep-2to4lep). searchsorted + exact-match against sorted set.
                if tag == "1LMET30" and excl_ids.size:
                    pos = np.searchsorted(excl_ids, gid).clip(0, excl_ids.size - 1)
                    dup = excl_ids[pos] == gid
                    if dup.any():
                        n_dropped_dup += int(dup.sum())
                        keep = ~dup
                        arr = arr[keep]
                        gid = gid[keep]
                if len(arr) == 0:
                    continue

                try:
                    arr = add_der_features_to_tree(arr)
                except Exception as ex:  # noqa: BLE001
                    logger.warning(f"DER features failed on {f} [{s}:{e}]: {ex}")

                # FULL output (all events, packed at full_seq)
                data_full = convert_tree_to_flatsequence_fixed(
                    arr, feature_config, type_encoding, full_seq
                )
                if data_full is not None:
                    full_w.add(data_full, gid)

                # REDUCED output (<=8 particles, packed at red_seq)
                total = sum(
                    np.asarray(ak.to_numpy(arr[b]))
                    for b in PARTICLE_COUNT_BRANCHES
                    if b in arr.fields
                )
                m8 = total <= MAX_PARTICLES
                if m8.any():
                    arr8 = arr[m8]
                    gid8 = gid[np.asarray(m8)]
                    data_red = convert_tree_to_flatsequence_fixed(
                        arr8, feature_config, type_encoding, red_seq
                    )
                    if data_red is not None:
                        red_w.add(data_red, gid8)

                del arr, gid
            del t
            gc.collect()
            logger.info(
                f"[convert] {tag} file {fi + 1}/{len(files)} done; "
                f"full={full_w.total:,} reduced={red_w.total:,} "
                f"dup_dropped={n_dropped_dup:,}"
            )

    full_w.flush()
    red_w.flush()
    logger.info("=" * 70)
    logger.info(
        f"DONE. full={full_w.total:,} events (seq={full_seq}) → {OUT_FULL}; "
        f"reduced={red_w.total:,} events (seq={red_seq}) → {OUT_RED}; "
        f"duplicates dropped={n_dropped_dup:,}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DATA_FEATURE_PATH))
    ap.add_argument("--batch", type=int, default=100_000)
    ap.add_argument("--scan-batch", type=int, default=500_000)
    ap.add_argument("--max-per-chunk", type=int, default=5_000_000)
    ap.add_argument(
        "--limit-files",
        type=int,
        default=0,
        help="If >0, use only the first N files from each skim (for validation).",
    )
    ap.add_argument(
        "--out-suffix",
        default="",
        help="Append to output dir names (for validation runs, e.g. _test).",
    )
    args = ap.parse_args()

    global OUT_FULL, OUT_RED
    OUT_FULL += args.out_suffix
    OUT_RED += args.out_suffix

    feature_config, type_encoding = load_feature_config(args.config)
    files_2 = sorted(glob.glob(f"{DIR_2TO4LEP}/*.root"))
    files_1 = sorted(glob.glob(f"{DIR_1LMET30}/*.root"))
    if args.limit_files > 0:
        files_2 = files_2[: args.limit_files]
        files_1 = files_1[: args.limit_files]
    logger.info(f"2to4lep files: {len(files_2)} | 1LMET30 files: {len(files_1)}")

    logger.info("=" * 70)
    logger.info("STAGE 1/2: lean scan (exclusion ids + seq lengths)")
    excl_ids, full_seq, red_seq, stats = scan_exclusion_and_seqlens(
        files_2, files_1, feature_config, args.scan_batch
    )
    logger.info(
        f"2to4lep events={stats['n_events']['2to4lep']:,} "
        f"1LMET30 events={stats['n_events']['1LMET30']:,}"
    )
    logger.info(f"2to4lep unique ids (exclusion set)={stats['n_excl']:,}")
    logger.info(f"per-type max (full): {stats['max_full']}")
    logger.info(f"FULL seq_len={full_seq}  REDUCED seq_len={red_seq}")

    logger.info("=" * 70)
    logger.info("STAGE 2/2: convert (dedup + dual output + int64 sidecar)")
    convert(
        files_2,
        files_1,
        feature_config,
        type_encoding,
        excl_ids,
        full_seq,
        red_seq,
        args.batch,
        args.max_per_chunk,
    )


if __name__ == "__main__":
    main()
