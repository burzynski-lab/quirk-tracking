#!/usr/bin/env python3
"""Delete prepared events the hepattn loader would crash on.

The loader asserts "No particles remaining" when no particle passes
pt > 5 GeV, |eta| < 2.5 and >= 3 hits in volumes 8/9. prep_quirks.py now
skips such events at write time (--min-hits); this cleans directories
produced before that guard existed.

  clean_events.py DIR [DIR ...] [--min-hits 3] [--dry-run]
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--min-hits", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for d in args.dirs:
        n_bad = n_tot = 0
        for pf in sorted(glob.glob(os.path.join(d, "*-parts.parquet"))):
            ev = os.path.basename(pf)[: -len("-parts.parquet")]
            hf = os.path.join(d, ev + "-hits.parquet")
            parts = pd.read_parquet(pf, columns=["particle_id", "px", "py", "pz"])
            hits = pd.read_parquet(hf, columns=["particle_id", "volume_id"])
            hits = hits[hits.volume_id.isin([8, 9])]
            pt = np.hypot(parts.px, parts.py)
            p3 = np.sqrt(parts.px**2 + parts.py**2 + parts.pz**2)
            eta = np.arctanh(np.clip(parts.pz / np.maximum(p3, 1e-12), -1 + 1e-12, 1 - 1e-12))
            counts = hits.particle_id.value_counts()
            nhits = np.array([counts.get(b, 0) for b in parts.particle_id])
            n_tot += 1
            if not ((pt > 5.0) & (np.abs(eta) < 2.5) & (nhits >= args.min_hits)).any():
                n_bad += 1
                if args.dry_run:
                    print(f"would remove {d}/{ev}")
                else:
                    os.remove(pf)
                    os.remove(hf)
        print(f"{d}: removed {n_bad}/{n_tot} events")


if __name__ == "__main__":
    main()
