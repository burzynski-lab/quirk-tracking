#!/usr/bin/env python3
"""Duplication metrics for an eval h5: are the declared tracks distinct?

Per event with >= 2 valid queries: pairwise IoU between predicted masks.
Reports, overall and per Lambda block:
  pair_iou:    mean pairwise pred-mask IoU (0 = disjoint tracks, 1 = copies)
  dup_frac:    fraction of multi-track events with any pair IoU > 0.5
  eff_raw:     standard p0.5 efficiency (duplicate-blind, as logged)
  eff_dedup:   p0.5 efficiency after greedy de-duplication (IoU > 0.5 keeps
               the first mask) - the honest per-quirk efficiency

  dup_metrics.py EVAL.h5
"""
import argparse
import glob

import h5py
import numpy as np

LAMS = [100, 167, 278, 464, 774, 1292, 2154, 3594, 5995, 10000]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_h5")
    ap.add_argument("--iou-cut", type=float, default=0.5)
    a = ap.parse_args()
    f = h5py.File(sorted(glob.glob(a.eval_h5))[-1])

    rows = {}
    for sid in f:
        g = f[sid]
        blk = int(sid) // 100000
        pv = np.asarray(g["preds/final/track_valid/track_valid"])[0].astype(bool)
        pm = np.asarray(g["preds/final/track_hit_valid/track_hit_valid"])[0].astype(bool)
        tv = np.asarray(g["targets/particle_valid"])[0].astype(bool)
        tm = np.asarray(g["targets/particle_hit_valid"])[0].astype(bool)
        ids = list(np.where(pv)[0])

        ious = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                A, B = pm[ids[i]], pm[ids[j]]
                ious.append((A & B).sum() / max((A | B).sum(), 1))

        # greedy de-dup: drop any mask overlapping an earlier kept one
        kept = []
        for i in ids:
            if all((pm[i] & pm[k]).sum() / max((pm[i] | pm[k]).sum(), 1) <= a.iou_cut for k in kept):
                kept.append(i)

        def eff(slots):
            found = 0
            for t in np.where(tv)[0]:
                if any((pm[s] & tm[t]).sum() / max(tm[t].sum(), 1) >= 0.5 for s in slots):
                    found += 1
            return found, int(tv.sum())

        fr, nt = eff(ids)
        fd, _ = eff(kept)
        r = rows.setdefault(blk, {"iou": [], "dup": [], "raw": [0, 0], "ded": [0, 0]})
        if ious:
            r["iou"] += ious
            r["dup"].append(float(max(ious) > a.iou_cut))
        r["raw"][0] += fr; r["raw"][1] += nt
        r["ded"][0] += fd; r["ded"][1] += nt

    print(f"{'Lambda':>7s} {'pair_iou':>9s} {'dup_frac':>9s} {'eff_raw':>8s} {'eff_dedup':>9s}")
    tot = {"iou": [], "dup": [], "raw": [0, 0], "ded": [0, 0]}
    for b in sorted(rows):
        r = rows[b]
        for k in ("iou", "dup"):
            tot[k] += r[k]
        for k in ("raw", "ded"):
            tot[k][0] += r[k][0]; tot[k][1] += r[k][1]
        print(f"{LAMS[b]:7d} {np.mean(r['iou']) if r['iou'] else float('nan'):9.2f} "
              f"{np.mean(r['dup']) if r['dup'] else float('nan'):9.2f} "
              f"{r['raw'][0]/max(r['raw'][1],1):8.2f} {r['ded'][0]/max(r['ded'][1],1):9.2f}")
    print(f"{'ALL':>7s} {np.mean(tot['iou']):9.2f} {np.mean(tot['dup']):9.2f} "
          f"{tot['raw'][0]/max(tot['raw'][1],1):8.2f} {tot['ded'][0]/max(tot['ded'][1],1):9.2f}")


if __name__ == "__main__":
    main()
