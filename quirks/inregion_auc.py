#!/usr/bin/env python3
"""In-region ranking power of the mask head.

For each matched (query, truth-quirk) pair: region = the query's top-K hits
by mask logit. Reports
  capture:      fraction of the quirk's true hits inside the region
                (is the REGION right?)
  in-region AUC: ranking of true vs background hits *within* the region
                (does the model order hits inside it, or is it random-40?)
  global AUC:    ranking over all hits (reference)

  inregion_auc.py EVAL.h5 [--topk 200]
"""
import argparse
import glob

import h5py
import numpy as np


def auc(score, label):
    order = np.argsort(score)
    r = np.empty(len(score)); r[order] = np.arange(len(score))
    pos = label.astype(bool); n1 = pos.sum(); n0 = len(label) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return (r[pos].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_h5")
    ap.add_argument("--topk", type=int, default=200)
    a = ap.parse_args()
    f = h5py.File(sorted(glob.glob(a.eval_h5))[-1])

    caps, in_aucs, gl_aucs = [], [], []
    for sid in f:
        g = f[sid]
        logit = np.asarray(g["outputs/final/track_hit_valid/track_hit_logit"])[0].astype(np.float64)
        tv = np.asarray(g["targets/particle_valid"])[0].astype(bool)
        tm = np.asarray(g["targets/particle_hit_valid"])[0].astype(bool)
        for i in np.where(tv)[0]:
            lab = tm[i]
            if lab.sum() < 3:
                continue
            s = logit[i]
            gl_aucs.append(auc(s, lab))
            top = np.argsort(s)[::-1][: a.topk]
            caps.append(lab[top].sum() / lab.sum())
            in_aucs.append(auc(s[top], lab[top]))
    caps, in_aucs, gl_aucs = map(np.array, (caps, in_aucs, gl_aucs))
    print(f"tracks measured: {len(caps)}   region = top-{a.topk} hits by mask logit")
    print(f"region capture:  mean {np.nanmean(caps):.3f}  median {np.nanmedian(caps):.3f}")
    print(f"in-region AUC:   mean {np.nanmean(in_aucs):.3f}  median {np.nanmedian(in_aucs):.3f}   (0.5 = random-within-region)")
    print(f"global AUC:      mean {np.nanmean(gl_aucs):.3f}")


if __name__ == "__main__":
    main()
