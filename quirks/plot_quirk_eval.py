#!/usr/bin/env python3
"""Event displays + regression truth/reco comparisons from a hepattn eval file.

Reads the <ckpt>_<split>_eval.h5 written by hepattn's PredictionWriter during
`run_tracking.py test`, together with the prepared parquet events it was
evaluated on, and produces:

  displays/event<N>.png   x-y and r-z views: all spacepoints (grey), truth
                          quirk hits (open circles, one colour per quirk) and
                          the hits claimed by each valid query (filled dots)
  regression.png          pred vs truth log10(Lambda^2/m) and the pred-truth
                          plane-normal opening angle
  summary.txt             per-event matching table

Query i is aligned with truth particle i in the eval file (the writer stores
post-matching predictions), so truth/reco pairs need no re-matching here.

  plot_quirk_eval.py EVAL.h5 DATA_DIR OUTDIR
"""
import argparse
import os
import sys

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

QCOLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red",
           "tab:purple", "tab:brown", "tab:pink", "tab:olive"]


def get(grp, path):
    return np.asarray(grp[path])[0]  # stored with leading batch dim of 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_h5")
    ap.add_argument("data_dir")
    ap.add_argument("outdir")
    ap.add_argument("--max-displays", type=int, default=50)
    args = ap.parse_args()
    os.makedirs(os.path.join(args.outdir, "displays"), exist_ok=True)

    f = h5py.File(args.eval_h5, "r")
    sample_ids = sorted(f.keys(), key=int)
    print(f"{len(sample_ids)} events in {args.eval_h5}")

    fom_true, fom_pred = [], []
    normal_angles = []
    summary = []

    for k, sid in enumerate(sample_ids):
        g = f[sid]
        try:
            pred_valid = get(g, "preds/final/track_valid/track_valid").astype(bool)
            pred_masks = get(g, "preds/final/track_hit_valid/track_hit_valid").astype(bool)
            true_valid = get(g, "targets/particle_valid").astype(bool)
            true_masks = get(g, "targets/particle_hit_valid").astype(bool)
        except KeyError as e:
            print(f"missing key in event {sid}: {e}")
            g.visit(lambda n: print("  ", n))
            sys.exit(1)

        # regression heads (query i <-> truth particle i)
        p_fom = get(g, "preds/final/track_fom/track_log10_f_over_m")
        t_fom = get(g, "targets/particle_log10_f_over_m")
        p_n = np.stack([get(g, f"preds/final/track_plane/track_n{c}") for c in "xyz"], axis=-1)
        t_n = np.stack([get(g, f"targets/particle_n{c}") for c in "xyz"], axis=-1)

        both = true_valid & pred_valid
        fom_true.append(t_fom[both])
        fom_pred.append(p_fom[both])
        pn = p_n[both] / np.maximum(np.linalg.norm(p_n[both], axis=-1, keepdims=True), 1e-12)
        tn = t_n[both] / np.maximum(np.linalg.norm(t_n[both], axis=-1, keepdims=True), 1e-12)
        cosang = np.abs(np.sum(pn * tn, axis=-1))  # normal sign is a convention
        normal_angles.append(np.degrees(np.arccos(np.clip(cosang, -1, 1))))

        # hit-level matching numbers
        tp = (pred_masks & true_masks).sum(-1)
        eff = tp / np.maximum(true_masks.sum(-1), 1)
        pur = tp / np.maximum(pred_masks.sum(-1), 1)
        for i in np.where(true_valid)[0]:
            summary.append(
                f"event {sid} q{i}: matched={bool(pred_valid[i])} "
                f"nhits_true={true_masks[i].sum()} nhits_pred={pred_masks[i].sum()} "
                f"eff={eff[i]:.2f} pur={pur[i]:.2f} "
                f"fom true={t_fom[i]:.2f} pred={p_fom[i]:.2f}"
            )

        # -------- event display --------
        if k >= args.max_displays:
            continue
        ev = f"event{int(sid):09d}"
        hits = pd.read_parquet(os.path.join(args.data_dir, ev + "-hits.parquet"))
        hits = hits[hits.volume_id.isin([8, 9])].reset_index(drop=True)
        parts = pd.read_parquet(os.path.join(args.data_dir, ev + "-parts.parquet"))
        if len(hits) != pred_masks.shape[-1]:
            print(f"WARNING {ev}: {len(hits)} parquet hits vs {pred_masks.shape[-1]} in eval file, skipping display")
            continue
        x, y, z = hits.x.to_numpy(), hits.y.to_numpy(), hits.z.to_numpy()
        r = np.hypot(x, y)
        qids = parts[parts.get("is_quirk", pd.Series(True, index=parts.index)).astype(bool)]["particle_id"].to_numpy()

        fig, axs = plt.subplots(1, 2, figsize=(14, 6.5))
        for ax, (a, b, la, lb) in zip(axs, [(x, y, "x [mm]", "y [mm]"), (z, r, "z [mm]", "r [mm]")]):
            ax.scatter(a, b, s=3, c="0.8", label="all SPs", rasterized=True)
            for j, q in enumerate(qids):
                m = hits.particle_id.to_numpy() == q
                ax.scatter(a[m], b[m], s=70, facecolors="none",
                           edgecolors=QCOLORS[j % len(QCOLORS)], linewidths=1.4,
                           label=f"truth quirk {j + 1} ({m.sum()} hits)")
            for i in np.where(pred_valid)[0]:
                m = pred_masks[i]
                ax.scatter(a[m], b[m], s=14, color=QCOLORS[i % len(QCOLORS)], marker="o",
                           label=f"pred q{i} ({m.sum()} hits)")
            ax.set_xlabel(la)
            ax.set_ylabel(lb)
        axs[0].set_aspect("equal")
        h, lab = axs[0].get_legend_handles_labels()
        fig.legend(h, lab, loc="upper center", ncol=3, fontsize=8, frameon=False)
        fig.suptitle(
            f"{ev}   truth: log10(Lambda^2/m)={t_fom[true_valid].mean():.2f}"
            + (f"   pred: {p_fom[both].mean():.2f}" if both.any() else "   (no matched query)"),
            y=0.02, va="bottom")
        fig.tight_layout(rect=(0, 0.05, 1, 0.9))
        fig.savefig(os.path.join(args.outdir, "displays", ev + ".png"), dpi=130)
        plt.close(fig)

    # -------- regression summary figure --------
    fom_true = np.concatenate(fom_true) if fom_true else np.array([])
    fom_pred = np.concatenate(fom_pred) if fom_pred else np.array([])
    normal_angles = np.concatenate(normal_angles) if normal_angles else np.array([])

    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    if len(fom_true):
        lo, hi = min(fom_true.min(), fom_pred.min()) - 0.3, max(fom_true.max(), fom_pred.max()) + 0.3
        axs[0].plot([lo, hi], [lo, hi], "k--", lw=1)
        axs[0].scatter(fom_true, fom_pred, s=12, alpha=0.6)
        axs[0].set_xlabel("truth log10(Lambda^2/m)")
        axs[0].set_ylabel("predicted")
        axs[0].set_title(f"f/m regression ({len(fom_true)} matched quirks)")
        res = fom_pred - fom_true
        axs[1].hist(res, bins=40)
        axs[1].set_xlabel("pred - truth log10(Lambda^2/m)")
        axs[1].set_title(f"mean={res.mean():.3f}  RMS={res.std():.3f}")
    if len(normal_angles):
        axs[2].hist(normal_angles, bins=40, range=(0, 90))
        axs[2].set_xlabel("plane-normal opening angle [deg]")
        axs[2].set_title(f"median={np.median(normal_angles):.1f} deg")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "regression.png"), dpi=130)

    with open(os.path.join(args.outdir, "summary.txt"), "w") as sf:
        sf.write("\n".join(summary) + "\n")
    print(f"wrote {args.outdir}/regression.png, displays/, summary.txt")
    if len(fom_true):
        print(f"fom residual mean={np.mean(fom_pred - fom_true):.3f} RMS={np.std(fom_pred - fom_true):.3f}")
    if len(normal_angles):
        print(f"plane normal median angle {np.median(normal_angles):.1f} deg")


if __name__ == "__main__":
    main()
