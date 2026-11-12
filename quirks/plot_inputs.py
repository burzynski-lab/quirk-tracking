#!/usr/bin/env python3
"""Plot the variables of prepared hepattn parquet (prep_colliderml.py or prep_quirks.py output).

Reads event<N>-hits.parquet / event<N>-parts.parquet from one or more data
directories (e.g. colliderml_v1/train, colliderml_v1/test/m50.L100) and adds the variables
hepattn's TrackML loader derives on load (hepattn/.../trackml/data.py,
load_event), with the same formulas, so the plots show what the network sees:
  hits:  x y z scaled by 0.01 (as the loader does), r s theta phi eta u v
  parts: p pt qopt eta phi
Hits are split into quirk hits (particle_id > 0; in quirks-only data every
kept particle is a quirk) and other hits; each directory gets its own colour.
s_arc and phase are plotted for quirk hits only (0 elsewhere by construction).

Writes OUTDIR/inputs.pdf (hit variables, particle variables, quirk pT by
charge and by per-event pT order, a few event displays) and OUTDIR/summary.txt (per-column statistics; flags constant
columns such as the all-zero ODD angle columns).

  plot_inputs.py DATA_DIR [DATA_DIR ...] -o OUTDIR [--events 500]
                 [--volumes 16 17 18 23 24 25 28 29 30]
                 [--displays 4] [--display-seed SEED]
"""
import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

DERIVED_HITS = ["r", "s", "theta", "phi", "eta", "u", "v"]
SKIP_HITS = {"particle_id"}
SKIP_PARTS = {"particle_id", "odd_particle_id"}
QUIRK_ONLY_HITS = {"s_arc", "phase"}


def add_loader_fields(hits, parts):
    """The fields hepattn's TrackML loader computes in load_event, same formulas."""
    for c in "xyz":
        hits[c] = hits[c] * 0.01
    hits["r"] = np.sqrt(hits["x"] ** 2 + hits["y"] ** 2)
    hits["s"] = np.sqrt(hits["x"] ** 2 + hits["y"] ** 2 + hits["z"] ** 2)
    hits["theta"] = np.arccos(hits["z"] / hits["s"])
    hits["phi"] = np.arctan2(hits["y"], hits["x"])
    hits["eta"] = -np.log(np.tan(hits["theta"] / 2))
    hits["u"] = hits["x"] / (hits["x"] ** 2 + hits["y"] ** 2)
    hits["v"] = hits["y"] / (hits["x"] ** 2 + hits["y"] ** 2)
    parts["p"] = np.sqrt(parts["px"] ** 2 + parts["py"] ** 2 + parts["pz"] ** 2)
    parts["pt"] = np.sqrt(parts["px"] ** 2 + parts["py"] ** 2)
    parts["qopt"] = parts["q"] / parts["pt"]
    parts["eta"] = np.arctanh(parts["pz"] / parts["p"])
    parts["phi"] = np.arctan2(parts["py"], parts["px"])


def load(data_dir, n_events, volumes):
    files = sorted(glob.glob(os.path.join(data_dir, "event*-parts.parquet")))
    if not files:
        raise SystemExit(f"no event*-parts.parquet in {data_dir}")
    if 0 < n_events < len(files):
        files = [files[i] for i in np.random.default_rng(0).choice(len(files), n_events, replace=False)]
    hits, parts = [], []
    for i, f in enumerate(files):
        p = pd.read_parquet(f)
        h = pd.read_parquet(f.replace("-parts.parquet", "-hits.parquet"))
        if volumes:
            h = h[h["volume_id"].isin(volumes)]
        h["event"] = i
        p["event"] = i
        hits.append(h)
        parts.append(p)
    hits, parts = pd.concat(hits, ignore_index=True), pd.concat(parts, ignore_index=True)
    add_loader_fields(hits, parts)
    return hits, parts, [os.path.basename(f)[:-len("-parts.parquet")] for f in files]


def bins_for(arrays):
    a = np.concatenate([x[np.isfinite(x)] for x in arrays if len(x)])
    if len(a) == 0:
        return None
    u = np.unique(a)
    if len(u) <= 40 and np.allclose(u, np.round(u)):        # integer-valued: one bin per value
        return np.arange(u.min() - 0.5, u.max() + 1.5)
    lo, hi = np.percentile(a, [0.5, 99.5])
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    return np.linspace(lo, hi, 60)


def hist_page(pdf, columns, series, title):
    """series: list of (label, colour, linestyle, DataFrame); one panel per column, 3x3 per page."""
    for start in range(0, len(columns), 9):
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        for ax, col in zip(axes.flat, columns[start:start + 9]):
            arrays = [df[col].to_numpy(dtype=float) for *_, df in series if col in df]
            b = bins_for(arrays)
            if b is None:
                ax.set_title(f"{col} (no values)")
                continue
            for label, colour, ls, df in series:
                if col not in df or len(df) == 0:
                    continue
                x = df[col].to_numpy(dtype=float)
                x = x[np.isfinite(x)]
                ax.hist(np.clip(x, b[0], b[-1]), bins=b, density=True, histtype="step",
                        color=colour, linestyle=ls, label=label, linewidth=1.3)
            ax.set_yscale("log")
            ax.set_title(col)
        for ax in axes.flat[len(columns[start:start + 9]):]:
            ax.axis("off")
        axes.flat[0].legend(fontsize=7)
        fig.suptitle(f"{title} (unit area; under/overflow in the edge bins)")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def quirk_pt_page(pdf, data):
    """pT of quirk+ vs quirk-, and of the harder vs softer quirk of each event (events with exactly two quirks)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    b = bins_for([parts["pt"].to_numpy(dtype=float) for _, _, _, parts, _ in data])
    if b is None:
        plt.close(fig)
        return
    for label, colour, _, parts, _ in data:
        pairs = parts[parts.groupby("event")["pt"].transform("size") == 2]
        by_event = pairs.groupby("event")["pt"]
        panels = [
            (axes[0], parts.loc[parts["q"] > 0, "pt"], "-", "quirk+"),
            (axes[0], parts.loc[parts["q"] < 0, "pt"], "--", "quirk-"),
            (axes[1], by_event.max(), "-", "higher pT"),
            (axes[1], by_event.min(), "--", "lower pT"),
        ]
        for ax, x, ls, what in panels:
            x = x.to_numpy(dtype=float)
            ax.hist(np.clip(x, b[0], b[-1]), bins=b, density=True, histtype="step", color=colour,
                    linestyle=ls, linewidth=1.3, label=f"{label} {what} ({len(x)})")
    axes[0].set_title("quirk pT by charge")
    axes[1].set_title("quirk pT ordered per event (events with 2 quirks)")
    for ax in axes:
        ax.set_xlabel("pT [GeV]")
        ax.set_yscale("log")
        ax.legend(fontsize=7)
    fig.suptitle("Quirk pT (unit area; under/overflow in the edge bins)")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def event_displays(pdf, label, hits, parts, names, events, per_page=4):
    """r-z and x-y displays of the given events (indices into names), per_page events per page."""
    for start in range(0, len(events), per_page):
        page = events[start:start + per_page]
        fig, axes = plt.subplots(2, per_page, figsize=(4.5 * per_page, 9), squeeze=False)
        for col, i in enumerate(page):
            event_display(axes[:, col], label, hits, parts, names, i)
        for ax in axes[:, len(page):].flat:
            ax.axis("off")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def event_display(axes, label, hits, parts, names, i):
    """One event: r-z view in axes[0], x-y view in axes[1]; quirk charge, pT and p in the legend."""
    h = hits[hits["event"] == i]
    p = parts[parts["event"] == i].set_index("particle_id")
    q = h["particle_id"] > 0
    for ax, (a, b, xl, yl) in zip(axes, [("z", "r", "z", "r"), ("x", "y", "x", "y")]):
        ax.scatter(h.loc[~q, a], h.loc[~q, b], s=1, c="lightgrey", label="other hits")
        for pid in np.unique(h.loc[q, "particle_id"]):
            m = h["particle_id"] == pid
            txt = f"quirk {pid}"
            if pid in p.index:
                txt += f" (q {p.at[pid, 'q']:+.0f}, pT {p.at[pid, 'pt']:.0f}, p {p.at[pid, 'p']:.0f} GeV)"
            ax.scatter(h.loc[m, a], h.loc[m, b], s=8, label=txt)
        ax.set_xlabel(f"{xl} [x0.01 mm]")
        ax.set_ylabel(f"{yl} [x0.01 mm]")
    title = f"{label}, {names[i]}"
    if len(p) and {"mass_gev", "lambda_ev"} <= set(p.columns):
        title += f"\nm {p['mass_gev'].iloc[0]:g} GeV, Lambda {p['lambda_ev'].iloc[0]:g} eV"
    axes[0].set_title(title)
    axes[0].legend(fontsize=7, markerscale=2)


def summary(name, df):
    rows = []
    for c in df.columns:
        if c == "event":
            continue
        x = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        f = x[np.isfinite(x)]
        flag = "CONSTANT" if len(f) and f.min() == f.max() else ""
        rows.append(f"  {c:24s} n {len(f):9d}  mean {np.mean(f) if len(f) else np.nan:11.4g}  "
                    f"std {np.std(f) if len(f) else np.nan:11.4g}  min {np.min(f) if len(f) else np.nan:11.4g}  "
                    f"max {np.max(f) if len(f) else np.nan:11.4g}  zero {np.mean(f == 0) if len(f) else np.nan:5.3f}  "
                    f"nan {len(x) - len(f):6d}  {flag}")
    return f"{name}\n" + "\n".join(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dirs", nargs="+", help="directories holding event*-hits/-parts.parquet")
    ap.add_argument("-o", "--outdir", required=True)
    ap.add_argument("--events", type=int, default=500, help="random events per directory (<=0: all)")
    ap.add_argument("--volumes", type=int, nargs="*",
                    help="keep only hits in these volume_ids, like the config's hit_volume_ids")
    ap.add_argument("--displays", type=int, default=4, help="event displays per directory, 4 per page")
    ap.add_argument("--display-seed", type=int,
                    help="pick the displayed events at random with this seed (default: the first ones loaded)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    data, text = [], []
    for k, d in enumerate(args.data_dirs):
        label = os.path.relpath(os.path.abspath(d), os.path.dirname(os.path.dirname(os.path.abspath(d))))
        hits, parts, names = load(d, args.events, args.volumes)
        n = len(names)
        data.append((label, colours[k % len(colours)], hits, parts, names))
        q = hits["particle_id"] > 0
        text.append(f"### {label}: {n} events, {len(hits)} hits ({q.mean():.4f} quirk hits), {len(parts)} particles")
        text.append(summary("hits (all):", hits))
        text.append(summary("particles:", parts))
        print(text[-3])

    hit_cols = [c for c in data[0][2].columns if c not in SKIP_HITS and c != "event"
                and pd.api.types.is_numeric_dtype(data[0][2][c])]
    hit_cols = [c for c in hit_cols if c not in DERIVED_HITS] + DERIVED_HITS
    part_cols = [c for c in data[0][3].columns if c not in SKIP_PARTS and c != "event"
                 and pd.api.types.is_numeric_dtype(data[0][3][c])]

    with PdfPages(os.path.join(args.outdir, "inputs.pdf")) as pdf:
        series = []
        for label, colour, hits, *_ in data:
            q = hits["particle_id"] > 0
            series.append((f"{label} quirk hits", colour, "-", hits[q]))
            series.append((f"{label} other hits", colour, ":", hits[~q]))
        hist_page(pdf, [c for c in hit_cols if c not in QUIRK_ONLY_HITS], series,
                  "Hit variables (file columns, then loader-derived r s theta phi eta u v)")
        qcols = [c for c in hit_cols if c in QUIRK_ONLY_HITS]
        if qcols:
            hist_page(pdf, qcols, [s for s in series if s[2] == "-"], "Per-quirk-hit trajectory targets")
        hist_page(pdf, part_cols, [(f"{label} particles", colour, "-", parts) for label, colour, _, parts, _ in data],
                  "Particle variables (file columns, then loader-derived p pt qopt eta phi)")
        quirk_pt_page(pdf, data)
        rng = np.random.default_rng(args.display_seed)
        for label, _, hits, parts, names in data:
            k = min(args.displays, len(names))
            shown = sorted(rng.choice(len(names), k, replace=False)) if args.display_seed is not None else range(k)
            event_displays(pdf, label, hits, parts, names, list(shown))
    with open(os.path.join(args.outdir, "summary.txt"), "w") as f:
        f.write("\n\n".join(text) + "\n")
    print(f"wrote {args.outdir}/inputs.pdf and summary.txt")


if __name__ == "__main__":
    main()
