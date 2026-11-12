#!/usr/bin/env python3
"""quirkml colliderML grid point (ODD detector) -> hepattn (TrackML-schema) parquet.

The colliderML counterpart of prep_quirks.py: reads one packaged quirkml dataset
(<point>/<version>/parquet, ColliderML layout: one row per event, list
columns) and writes per event event<N>-hits.parquet / event<N>-parts.parquet
for hepattn's TrackML data module, split into train/, val/ and
test/m<M>.L<L>/ under OUTDIR.

  hits:  x y z [mm], volume_id (ODD numbering: 16-18 pixel, 23-25 short
         strips, 28-30 long strips), particle_id (0 = noise), every ODD
         cluster column under its ODD name (loc0 loc1 var_loc0 var_loc1 time
         size_loc0 size_loc1 n_channels sum_activation local_eta local_phi
         global_eta global_phi eta_angle phi_angle), layer_id, surface_id,
         log1p_ of sum_activation, n_channels, size_loc0, size_loc1 (all
         long-tailed; in the v1 grid time, local_eta, local_phi, global_eta,
         global_phi, eta_angle and phi_angle are all zero), and
         s_arc [m], phase [turns] along the analytic quirk trajectory (as in
         prep_quirks.py; 0 for non-quirk hits)
  parts: particle_id (ODD particle_id + 1, since ODD numbers from 0 and
         hepattn reserves 0 for noise), odd_particle_id, px py pz [GeV], q,
         is_quirk, mass_gev, lambda_ev, log10_f_over_m, plane normal
         nx ny nz, vertex vx vy vz [mm]

Selection and split follow slurm/prep.slurm (grid_v3): only the two quirks
are kept as particles (--no-quirks-only keeps all), events need exactly two
quirks with pt > 5 GeV, |eta| < 2.5 and >= --min-hits hits, and the events
are split by sorted event id into --split fractions (train,val,test);
train/val take only points with Lambda <= --trainval-max-lambda. The packaged
events are numbered run by run, so the default 0.6,0.2,0.2 is runs 0-2 /
run 3 / run 4 for 5 runs.

  prep_colliderml.py <point>/v1/parquet OUTDIR --offset N [--split 0.6,0.2,0.2]
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep_quirks import PDG, quirk_trajectory_samples  # noqa: E402

HIT_FLOAT = ["loc0", "loc1", "var_loc0", "var_loc1", "time", "size_loc0", "size_loc1",
             "n_channels", "sum_activation", "local_eta", "local_phi", "global_eta",
             "global_phi", "eta_angle", "phi_angle"]
HIT_INT = ["volume_id", "layer_id", "surface_id"]
PART_COLS = ["particle_id", "pdg_id", "charge", "px", "py", "pz", "vx", "vy", "vz"]


class Lists:
    """A list column of a one-row-per-event table, sliced per event without Python lists."""

    def __init__(self, table, col):
        a = table.column(col).combine_chunks()
        self.off = a.offsets.to_numpy()
        self.inner = a.values
        if col == "particle_ids":     # list<list<id>>: keep the per-hit offsets too
            self.ioff = self.inner.offsets.to_numpy()
            self.flat = self.inner.values.to_numpy(zero_copy_only=False)
        else:
            self.vals = self.inner.to_numpy(zero_copy_only=False)

    def __call__(self, row):
        return self.vals[self.off[row]:self.off[row + 1]]

    def hit_particles(self, row):
        """(hit index, particle id) for every particle linked to every hit of the event."""
        h0, h1 = self.off[row], self.off[row + 1]
        io = self.ioff[h0:h1 + 1]
        return np.repeat(np.arange(h1 - h0), np.diff(io)), self.flat[io[0]:io[-1]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("point", help="a packaged quirkml dataset: <point>/<version>/parquet")
    ap.add_argument("outdir")
    ap.add_argument("--mass", type=float, help="GeV (default: from the point name m<M>_L<L>_mu<mu>)")
    ap.add_argument("--lam", "--lambda", dest="lam", type=float, help="eV (default: from the point name)")
    ap.add_argument("--offset", type=int, required=True,
                    help="event-number offset for unique names across points (event ids are < 100000)")
    ap.add_argument("--split", default="0.6,0.2,0.2", help="train,val,test fractions of the events")
    ap.add_argument("--trainval-max-lambda", type=float, default=3000.0,
                    help="points above this Lambda [eV] go to test only")
    ap.add_argument("--quirks-only", action=argparse.BooleanOptionalAction, default=True,
                    help="keep only the two quirks as particles (other hits are noise)")
    ap.add_argument("--exactly-two", action=argparse.BooleanOptionalAction, default=True,
                    help="require exactly two quirks passing pt/eta/min-hits")
    ap.add_argument("--min-hits", type=int, default=5)
    ap.add_argument("--events", type=int, default=-1, help="convert only the first N events (testing)")
    args = ap.parse_args()

    name = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(args.point))))
    m = re.match(r"m([\d.]+)_L([\d.]+)_mu\d+$", name)
    mass = args.mass if args.mass is not None else float(m.group(1))
    lam = args.lam if args.lam is not None else float(m.group(2))
    frac = np.array([float(f) for f in args.split.split(",")])
    if len(frac) != 3 or (frac < 0).any() or abs(frac.sum() - 1) > 1e-6:
        sys.exit(f"--split {args.split}: need three non-negative fractions adding up to 1")

    def table(sub):
        files = sorted(glob.glob(os.path.join(args.point, sub, "*.parquet")))
        if not files:
            sys.exit(f"no {sub} parquet under {args.point}")
        return pq.read_table(files)
    parts_t = table("truth/particles").select(["event_id"] + PART_COLS)
    hits_t = table("reco/tracker_hits").select(["event_id", "x", "y", "z", "particle_ids"] + HIT_FLOAT + HIT_INT)
    prow = {e: i for i, e in enumerate(parts_t.column("event_id").to_numpy())}
    hrow = {e: i for i, e in enumerate(hits_t.column("event_id").to_numpy())}
    P = {c: Lists(parts_t, c) for c in PART_COLS}
    H = {c: Lists(hits_t, c) for c in ["x", "y", "z", "particle_ids"] + HIT_FLOAT + HIT_INT}

    # split by sorted event id; the packaged events are numbered run by run
    events = np.array(sorted(set(prow) & set(hrow)))
    if args.events >= 0:
        events = events[:args.events]
    cut = np.round(np.cumsum(frac)[:2] * len(events)).astype(int)
    test_dir = f"test/m{mass:g}.L{lam:g}"
    split_of = np.where(np.arange(len(events)) < cut[0], "train",
                        np.where(np.arange(len(events)) < cut[1], "val", test_dir))
    if lam > args.trainval_max_lambda:
        split_of[split_of != test_dir] = ""        # no train/val events from high-Lambda points
    for s in set(split_of) - {""}:
        os.makedirs(os.path.join(args.outdir, s), exist_ok=True)

    written = {}
    for ev, split in zip(events, split_of):
        if not split:
            continue
        pr, hr = prow[ev], hrow[ev]
        pid_odd = P["particle_id"](pr).astype(np.int64)
        pdg = P["pdg_id"](pr)
        isq = np.abs(pdg) == PDG
        if isq.sum() != 2:
            continue
        keep = isq if args.quirks_only else np.ones_like(isq)
        px, py, pz = (P[c](pr)[keep].astype(np.float64) for c in ("px", "py", "pz"))
        parts = pd.DataFrame({
            "particle_id": pid_odd[keep] + 1,
            "odd_particle_id": pid_odd[keep],
            "px": px, "py": py, "pz": pz,
            "q": P["charge"](pr)[keep].astype(np.float32),
            "is_quirk": isq[keep],
            "mass_gev": mass,
            "lambda_ev": lam,
            "log10_f_over_m": np.log10(lam**2 / mass),
        })
        for c in "xyz":
            parts[f"v{c}"] = P[f"v{c}"](pr)[keep]

        # hit -> particle: the quirk if one of the hit's particles is a quirk,
        # else its first particle; particles not kept become noise (0)
        sp = np.stack([H[c](hr) for c in "xyz"], axis=1).astype(np.float64)
        hit_idx, linked = H["particle_ids"].hit_particles(hr)
        linked = linked.astype(np.int64) + 1
        pid = np.zeros(len(sp), dtype=np.int64)
        first = np.unique(hit_idx, return_index=True)
        pid[first[0]] = linked[first[1]]
        qids = pid_odd[isq] + 1
        isql = np.isin(linked, qids)
        pid[hit_idx[isql]] = linked[isql]
        pid[~np.isin(pid, parts["particle_id"].to_numpy())] = 0

        hits = pd.DataFrame({"x": sp[:, 0], "y": sp[:, 1], "z": sp[:, 2], "particle_id": pid})
        for c in HIT_FLOAT:
            hits[c] = H[c](hr).astype(np.float32)
        for c in HIT_INT:
            hits[c] = H[c](hr).astype(np.int32)
        for c in ("sum_activation", "n_channels", "size_loc0", "size_loc1"):
            hits[f"log1p_{c}"] = np.log1p(hits[c]).astype(np.float32)

        # the hepattn loader's particle cuts; exactly two quirks must survive
        pt = np.hypot(px, py)
        p3 = np.sqrt(px**2 + py**2 + pz**2)
        eta = np.arctanh(np.clip(pz / np.maximum(p3, 1e-12), -1 + 1e-12, 1 - 1e-12))
        counts = pd.Series(pid).value_counts()
        nhits = np.array([counts.get(b, 0) for b in parts["particle_id"]])
        passing = (pt > 5.0) & (np.abs(eta) < 2.5) & (nhits >= args.min_hits)
        if args.exactly_two:
            if (passing & parts["is_quirk"].to_numpy()).sum() != 2:
                continue
        elif not passing.any():
            continue

        # truth pair plane through the vertex (both quirks share it)
        iq = np.where(isq)[0][0]
        v = np.array([P[f"v{c}"](pr)[iq] for c in "xyz"], dtype=np.float64)
        qmask = np.isin(pid, qids)
        n = np.array([0.0, 0.0, 1.0])
        if qmask.sum() >= 3:
            _, _, vt = np.linalg.svd(sp[qmask] - v, full_matrices=False)
            n = vt[-1]
            if n[2] < 0:
                n = -n
        for i, c in enumerate("nx ny nz".split()):
            parts[c] = n[i]

        # per-hit position along the analytic quirk trajectory, as in prep_quirks.py
        hits["s_arc"] = np.float32(0.0)
        hits["phase"] = np.float32(0.0)
        qp = np.stack([P[c](pr)[isq].astype(np.float64) for c in ("px", "py", "pz")], axis=1)
        qE = np.sqrt((qp**2).sum(1) + mass**2)
        beta = qp.sum(0) / qE.sum()
        b2 = beta @ beta
        if 0 < b2 < 1:
            gam = 1.0 / np.sqrt(1.0 - b2)
            k = (gam - 1.0) / b2
            for j, qid in enumerate(qids):
                hm = pid == qid
                if not hm.any():
                    continue
                traj = quirk_trajectory_samples(qp[j], mass, lam, beta, gam, k, b2)
                if traj is None:
                    continue
                pts, s, phase = traj
                rel = sp[hm] - v
                near = np.argmin(((rel[:, None, :] - pts[None, :, :]) ** 2).sum(-1), axis=1)
                hits.loc[hm, "s_arc"] = (s[near] / 1000.0).astype(np.float32)
                hits.loc[hm, "phase"] = phase[near].astype(np.float32)

        stem = os.path.join(args.outdir, split, f"event{args.offset + int(ev):09d}")
        hits.to_parquet(stem + "-hits.parquet")
        parts.to_parquet(stem + "-parts.parquet")
        written[split] = written.get(split, 0) + 1

    summary = ", ".join(f"{s} {n}" for s, n in sorted(written.items()))
    print(f"{name} (m = {mass:g} GeV, Lambda = {lam:g} eV): {len(events)} events -> {summary or 'none'} in {args.outdir}")


if __name__ == "__main__":
    main()
