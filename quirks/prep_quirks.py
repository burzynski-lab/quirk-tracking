#!/usr/bin/env python3
"""GNN4ITk dump -> hepattn (TrackML-schema) parquet.

Per event writes event<N>-hits.parquet / event<N>-parts.parquet compatible
with hepattn's TrackML data module:
  hits:  x y z [mm], volume_id (8=pixel, 9=strip), particle_id (0 = not a
         kept particle), cluster features (charge_frac, leta, lphi, lx, ly,
         lz, geta, gphi from the dump's cluster block)
  parts: particle_id, px py pz [GeV], q, plus quirk extras:
         is_quirk, mass_gev, lambda_ev, log10_f_over_m (the identifiable
         trajectory combination Lambda^2/m), plane normal nx ny nz (truth
         pair plane through the vertex; identical for both quirks)

--quirks-only keeps only the two quirks in parts (all other hits become
particle_id 0, i.e. noise) - stage-A configuration where the model's sole
job is quirk finding.

v2 schema additions (all consumed via config, ablation-friendly):
  hits:  cl_charge (raw cluster charge count, log1p), cl_size (pixel count)
         - ionization observables, strong for slow heavily-ionising quirks;
         s_arc [m], phase [turns 0-1]: position of each QUIRK hit along the
         analytic string trajectory (nearest point on the integrable
         zig-zag anchored at the vertex; ~5 mm faithful). 0 for other hits.
  parts: vx vy vz [mm] production vertex.

  prep_quirks.py DUMP.root OUTDIR --mass 100 --lambda 464 [--quirks-only]
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
import uproot

PDG = 10000100
HBARC_GEV_MM = 1.9732705e-13  # hbar*c in GeV*mm


def quirk_trajectory_samples(p3, m, lam_ev, beta, gam, k, b2, n_period=6, n_samp=12000):
    """Analytic lab trajectory of one quirk relative to its vertex.

    Returns (pts[mm] (N,3), s_arc[mm] (N,), phase[turns] (N,)). Same
    integrable string-only motion as quirk_trajectory.py: constant force
    F = Lambda^2 along the pair axis in the pair rest frame, boosted to lab.
    """
    bp = beta @ p3
    E = np.sqrt(p3 @ p3 + m * m)
    prest = p3 + (k * bp - gam * E) * beta
    p0 = np.linalg.norm(prest)
    if p0 <= 0:
        return None
    nhat = prest / p0
    F = (lam_ev * 1e-9) ** 2
    tau = 4.0 * p0 / F
    E0 = np.sqrt(p0**2 + m**2)
    t = np.linspace(0.0, n_period * tau, n_samp)
    ph = (t % tau) * F
    pmag = np.where(ph <= 2 * p0, p0 - ph, ph - 3 * p0)
    xmag = np.where(ph <= 2 * p0, 1.0, -1.0) * (E0 - np.sqrt(pmag**2 + m**2)) / F
    X = xmag[:, None] * nhat[None, :]
    bx = X @ beta
    lab = (X + (k * bx + gam * t)[:, None] * beta[None, :]) * HBARC_GEV_MM
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(lab, axis=0), axis=1))])
    return lab, s, (t % tau) / tau


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("outdir")
    ap.add_argument("--mass", type=float, required=True, help="GeV")
    ap.add_argument("--lam", "--lambda", dest="lam", type=float, required=True, help="eV")
    ap.add_argument("--quirks-only", action="store_true")
    ap.add_argument("--min-hits", type=int, default=3,
                    help="skip events where no kept particle would survive the "
                         "hepattn loader cuts (pt>5, |eta|<2.5, >= this many hits)")
    ap.add_argument("--events", type=int, default=-1)
    ap.add_argument("--offset", type=int, default=0,
                    help="event-number offset for unique names across files")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    t = uproot.open(args.dump)["GNN4ITk"]
    n_ev = t.num_entries if args.events < 0 else min(args.events, t.num_entries)
    br = t.arrays(["SPx", "SPy", "SPz", "SPCL1_index", "CLhardware",
                   "CLcharge_count", "CLpixel_count", "CLloc_eta", "CLloc_phi",
                   "CLloc_direction1", "CLloc_direction2", "CLloc_direction3",
                   "CLglob_eta", "CLglob_phi",
                   "CLparticleLink_barcode", "Part_barcode", "Part_pdg_id",
                   "Part_passed", "Part_px", "Part_py", "Part_pz",
                   "Part_vx", "Part_vy", "Part_vz"], entry_stop=n_ev)

    n_written = 0
    for ev in range(n_ev):
        bar = np.asarray(br["Part_barcode"][ev])
        pdg = np.asarray(br["Part_pdg_id"][ev])
        passed = np.asarray(br["Part_passed"][ev]) == 1
        isq = (np.abs(pdg) == PDG) & passed
        if isq.sum() != 2:
            continue

        keep = isq if args.quirks_only else passed
        px = np.asarray(br["Part_px"][ev])[keep] / 1000.0
        py = np.asarray(br["Part_py"][ev])[keep] / 1000.0
        pz = np.asarray(br["Part_pz"][ev])[keep] / 1000.0
        parts = pd.DataFrame({
            "particle_id": bar[keep].astype(np.int64),
            "px": px, "py": py, "pz": pz,
            "q": np.sign(pdg[keep]) * 0 + 1.0,   # placeholder; unused by tasks
            "is_quirk": (np.abs(pdg[keep]) == PDG),
            "mass_gev": args.mass,
            "lambda_ev": args.lam,
            "log10_f_over_m": np.log10(args.lam**2 / args.mass),
        })
        # production vertex (beamspot-smeared; needed to anchor analytic
        # trajectory overlays -- z spread is tens of mm)
        for c in "xyz":
            parts[f"v{c}"] = np.asarray(br[f"Part_v{c}"][ev])[keep]

        # hits
        sp = np.stack([np.asarray(br["SPx"][ev]), np.asarray(br["SPy"][ev]),
                       np.asarray(br["SPz"][ev])], axis=1)
        cl1 = np.asarray(br["SPCL1_index"][ev])
        clb = br["CLparticleLink_barcode"][ev]
        clb0 = np.array([b[0] if len(b) else 0 for b in clb], dtype=np.int64)
        pid = clb0[cl1]
        pid[~np.isin(pid, parts["particle_id"].to_numpy())] = 0

        hw = np.asarray(br["CLhardware"][ev])[cl1] == "PIXEL"
        chg = np.asarray(br["CLcharge_count"][ev], dtype=np.float32)[cl1]
        npix = np.asarray(br["CLpixel_count"][ev], dtype=np.float32)[cl1]
        hits = pd.DataFrame({
            "x": sp[:, 0], "y": sp[:, 1], "z": sp[:, 2],
            "volume_id": np.where(hw, 8, 9).astype(np.int32),
            "particle_id": pid,
            "charge_frac": chg / np.maximum(npix, 1.0),
            "cl_charge": np.log1p(chg),     # ionization: raw cluster charge
            "cl_size": npix,                # ionization: cluster pixel count
            "leta": np.asarray(br["CLloc_eta"][ev], dtype=np.float32)[cl1],
            "lphi": np.asarray(br["CLloc_phi"][ev], dtype=np.float32)[cl1],
            "lx": np.asarray(br["CLloc_direction1"][ev], dtype=np.float32)[cl1],
            "ly": np.asarray(br["CLloc_direction2"][ev], dtype=np.float32)[cl1],
            "lz": np.asarray(br["CLloc_direction3"][ev], dtype=np.float32)[cl1],
            "geta": np.asarray(br["CLglob_eta"][ev], dtype=np.float32)[cl1],
            "gphi": np.asarray(br["CLglob_phi"][ev], dtype=np.float32)[cl1],
        })

        # skip events the hepattn loader would reject with "No particles
        # remaining": every kept particle must fail pt>5 / |eta|<2.5 /
        # min-hits for that to happen, so require at least one survivor
        pt = np.hypot(px, py)
        p3 = np.sqrt(px**2 + py**2 + pz**2)
        eta = np.arctanh(np.clip(pz / np.maximum(p3, 1e-12), -1 + 1e-12, 1 - 1e-12))
        counts = pd.Series(pid).value_counts()
        nhits = np.array([counts.get(b, 0) for b in parts["particle_id"]])
        if not ((pt > 5.0) & (np.abs(eta) < 2.5) & (nhits >= args.min_hits)).any():
            continue

        # truth pair plane through the vertex (both quirks share it)
        iq = np.where(isq)[0][0]
        v = np.array([br["Part_vx"][ev][iq], br["Part_vy"][ev][iq],
                      br["Part_vz"][ev][iq]])
        qmask = np.isin(pid, bar[isq])
        n = np.array([0.0, 0.0, 1.0])
        if qmask.sum() >= 3:
            _, _, vt = np.linalg.svd(sp[qmask] - v, full_matrices=False)
            n = vt[-1]
            if n[2] < 0:
                n = -n
        for i, c in enumerate("nx ny nz".split()):
            parts[c] = n[i]

        # per-hit position along the analytic quirk trajectory: arc length
        # s_arc [m] and oscillation phase [turns]; 0 for non-quirk hits
        hits["s_arc"] = np.float32(0.0)
        hits["phase"] = np.float32(0.0)
        qpx = np.asarray(br["Part_px"][ev])[isq] / 1000.0
        qpy = np.asarray(br["Part_py"][ev])[isq] / 1000.0
        qpz = np.asarray(br["Part_pz"][ev])[isq] / 1000.0
        qE = np.sqrt(qpx**2 + qpy**2 + qpz**2 + args.mass**2)
        P = np.array([qpx.sum(), qpy.sum(), qpz.sum()])
        Etot = qE.sum()
        beta = P / Etot
        b2 = beta @ beta
        if 0 < b2 < 1:
            gam = 1.0 / np.sqrt(1.0 - b2)
            k = (gam - 1.0) / b2
            for j, qbar in enumerate(bar[isq]):
                hm = pid == qbar
                if not hm.any():
                    continue
                traj = quirk_trajectory_samples(
                    np.array([qpx[j], qpy[j], qpz[j]]), args.mass, args.lam,
                    beta, gam, k, b2)
                if traj is None:
                    continue
                pts, s, phase = traj
                rel = sp[hm] - v
                near = np.argmin(
                    ((rel[:, None, :] - pts[None, :, :]) ** 2).sum(-1), axis=1)
                hits.loc[hm, "s_arc"] = (s[near] / 1000.0).astype(np.float32)
                hits.loc[hm, "phase"] = phase[near].astype(np.float32)

        name = f"event{args.offset + ev:09d}"
        hits.to_parquet(os.path.join(args.outdir, name + "-hits.parquet"))
        parts.to_parquet(os.path.join(args.outdir, name + "-parts.parquet"))
        n_written += 1
    print(f"{os.path.basename(args.dump)}: wrote {n_written}/{n_ev} events -> {args.outdir}")


if __name__ == "__main__":
    main()
