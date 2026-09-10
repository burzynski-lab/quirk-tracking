# quirk-tracking

MaskFormer-based quirk-pair finding for the ATLAS quirk search, built on
[hepattn](https://github.com/burzynski-lab/hepattn) (our fork, pinned as a
submodule). Input is the `Dump_GNN4Itk.root` ntuples from the quirk
reconstruction chain. See `PLAN.md` for the physics plan and
`quirks/architecture.svg` for the model diagram.

## Setup on schooner (OSCER)

```bash
# 1. clone (anywhere in $HOME; keep it small - all data/envs go to ourdisk)
git clone --recursive https://github.com/burzynski-lab/quirk-tracking.git
cd quirk-tracking

# 2. bake your paths into the slurm scripts (default scratch:
#    /ourdisk/hpc/ouhep/$USER/dont_archive/quirk-tracking)
./bootstrap.sh

# 3. build the environment (GPU node, ~30 min: container + pixi env)
sbatch slurm/setup_env.slurm      # wait for ENVDONE in the log

# 4. convert the GNN4ITk dumps to training parquet (~1-2 h)
sbatch slurm/prep.slurm           # wait for PREPDONE

# 5. smoke test (2 batches), then train
EXTRA="--trainer.fast_dev_run 10" sbatch --export=ALL slurm/train.slurm
sbatch slurm/train.slurm
```

Logging goes to Comet: `export COMET_API_KEY=...` in your shell before
submitting (get a key at comet.com). Without it, runs log offline.

## Configs

| file | use |
|---|---|
| `quirks/configs/quirks-tracking-q4.yaml` | **recommended** — 4 queries |
| `quirks/configs/quirks-tracking.yaml` | 8-query baseline (kept for reference) |

Empirical findings baked into these configs — do not undo them casually:

- **4 queries, not 8.** With quirks-only targets every event has ≤2 real
  objects; at 8 queries the validity head's cheapest solution is "everything
  is null" and it stays there for tens of epochs. At 4 queries (2 valid /
  2 null) it declares tracks from the first epoch. Do not drop to 2: the
  head then never sees a null example and is useless on background.
- **Full self-attention, no sliding window.** The 512-hit φ-window exists
  for 100k-hit TrackML events; ours have 150–6k hits, where full attention
  is cheap, the window path crashes on events shorter than the window, and
  the two quirk arms are back-to-back in φ so a φ-window would cut the very
  correlation we want.
- **`track_valid` BCE loss weight 1.0** (matching cost stays 0.1 so the
  Hungarian assignment remains mask-dominated).
- **No hit filter** (`encoder_tasks` empty): nothing to prune at 2–3k
  spacepoints. `encoder_loss = 0` on Comet is therefore expected.

## Evaluating a checkpoint

```bash
# writes <ckpt>_<split>_eval.h5 next to the checkpoint, then event displays
# (with analytic trajectory overlays) + regression truth/reco plots
python quirks/plot_quirk_eval.py EVAL.h5 DATA_DIR OUTDIR
```

The eval h5 comes from `run_tracking.py test --config ... --ckpt_path ...`
(the `PredictionWriter` callback in the config writes it). Trajectory
overlays need the `vx,vy,vz` columns, i.e. data prepared with the current
`prep_quirks.py`.

## Interactive GPU shell

```bash
srun -p ouheptmp --gres gpu:1 --mem 100G --cpus-per-task 8 --pty bash
source env.sh
export APPTAINER_TMPDIR=/tmp/$USER-apptainer PIXI_HOME=$SCRATCH/pixi-home PIXI_CACHE_DIR=$SCRATCH/pixi-cache
apptainer shell --nv --bind /ourdisk,/home $SCRATCH/containers/pixi.sif
cd hepattn && pixi shell
```

## Layout

| path | what |
|---|---|
| `hepattn/` | framework (submodule, our fork — do not edit upstream files casually; PR to the fork) |
| `quirks/prep_quirks.py` | GNN4ITk dump → per-event TrackML-schema parquet (+ quirk targets: `log10_f_over_m`, plane normal, production vertex; skips events the loader would reject) |
| `quirks/plot_quirk_eval.py` | event displays + regression truth/reco comparisons from an eval h5 |
| `quirks/configs/` | MaskFormer configs (see above) |
| `quirks/architecture.svg` | model diagram |
| `templates/`, `bootstrap.sh` | slurm scripts get generated into `slurm/` with your paths |
| `$SCRATCH` (ourdisk) | container, pixi env/caches, data, logs — **never in $HOME** (quota) |

## Roadmap: model improvements, in priority order

Targeting the current bottleneck (hit-mask learning) first:

1. **Per-hit phase/arc-length supervision** — the highest-value idea.
   At prep time, label every truth quirk hit with its arc length and
   oscillation phase along the analytic string trajectory (the tracer in
   `plot_quirk_eval.py` matches real hits to ~4–5 mm, good enough for
   labels). Add a query-conditioned per-hit regression head (hepattn's
   object–hit task machinery supports this) predicting the phase. This is
   dense supervision through the same (query × hit) pathway as the mask,
   forces the model to learn the *ordered periodic path* rather than an
   unordered hit set, and — since the phase rate is Λ²/m — triangulates the
   f/m regression from every hit pair instead of one scalar per track.
2. **Encoder auxiliary head**: per-hit "quirk hit" classifier in the empty
   `encoder_tasks` slot. Gives the encoder a direct gradient for separating
   quirk hits from background instead of relying on gradients through the
   decoder.
3. **Capacity**: decoder depth 3 → 6 (more mask-refinement iterations,
   each deep-supervised) and/or dim 256 → 384. Our events are 40× smaller
   than what the defaults were sized for; this costs minutes per epoch.
4. **Ionization features**: add absolute cluster charge and cluster size to
   the hit features. Quirks are slow (β* ~ 0.3–0.8) and heavily ionizing —
   strong per-hit discriminators the model currently never sees.
5. **Tie the pair's plane predictions**: both quirks share one plane; add a
   consistency loss between the two matched queries' normals (or predict
   the plane once from a symmetric pooling of both). Free physics
   constraint; also enables coplanarity-based hit cleaning at inference
   (prune claimed hits by distance to the predicted plane).
6. **Heteroscedastic f/m regression**: predict (μ, σ) with a Gaussian NLL.
   Resolution varies hugely across Λ (no visible oscillation at 10 keV);
   a learned σ stops unresolvable events dragging the head to the dataset
   mean and gives the downstream (m, Λ) fit per-event uncertainties.
7. **Mask-weighted coplanarity loss**: penalize claimed hits by distance to
   the predicted plane — couples the mask and geometry heads.
8. **Pair-as-one-object** (reserve): one query per QQ̄ pair with a single
   mask and one (plane, f/m). Eliminates arm-swapping between the two
   queries; adopt if evals show persistent cross-arm confusion.
9. **Physics decoder** (ambitious, after 1 hits its ceiling): regress
   trajectory parameters per query, render the analytic zig-zag
   differentiably, and use distance-to-trajectory as a mask/attention
   prior — locality defined along the physical path, not in φ.

Deliberately rejected: hit filtering (nothing to prune), windowed/φ-local
attention (wrong prior for back-to-back oscillating pairs), seeding queries
from innermost hits (`is_first` assumes helical tracks), truth trajectory as
an input (only exists at training time — it can only ever be supervision).

Also required before any physics claim: **background-only training/eval
events** — the model has never seen a quirk-free event, so its fake rate on
Standard Model background is unmeasured.

## Notes

- The container is only CUDA+glibc+pixi; the environment builds from
  `hepattn/pixi.lock` and installs hepattn editable from the submodule.
- Everything heavier than code lives on ourdisk; `$HOME` quota is tight.
- Dumps for m = 100 (all 10 Lambda points) are at
  `/ourdisk/hpc/ouhep/jburzyns/dont_archive/Quirks/chain/gnndump/`.
