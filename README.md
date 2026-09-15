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
| `quirks/configs/quirks-tracking-v2.yaml` | **the** training config — full ingredient stack, ABLATION-tagged |
| `quirks/configs/quirks-tracking-v2-cpu.yaml` | CPU inference/smoke variant of the same (torch attention, no compile) |

Earlier configs (8-query original, 4-query baseline) are in git history;
their findings are captured below rather than kept as files.

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
- **No hit filtering yet**: not needed for memory/compute at 2–3k
  spacepoints (its original purpose at TrackML scale); whether it is
  needed for *physics* — reducing the background the mask head must
  reject — is an open question (see Next steps). The v2 encoder task is
  an auxiliary classifier only (`mask_keys: false`); it removes nothing.
- **AdamW with gradient clipping**, not Lion: sign-based updates cannot
  accumulate the sign-noisy per-event gradients of diverse data once the
  collapse directions are priced out of the loss.

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
| `quirks/plot_quirk_eval.py` | event displays (trajectory drawn to the figure edge) + regression truth/reco comparisons from an eval h5 |
| `quirks/inregion_auc.py`, `quirks/dup_metrics.py` | per-Lambda ranking/capture and duplication diagnostics from an eval h5 |
| `quirks/prep_grid_v3.slurm` | the all-mass dataset recipe (train Lambda<=3000, per-point test dirs) |
| `quirks/configs/` | MaskFormer configs (see above) |
| `quirks/architecture.svg` | model diagram |
| `templates/`, `bootstrap.sh` | slurm scripts get generated into `slurm/` with your paths |
| `$SCRATCH` (ourdisk) | container, pixi env/caches, data, logs — **never in $HOME** (quota) |

## Next steps

Implemented and in the current config (see the ABLATION tags): per-hit
phase/arc-length supervision (weight 20), encoder quirk-hit classifier,
decoder depth 6, ionization features, plane pair-consistency, mask-weighted
coplanarity loss, mask-loss rebalance (null_weight 0.05, dice 10),
finding-first weighting, AdamW + gradient clipping, exactly-2-quirks x
>=5-spacepoints prep selection, all-mass `grid_v3` dataset (train on
Lambda <= 3000 eV, per-point test dirs for the efficiency grid).
Each is a one-block removal for ablation studies.

Tooling in `quirks/`: `inregion_auc.py` (region capture vs. within-region
ranking, per Lambda), `dup_metrics.py` (pairwise pred-mask IoU, duplicate
fraction, de-duplicated efficiency).

Open, in priority order:

1. **Fix the plane-normal target — it has never learned.** `track_plane`
   smooth-L1 sits at the constant-predictor value (0.167) in every run.
   Cause: the label's sign convention (nz >= 0) is nearly random for our
   mostly-transverse normals, so n and -n label the same plane and the
   loss averages to the mean. Fix: a sign-free representation — plane
   azimuth as (cos 2phi0, sin 2phi0) plus |nz| — or a sign-symmetric loss
   (min over +-n). Prep column + loss change. The coplanarity and
   pair-consistency terms read this head, so fixing it may unlock both.
2. **Mask orthogonality**: ~55% of multi-track events are duplicates
   (pairwise pred-mask IoU > 0.5) — one merged-pair mask copied into both
   slots. Finding ("pair found", ~0.5-0.6) is honest, but arm separation
   is unsolved and no logged metric sees it. Add a pairwise soft-IoU
   penalty between valid queries' masks (truth masks are disjoint, so zero
   overlap is the correct target; small custom task like the coplanarity
   one). Success = mid-Lambda dup_frac falling without high-Lambda
   efficiency loss (at Lambda >= 2 keV the arms are physically merged).
   Also add a strict one-to-one efficiency to the val metrics.
3. **Background-only events** — required before any physics claim: the
   model has never seen a quirk-free event, so its fake rate on SM
   background is unmeasured and the validity head is unfalsifiable.
   Cheap proxy: drop quirk-linked hits from signal events at prep time;
   real low-mu SM MC later. Needs the loader's empty-event guard relaxed.
4. **Hit filtering for physics (not memory)**: the mask head faces ~2k
   background hits per ~10 true; a filtering stage could cut that load.
   Two escalation levels: flip the encoder classifier to `mask_keys: true`
   (soft in-model filter; one config line — the classifier now rejects
   ~69% of background at 94% quirk-hit recall, so this is becoming
   viable) or a separate upstream filter model pruning in the dataloader
   via `hit_eval_path`. Any hard filter risks irrecoverable efficiency
   loss — measure the filter's own quirk-hit efficiency first.
5. **Batched training**: batch size 1 leaves >95% of each step as
   overhead (dataloading, scipy matcher, Python loss loops, launch
   latency); a padded-collate for variable-length events would plausibly
   give 2-4x on the same GPUs. Framework work in the fork; more valuable
   than faster hardware.
6. **Attention-pooled regression inputs**: let the f/m and plane heads read
   a mask-weighted pooling of hit embeddings instead of the query vector
   alone — the oscillation scale lives in hit geometry. f/m is now
   learning (val residual 0.57 and falling) so this is lower priority
   than it was; revisit if it plateaus.
7. **Pair-as-one-object** (reserve): one query per QQbar pair with a
   single mask and one (plane, f/m). Dissolves the duplication problem
   rather than penalising it; adopt if item 2 fails. `merge_quirk_pair`
   in the data module already implements the target side.
8. **Physics decoder** (ambitious): regress trajectory parameters per
   query, render the analytic zig-zag differentiably, and use
   distance-to-trajectory as a mask/attention prior — locality defined
   along the physical path, not in phi.

Deliberately rejected: windowed/phi-local attention (wrong prior for
back-to-back oscillating pairs), seeding queries from innermost hits
(`is_first` assumes helical tracks), truth trajectory as an input (only
exists at training time — it can only ever be supervision).

## Notes

- The container is only CUDA+glibc+pixi; the environment builds from
  `hepattn/pixi.lock` and installs hepattn editable from the submodule.
- Everything heavier than code lives on ourdisk; `$HOME` quota is tight.
- Dumps for m = 100 (all 10 Lambda points) are at
  `/ourdisk/hpc/ouhep/jburzyns/dont_archive/Quirks/chain/gnndump/`.
