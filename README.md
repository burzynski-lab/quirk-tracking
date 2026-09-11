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
| `quirks/configs/quirks-tracking-v2.yaml` | **recommended** — full ingredient stack, ABLATION-tagged |
| `quirks/configs/quirks-tracking-v2-cpu.yaml` | CPU inference/smoke variant (torch attention, no compile) |
| `quirks/configs/quirks-tracking-q4.yaml` | pre-v2 4-query baseline (ablation reference) |
| `quirks/configs/quirks-tracking.yaml` | original 8-query baseline (historical) |

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
- **No hit filtering**: nothing to prune at 2–3k spacepoints. The v2
  encoder task is an auxiliary classifier only (`mask_keys: false`) — it
  teaches the encoder which hits are quirks but removes nothing.
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
| `quirks/plot_quirk_eval.py` | event displays + regression truth/reco comparisons from an eval h5 |
| `quirks/configs/` | MaskFormer configs (see above) |
| `quirks/architecture.svg` | model diagram |
| `templates/`, `bootstrap.sh` | slurm scripts get generated into `slurm/` with your paths |
| `$SCRATCH` (ourdisk) | container, pixi env/caches, data, logs — **never in $HOME** (quota) |

## Next steps

Implemented and in the current config (see the ABLATION tags): per-hit
phase/arc-length supervision, encoder quirk-hit classifier, decoder depth 6,
ionization features, plane pair-consistency, mask-weighted coplanarity loss,
mask-loss rebalance (null_weight, dice), finding-first weighting, AdamW +
gradient clipping. Each is a one-block removal for ablation studies.

Open, in priority order:

1. **Prep quality cut: require both quirks to have >= 5 spacepoints.**
   Events where a quirk leaves fewer hits are barely reconstructable and
   dilute the mask supervision; keep the acceptance loss bookkept per
   Lambda (the removal rate is physics, not junk). Prep-flag change +
   re-prep.
2. **Purity curriculum**: warm-resume with `null_weight` raised (0.01 ->
   ~0.05) once recall is established. The low value is what escapes the
   claim-nothing collapse early; it also caps purity by making claimed
   background nearly free. The designated knob for the ~hundreds-of-hits
   mask plateau.
3. **Background-only events** — required before any physics claim: the
   model has never seen a quirk-free event, so its fake rate on SM
   background is unmeasured and the validity head is unfalsifiable.
   Cheap proxy: drop quirk-linked hits from signal events at prep time;
   real low-mu SM MC later. Needs the loader's empty-event guard relaxed.
4. **Pair-as-one-object**: one query per QQbar pair with a single mask and
   one (plane, f/m). Evals show the failure it targets is real: declared
   tracks are near-duplicates (mask IoU ~0.9) covering one arm's region
   rather than one query per arm. `merge_quirk_pair` in the data module
   already implements the target side; flip it and halve the queries.
5. **Attention-pooled regression inputs**: let the f/m and plane heads read
   a mask-weighted pooling of hit embeddings instead of the query vector
   alone — the oscillation scale lives in hit geometry. Relevant if the
   regressions stay near dataset-mean after finding converges.
6. **Physics decoder** (ambitious): regress trajectory parameters per
   query, render the analytic zig-zag differentiably, and use
   distance-to-trajectory as a mask/attention prior — locality defined
   along the physical path, not in phi.

Deliberately rejected: hit filtering (nothing to prune), windowed/phi-local
attention (wrong prior for back-to-back oscillating pairs), seeding queries
from innermost hits (`is_first` assumes helical tracks), truth trajectory as
an input (only exists at training time — it can only ever be supervision).

## Notes

- The container is only CUDA+glibc+pixi; the environment builds from
  `hepattn/pixi.lock` and installs hepattn editable from the submodule.
- Everything heavier than code lives on ourdisk; `$HOME` quota is tight.
- Dumps for m = 100 (all 10 Lambda points) are at
  `/ourdisk/hpc/ouhep/jburzyns/dont_archive/Quirks/chain/gnndump/`.
