# Quirk tracking with hepattn (MaskFormer) — plan

Goal: replace the invented flat eps_find = 0.5 with a measured quirk-pair
finder, using the MaskFormer encoder–decoder transformer from
[hepattn](https://github.com/samvanstroud/hepattn) on our GNN4ITk dumps.
The ExaTrkX/GNN line is pursued separately; this is the transformer track.

## Why this maps well

- MaskFormer treats reconstruction as object queries predicting hit masks —
  a quirk is just an object whose hit set is non-helical. No helix prior
  anywhere in the architecture.
- Per-object regression heads exist (`ObjectRegressionTask`), so regressing
  the identifiable trajectory combination **log10(Lambda^2/m)** and the
  **pair plane normal** are config-level additions, not framework changes.
- Our events are tiny by TrackML standards (~2–3k space points vs 100k), so
  the **hit-filtering stage is skipped** — the full event fits in one
  attention window (window_size 512 can even be raised or local attention
  disabled).

## Layout

| piece | where |
|---|---|
| hepattn clone (untouched) | `Tracking/hepattn/` (gitignored; pinned: 1df05cc) |
| our code | `Tracking/quirks/` — converter, configs, slurm |
| env, data, logs, containers | `/ourdisk/hpc/ouhep/jburzyns/dont_archive/Quirks/hepattn/` (home quota!) |

Environment: pixi inside the upstream CUDA apptainer image; `PIXI_HOME`,
`PIXI_CACHE_DIR`, `APPTAINER_CACHEDIR` and detached pixi envs all point at
ourdisk. Built by `Tracking/setup_env.slurm` (gpu partition; A100/L40S/H100
all present on schooner). GPU shell for interactive work:

```
srun -p gpu --gres gpu:1 --mem 100G --cpus-per-task 8 --pty bash
apptainer shell --nv --bind /ourdisk/hpc/ouhep/jburzyns,/home/jburzyns \
  /ourdisk/hpc/ouhep/jburzyns/dont_archive/Quirks/hepattn/containers/pixi.sif
export PIXI_HOME=... PIXI_CACHE_DIR=...   # see setup_env.slurm
cd quirk-chain/Tracking/hepattn && pixi shell
```

## Stages

**0. Environment** — `setup_env.slurm` (running). Verifies torch+CUDA+FA2.

**1. Data prep** — `Tracking/quirks/prep_quirks.py` (working, tested):
GNN4ITk dump -> per-event TrackML-schema parquet the stock
`TrackMLDataModule` reads unmodified. `--quirks-only` keeps only the two
quirks as targets (all other hits become noise) — stage-A setup where the
model's sole job is quirk finding. Extras carried per particle: `is_quirk`,
`mass_gev`, `lambda_ev`, `log10_f_over_m`, truth plane normal `nx,ny,nz`.
m=100 dumps for all 10 Lambda points (~390 GB) are on ourdisk under
`chain/gnndump/m100.L*/`; a prep sweep turns them into
`hepattn/data/m100/{train,val,test}` (split by file; ~90k events total).

**2. Baseline training** — `Tracking/quirks/configs/quirks-tracking.yaml`
(draft, validate once env is up): stock MaskFormer, 4 object queries,
mask + classification + IoU tasks as in TrackML, single GPU, CSV logger
(no comet account needed). Success metric: hit-mask efficiency/purity vs
the truth assignment, per Lambda — this is the measured eps_find.

**3. Physics extensions** (in order of value):
- **F/m regression**: `ObjectRegressionTask` on `log10_f_over_m` (in the
  draft config). Output seeds the analytic fit and removes the
  fixed-(m,Lambda) hypothesis from reconstruction (the hypothesis-free fit
  discussed for the contour).
- **Coplanarity**: two options, try in this order:
  (a) *auxiliary target*: regress the pair plane normal (`nx,ny,nz`, in the
  draft config) — pushes the representation to learn the plane;
  (b) *feature engineering*: add per-hit scalar-triple-product features or a
  pair-supervised contrastive loss on the embeddings (both arms pulled
  together). (b) needs small code additions in `Tracking/quirks/`.
- **Cluster charge**: `charge_frac` (summed ToT / pixels) is already in the
  hits; validated separation (saturation spike at high Lambda, dE/dx shift
  at high mass). Extend with max-ToT if useful.
- Later: mass scan mixing (m=100 first; add other masses to teach the
  F/m regression to generalize), full-tracking stage B (all particles as
  targets, event_max_num_particles ~2500).

**4. Evaluation** — per-(m,Lambda) pair-finding efficiency (both quirks'
masks >=X% pure/complete) -> replaces eps_find in the contour; regression
response for F/m; comparison against truth-seeded fit chi2.

## Student runbook (once env job finishes)

```
# 1. prep the data (CPU, ~1-2 h over all m100 files)
sbatch Tracking/quirks/prep_all.slurm
# 2. smoke test (GPU, minutes)
... pixi shell ...
cd Tracking/hepattn/src/hepattn/experiments/trackml
python run_tracking.py fit --config ../../../../../quirks/configs/quirks-tracking.yaml \
    --trainer.fast_dev_run 10
# 3. real training
sbatch Tracking/quirks/train.slurm
```

## Open questions

- 4 queries vs 2: null-query handling with tiny object counts; check
  matcher behavior.
- Strip SPs have no charge (`charge_frac` -> ToT only for pixels; strips get
  count-based placeholder). Consider a per-volume embedding.
- The 2 events/file skipped by prep are !=2-passed-quirk events (endcap
  losses); they are exactly the events the analysis loses at "2 quirks" —
  keep them out of training but remember them in the efficiency
  denominator.
