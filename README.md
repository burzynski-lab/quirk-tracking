# quirk-tracking

MaskFormer-based quirk-pair finding for the ATLAS quirk search, built on
[hepattn](https://github.com/burzynski-lab/hepattn) (our fork, pinned as a
submodule). Input is the `Dump_GNN4Itk.root` ntuples from the quirk
reconstruction chain. `PLAN.md` has the physics plan and
`quirks/architecture.svg` is the model diagram.

Everything runs inside an apptainer container on sooner (OSCER), with
[pixi](https://pixi.sh) managing the python environment inside it. There are
two ways to run any step:

- **batch**, with `sbatch slurm/<step>.slurm`, which wraps the container call.
  Use this for real jobs.
- **interactive**, where you grab a node, enter the container and run
  `pixi run python ...` yourself. Use this for smoke tests and debugging.

Both are given for each step below.

---

## 1. Setup (once)

```bash
# clone with the submodule, anywhere in $HOME (keep $HOME small: data,
# containers and environments all live on ourdisk)
git clone --recursive https://github.com/burzynski-lab/quirk-tracking.git
cd quirk-tracking

# generate slurm/ from templates/ with your paths baked in.
# slurm/ is NOT in the repo - it does not exist until you run this.
./bootstrap.sh                 # or: ./bootstrap.sh /my/own/scratch/dir
ls slurm/                      # setup_env.slurm prep.slurm prep_colliderml.slurm train.slurm eval.slurm
cat env.sh                     # the QT (repo) and SCRATCH (ourdisk) paths it used

# build the container and the pixi environment (~30 min)
sbatch slurm/setup_env.slurm   # finished when the log prints ENVDONE
```

Comet logging: `export COMET_API_KEY=...` in your shell before submitting
anything (get a key at comet.com). Without it, runs log offline.

## 2. Prepare the data (once per dataset)

Converts the GNN4ITk dumps to per-event parquet, one array task per mass,
1-2 h each:

```bash
sbatch slurm/prep.slurm        # finished when each log prints PREP_M<mass>_DONE
```

This writes `$SCRATCH/data/grid_v3/` with `train/`, `val/` and `test/`,
where `test/` holds one subdirectory per (mass, Lambda) point. Use other
dumps with `--export=ALL,DUMPDIR=/path/to/dumps`.

## 3. Training

**Batch:**

```bash
# smoke test first: 10 batches, a couple of minutes
sbatch --export=ALL,EXTRA="--trainer.fast_dev_run 10" slurm/train.slurm

# the real thing
sbatch --export=ALL,NAME=my-run slurm/train.slurm

# more GPUs: ask slurm for them AND tell lightning about them
sbatch --gres=gpu:2 --export=ALL,NGPU=2,NAME=my-run slurm/train.slurm
```

Options, all via `--export=ALL,VAR=...`: `DATA` (dataset root, default
`$SCRATCH/data/grid_v3`), `NAME` (Comet experiment name), `EPOCHS` (default
50), `NGPU`, `EXTRA` (flags passed straight through to the CLI).

Results go to `$SCRATCH/logs/<NAME>_<timestamp>/`, checkpoints in its
`ckpts/` subdirectory.

**Interactive:**

```bash
srun -p ouhep_gpu --gres gpu:1 --mem 100G --cpus-per-task 8 --pty bash
source env.sh
export APPTAINER_TMPDIR=/tmp/$USER-apptainer \
       PIXI_HOME=$SCRATCH/pixi-home PIXI_CACHE_DIR=$SCRATCH/pixi-cache
apptainer shell --nv --bind /ourdisk,/home $SCRATCH/containers/pixi.sif

cd $QT/hepattn
pixi run python src/hepattn/experiments/trackml/run_tracking.py fit \
  --config $QT/quirks/configs/quirks-tracking-v2.yaml \
  --data.train_dir $SCRATCH/data/grid_v3/train \
  --data.val_dir   $SCRATCH/data/grid_v3/val \
  --data.test_dir  $SCRATCH/data/grid_v3/test \
  --trainer.devices 1 --trainer.max_epochs 50 \
  --trainer.default_root_dir $SCRATCH/logs
```

`pixi shell` in place of `pixi run` drops you inside the environment, after
which plain `python ...` works.

## 4. Evaluation

Two steps: `run_tracking.py test` writes an h5 of predictions and targets,
then the analysis scripts read that h5. `slurm/eval.slurm` does the first,
and the second as well if you pass `PLOTS=1`.

**Batch:**

```bash
CKPT=$SCRATCH/logs/<run>/ckpts/epoch=012-val_loss=....ckpt

# the whole test split
sbatch --export=ALL,CKPT=$CKPT slurm/eval.slurm

# the same, plus event displays and regression plots into <run>/plots
sbatch --export=ALL,CKPT=$CKPT,PLOTS=1 slurm/eval.slurm

# the per-(mass, Lambda) efficiency grid: one array task per test/ point
sbatch --array=0-108 --export=ALL,CKPT=$CKPT slurm/eval.slurm
```

**Interactive** (in the container shell from step 3):

```bash
D=$SCRATCH/data/grid_v3/test
cd $QT/hepattn
pixi run python src/hepattn/experiments/trackml/run_tracking.py test \
  --config $QT/quirks/configs/quirks-tracking-v2.yaml \
  --data.train_dir $D --data.val_dir $D --data.test_dir $D \
  --ckpt_path $CKPT
```

The h5 lands next to the checkpoint, named

```
<ckpt dir>/<ckpt name>_<basename of the data dir>_eval.h5
```

so evaluating `test/m100.L464/` gives `..._m100.L464_eval.h5`, and grid
points never overwrite each other.

Then read it:

```bash
pixi run python $QT/quirks/plot_quirk_eval.py EVAL.h5 DATA_DIR OUTDIR [--max-displays 50]
pixi run python $QT/quirks/inregion_auc.py    EVAL.h5 [--topk 200]
pixi run python $QT/quirks/dup_metrics.py     EVAL.h5
```

`plot_quirk_eval.py` draws event displays and truth/reco regression plots,
`inregion_auc.py` reports per-Lambda region capture and ranking, and
`dup_metrics.py` reports duplicate masks and the de-duplicated efficiency.

### Three things that break evaluation

- **`--config` must be the config the checkpoint was trained with.** A
  different architecture will not load the weights.
- **Pass all three data dirs** (`train_dir`, `val_dir`, `test_dir`) even
  though only `test_dir` is read, because the data module builds all three.
- **Trajectory overlays need the `vx,vy,vz` columns**, i.e. data prepared
  with the current `prep_quirks.py`. Older parquet has none, and the
  displays fall back to straight lines from the origin.

## 5. The colliderML dataset (quirkml grid, ODD detector)

Sections 2-4 use the GNN4ITk dumps (`grid_v3`). The colliderML dataset is
the quirkml grid simulated in the Open Data Detector (ODD): 11 masses
(50-300 GeV, steps of 25) x 10 Lambda (100-10000 eV), mu = 0, 110 points.
It has its own prep script and config; the training and evaluation scripts
are the same, you only point them at it.

### Prepare

`quirks/prep_colliderml.py` converts each packaged quirkml point
(`<point>/v1/parquet`) to hepattn parquet, one array task per point,
CPU only:

```bash
sbatch slurm/prep_colliderml.slurm   # each log prints PREP_COLLIDERML_DONE
```

It writes `$SCRATCH/data/colliderml_v1/` with:

- **Selection:** events with exactly two quirks, each with pT > 5 GeV,
  |eta| < 2.5 and >= 5 hits. Only the quirks are kept as particles; all
  other hits stay as noise (`particle_id` 0).
- **Split:** per point, events sorted by id, 60/20/20 into `train/`, `val/`
  and `test/m<M>.L<L>/` (with 5 runs per point: runs 0-2 / 3 / 4).
- **High Lambda held out:** only points with Lambda <= 3000 eV go into
  `train/` and `val/` (77 points). The Lambda = 3594, 5995 and 10000 points
  write only their 20% test share, so they test generalisation to string
  tensions the model never trained on.

Options via `--export=ALL,VAR=...`: `GRID` (quirkml datasets root),
`VERSION` (default `v1`), `OUT` (default `$SCRATCH/data/colliderml_v1`),
`EXTRA` (flags for `prep_colliderml.py`, e.g. `--split 0.8,0.1,0.1` or
`--trainval-max-lambda 10000`). Write a changed recipe to a new `OUT`
(`colliderml_v2`, ...) rather than over `colliderml_v1`.

### Train

Same `train.slurm`, with `DATA` and `CONFIG` pointing at colliderML. Run
`source env.sh` first so `$SCRATCH` and `$QT` are set; if either is empty,
the job fails with "num_events must be greater than 0" or "File does not
exist".

```bash
source env.sh

# smoke test: 10 batches, ~3 min
sbatch --export=ALL,DATA=$SCRATCH/data/colliderml_v1,CONFIG=$QT/quirks/configs/quirks-tracking-colliderml.yaml,NAME=cml-smoke,EXTRA="--trainer.fast_dev_run 10" slurm/train.slurm

# full training
sbatch --export=ALL,DATA=$SCRATCH/data/colliderml_v1,CONFIG=$QT/quirks/configs/quirks-tracking-colliderml.yaml,NAME=my-run slurm/train.slurm

# resume from a checkpoint (continues at the next epoch, optimizer state included)
sbatch --export=ALL,DATA=$SCRATCH/data/colliderml_v1,CONFIG=$QT/quirks/configs/quirks-tracking-colliderml.yaml,NAME=my-run,EXTRA="--ckpt_path $SCRATCH/logs/<run>/ckpts/epoch=NNN-val_loss=....ckpt" slurm/train.slurm
```

For reference, the first full run (`odd-fullstat`, 33,798 train / 11,301
val events, 1 GPU on `ouhep_gpu`) took ~2 h 15 min per epoch; val loss was
best at epoch 39 and rose after ~41, so ~40 epochs is enough
(`EPOCHS=40`). `EPOCHS` (default 50) overrides `max_epochs` in the config.

### Evaluate

Same `eval.slurm`, with `DATA` and `CONFIG` set as for training. The
colliderML `test/` has 110 point directories, hence `--array=0-109`:

```bash
sbatch --array=0-109 --export=ALL,CKPT=$CKPT,DATA=$SCRATCH/data/colliderml_v1,CONFIG=$QT/quirks/configs/quirks-tracking-colliderml.yaml slurm/eval.slurm
```

## 6. Plotting the input variables

`quirks/plot_inputs.py` plots what the network sees: every column of the
prepared parquet plus the variables hepattn's loader derives on load (r, s,
theta, phi, eta, u, v for hits; p, pt, qopt, eta, phi for particles). It
works on both colliderML and GNN4ITk output and takes any number of data
directories, each drawn in its own colour.

It writes two files into `OUTDIR`:

- `inputs.pdf`:
  - hit variables, quirk hits (solid) vs other hits (dotted)
  - per-quirk-hit trajectory targets (`s_arc`, `phase`)
  - particle variables
  - quirk pT by charge (quirk+ vs quirk-) and per event (higher vs lower pT)
  - r-z and x-y event displays, with each quirk's charge, pT and p in the
    legend, 4 per page
- `summary.txt`: per-column mean, std, min, max and zero/NaN fractions.
  It flags constant columns (e.g. the all-zero ODD angle columns).

```
plot_inputs.py DATA_DIR [DATA_DIR ...] -o OUTDIR
               [--events 500]        # random events per directory (<= 0: all)
               [--volumes 16 17 ...] # keep only these volume_ids, like the config's hit_volume_ids
               [--displays 4]        # event displays per directory
               [--display-seed SEED] # pick the displayed events at random (default: the first ones)
```

**Batch-free, fastest:** the plotter needs no GPU and runs in ~2 min on a
login node by calling the environment's python directly, which skips
`pixi run`'s start-up check:

```bash
source env.sh
PY="apptainer exec --bind /ourdisk,/scratch,/home $SCRATCH/containers/pixi.sif \
    $SCRATCH/envs/hepattn-14158428132650205534/envs/default/bin/python"
VOLS="--volumes 16 17 18 23 24 25 28 29 30"   # the colliderML config's hit_volume_ids
D=$SCRATCH/data/colliderml_v1

# train vs val vs two test points
$PY $QT/quirks/plot_inputs.py $D/train $D/val $D/test/m50.L100 $D/test/m300.L10000 \
    -o $SCRATCH/plots/cml_inputs $VOLS

# one point, all its events, 20 random displays
$PY $QT/quirks/plot_inputs.py $D/test/m50.L100 -o $SCRATCH/plots/m50_L100 \
    --events 0 --displays 20 --display-seed 1 $VOLS
```

The `envs/hepattn-<hash>` name is fixed by pixi; `ls $SCRATCH/envs/` shows
it. Inside an interactive container shell (section 3),
`pixi run python $QT/quirks/plot_inputs.py ...` works too.

To plot another (mass, Lambda) point, pass its `test/m<M>.L<L>` directory.
High-Lambda points (Lambda > 3000) exist only under `test/`.

---

## Configs

| file | use |
|---|---|
| `quirks/configs/quirks-tracking-v2.yaml` | the training config, use this one |
| `quirks/configs/quirks-tracking-v2-cpu.yaml` | CPU variant for inference and smoke tests (torch attention, no compile) |
| `quirks/configs/quirks-tracking-colliderml.yaml` | the training config for the colliderML dataset (section 5) |

To evaluate on CPU, use that config and add
`EXTRA="--trainer.accelerator cpu --trainer.devices 1 --trainer.precision 32"`.

## Layout

| path | what |
|---|---|
| `hepattn/` | the framework (submodule, our fork. PR changes to the fork, do not edit in place) |
| `quirks/prep_quirks.py` | GNN4ITk dump to per-event parquet |
| `quirks/prep_colliderml.py` | quirkml colliderML (ODD) grid point to per-event parquet |
| `quirks/plot_inputs.py` | input-variable histograms, quirk pT and event displays from prepared parquet |
| `quirks/plot_quirk_eval.py` | event displays + regression plots from an eval h5 |
| `quirks/inregion_auc.py`, `quirks/dup_metrics.py` | ranking and duplication diagnostics from an eval h5 |
| `quirks/configs/` | model configs |
| `templates/`, `bootstrap.sh` | `./bootstrap.sh` writes `slurm/` from these with your paths |
| `$SCRATCH` (ourdisk) | container, pixi environment, data, logs. **Never in `$HOME`** (quota) |

If `sbatch slurm/...` says the file does not exist, `./bootstrap.sh` has not
been run, or was run from a different directory.
