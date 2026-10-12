# quirk-tracking

MaskFormer-based quirk-pair finding for the ATLAS quirk search, built on
[hepattn](https://github.com/burzynski-lab/hepattn) (our fork, pinned as a
submodule). Input is the `Dump_GNN4Itk.root` ntuples from the quirk
reconstruction chain. `PLAN.md` has the physics plan;
`quirks/architecture.svg` is the model diagram.

Everything runs inside an apptainer container on schooner (OSCER), with
[pixi](https://pixi.sh) managing the python environment inside it. There are
two ways to run any step:

- **batch** — `sbatch slurm/<step>.slurm`, which wraps the container call.
  Use this for real jobs.
- **interactive** — grab a node, enter the container, run `pixi run python
  ...` yourself. Use this for smoke tests and debugging.

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
ls slurm/                      # setup_env.slurm prep.slurm train.slurm eval.slurm
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
  though only `test_dir` is read — the data module builds all three.
- **Trajectory overlays need the `vx,vy,vz` columns**, i.e. data prepared
  with the current `prep_quirks.py`. Older parquet has none, and the
  displays fall back to straight lines from the origin.

---

## Configs

| file | use |
|---|---|
| `quirks/configs/quirks-tracking-v2.yaml` | the training config — use this one |
| `quirks/configs/quirks-tracking-v2-cpu.yaml` | CPU variant for inference and smoke tests (torch attention, no compile) |

To evaluate on CPU, use that config and add
`EXTRA="--trainer.accelerator cpu --trainer.devices 1 --trainer.precision 32"`.

## Layout

| path | what |
|---|---|
| `hepattn/` | the framework (submodule, our fork — PR changes to the fork, don't edit in place) |
| `quirks/prep_quirks.py` | GNN4ITk dump to per-event parquet |
| `quirks/plot_quirk_eval.py` | event displays + regression plots from an eval h5 |
| `quirks/inregion_auc.py`, `quirks/dup_metrics.py` | ranking and duplication diagnostics from an eval h5 |
| `quirks/configs/` | model configs |
| `templates/`, `bootstrap.sh` | `./bootstrap.sh` writes `slurm/` from these with your paths |
| `$SCRATCH` (ourdisk) | container, pixi environment, data, logs — **never in `$HOME`** (quota) |

If `sbatch slurm/...` says the file does not exist, `./bootstrap.sh` has not
been run, or was run from a different directory.
