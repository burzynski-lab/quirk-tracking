# quirk-tracking

MaskFormer-based quirk-pair finding for the ATLAS quirk search, built on
[hepattn](https://github.com/burzynski-lab/hepattn) (our fork, pinned as a
submodule). Input is the `Dump_GNN4Itk.root` ntuples from the quirk
reconstruction chain. See `PLAN.md` for the physics plan.

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
| `quirks/prep_quirks.py` | GNN4ITk dump → per-event TrackML-schema parquet (+ quirk targets: `log10_f_over_m`, plane normal) |
| `quirks/configs/quirks-tracking.yaml` | MaskFormer config: 8 queries, mask + classification + IoU + two regression heads, no hit filter |
| `templates/`, `bootstrap.sh` | slurm scripts get generated into `slurm/` with your paths |
| `$SCRATCH` (ourdisk) | container, pixi env/caches, data, logs — **never in $HOME** (quota) |

## Notes

- The container is only CUDA+glibc+pixi; the environment builds from
  `hepattn/pixi.lock` and installs hepattn editable from the submodule.
- Everything heavier than code lives on ourdisk; `$HOME` quota is tight.
- Dumps for m = 100 (all 10 Lambda points) are at
  `/ourdisk/hpc/ouhep/jburzyns/dont_archive/Quirks/chain/gnndump/`.
