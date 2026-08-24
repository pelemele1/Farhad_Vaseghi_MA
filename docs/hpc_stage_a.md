# Running Stage A training on NHR@FAU TinyGPU

Operational how-to for training the Stage A distortion head on your own FAU HPC
allocation. Everything here is verified against the live docs at
[doc.nhr.fau.de/clusters/tinygpu](https://doc.nhr.fau.de/clusters/tinygpu/) and
[doc.nhr.fau.de/environment/python-env](https://doc.nhr.fau.de/environment/python-env/) —
not guessed. `portal.hpc.fau.de` is your account/project management portal (allocations,
project IDs); it's login-gated and this doc doesn't cover it — that's between you and your
supervisor's project.

Per the project's standing rule, Claude prepares this code and these job scripts but never
submits or runs a real training job — everything below is something **you** run.

## 1. Get onto the cluster

```bash
ssh <your-hpc-username>@tinyx.nhr.fau.de
```

(Configure your SSH key first if you haven't — see `doc.nhr.fau.de` under "Access to
NHR@FAU systems" if `ssh` prompts for a password every time.)

## 2. Put the code and data on `$WORK`

`$HOME` is small (100 GB, backed up) and meant for source/important results only. `$WORK`
has no backup but a much larger quota (1000 GB on Tier3) and is the right place for a full
working clone, since the code itself is already safely versioned in git — there's nothing
here that needs a second backup.

```bash
cd $WORK
git clone <your repo URL> Farhad_Vaseghi_MA
cd Farhad_Vaseghi_MA
git checkout feature/stage-a-mio-tcd
```

`data/`, `weights/`, and `checkpoints/` are gitignored (regenerable, not part of the repo),
so they won't come along with the clone. Two ways to get the dataset there:

- **Copy what you already built locally** (fastest, reuses the exact 4000-image dataset
  from your Windows machine):
  ```bash
  # from your local machine
  scp -r data/processed/stage_a <user>@tinyx.nhr.fau.de:$WORK/Farhad_Vaseghi_MA/data/processed/
  scp weights/yolo11m.pt <user>@tinyx.nhr.fau.de:$WORK/Farhad_Vaseghi_MA/weights/
  ```
- **Or regenerate on the cluster** — upload `MIO-TCD-Localization.tar` and rerun
  `scripts/sample_mio_tcd.py` + `scripts/build_stage_a_dataset.py` there. `yolo11m.pt` will
  auto-download the first time `FrozenYOLOBackbone` runs, but only if the node has internet
  access (see the proxy note in step 3) — copying the already-downloaded file over is
  simpler and avoids depending on that.

## 3. One-time environment setup

Do this from an **interactive** GPU job, not the plain frontend, so GPU support is detected
correctly when packages install:

```bash
salloc.tinygpu --gres=gpu:1 --time=01:00:00
cd $WORK/Farhad_Vaseghi_MA
bash scripts/hpc/setup_env.sh
```

This creates a conda env named `stage_a` (Python 3.11) with everything in
`requirements.txt`, including a CUDA-enabled `torch` build (unlike the CPU-only one used for
local development). It's safe to rerun if it fails partway through. If a package fails to
download, some compute nodes need a proxy — the script already sets
`http_proxy`/`https_proxy=http://proxy.nhr.fau.de:80` before installing.

## 4. Submit the training job

```bash
cd $WORK/Farhad_Vaseghi_MA
sbatch.tinygpu scripts/hpc/train_stage_a.slurm
```

`scripts/hpc/train_stage_a.slurm` requests 1 RTX3080 GPU for 6 hours (well under the 24h
cap) and runs `scripts/train_stage_a.py --device cuda` with `--epochs 20 --batch-size 16`
(lowered from 32 -- the RTX3080's 10GB VRAM is much smaller than an A100's 40GB).
**Tune these** once you've seen how fast an epoch actually runs on your data — the values
in the script are a starting point, not a validated setting. (`a100` was tried first, but
this account's association has no GPU quota there -- the job sat pending forever with
reason `AssocGrpGRES`.)

If your account has access to a different GPU type, swap the partition/GPU type -- `a100`
or `v100` work the same way (edit `--gres=gpu:a100:1` / `-p a100`, etc.). Check what your
account can actually submit to with:

```bash
sinfo.tinygpu
```

## 5. Monitor and retrieve results

```bash
squeue.tinygpu -u $USER          # job status
tail -f stage_a_<jobid>.out      # live training log (written to the repo root)
```

The trained head lands at `checkpoints/stage_a/stage_a_head.pt` (a dict with
`head_state_dict` and `class_names`) once training finishes. Copy it back to your local
machine or `$HPCVAULT` (backed up, good for keeping results long-term) with `scp` or `rsync`.
