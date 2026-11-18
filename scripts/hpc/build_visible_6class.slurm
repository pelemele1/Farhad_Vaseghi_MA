#!/bin/bash -l
#
# Session 31: build the 6-class synthetic dataset (the capture protocol's
# contamination types) in the planned amounts -- 21,115 images from the 1,000
# MIO-TCD photos: per photo one clean variant and one per (type, severity),
# plus the two-type variants, 10% of the dataset
# (see src/soiling/dataset_builder.py::plan_variant_kinds and the README's
# "Planned dataset"). Replaces the 3-class dataset in data/processed/visible.
#
# Steps: (1) the test suite, one file per core (scripts/hpc/run_tests_parallel.sh)
# -- if any test fails the job stops here and the old dataset is untouched;
# (2) the build; (3) the label check of every image; (4) two check sheets.
#
# The work is pure CPU (the generators are OpenCV/NumPy), about 37 core-hours
# for the build, so the job is sized by its cores, not by the GPU; TinyGPU still
# wants --gres on every job. Submit from the repository root on the cluster:
#
#   mkdir -p logs/jobs        # Slurm writes the log there and does not create it
#   sbatch.tinygpu --export=NONE,OUT=data/processed/visible_pilot,MAX_SOURCES=20 \
#       --time=1:00:00 scripts/hpc/build_visible_6class.slurm     # a small test first
#   sbatch.tinygpu scripts/hpc/build_visible_6class.slurm          # then the full build
#
# Settings reach the job only through --export (the job runs with
# --export=NONE): OUT, MAX_SOURCES, WORKERS, SEED, SOURCE, CONDA_ENV, and
# SKIP_TESTS=1 to go straight to the build (e.g. after the tests passed once).
#
#SBATCH --job-name=build_visible_6class
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx3080
#SBATCH --time=24:00:00
#SBATCH --output=logs/jobs/%x_%j.out
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module load python
conda activate "${CONDA_ENV:-stage_a}"

SOURCE=${SOURCE:-data/raw/mio_tcd/images}
OUT=${OUT:-data/processed/visible}
WORKERS=${WORKERS:-$(nproc)}   # the cores this job was given
SEED=${SEED:-0}

set -e   # after conda activate (which is not set -e safe): any failing step fails the job
mkdir -p logs/jobs "$(dirname "$OUT")"

if [ "${SKIP_TESTS:-0}" != 1 ]; then
    bash scripts/hpc/run_tests_parallel.sh "$WORKERS" "logs/jobs/tests_${SLURM_JOB_ID:-local}"
fi

# A rebuild must not be mixed with the files of the old 3-class dataset: the
# image and label file names repeat per source photo, so leftovers would be
# silently kept and read back as part of the new dataset.
if [ -d "$OUT" ]; then
    OLD="${OUT}_old_$(date +%Y%m%d_%H%M%S)"
    echo "Moving the existing $OUT aside to $OLD (delete it once the new dataset is checked)"
    mv "$OUT" "$OLD"
fi

echo "Building $OUT from $SOURCE with $WORKERS workers on $(hostname)"
python3 -u scripts/build_visible_dataset.py \
    --source "$SOURCE" \
    --out "$OUT" \
    --seed "$SEED" \
    --workers "$WORKERS" \
    ${MAX_SOURCES:+--max-sources $MAX_SOURCES}

# Every label checked against its image, label map, tiles and the planned
# amounts; the job fails (and says why) if anything is off. A partial test
# build (MAX_SOURCES) skips the amounts, which only hold for all photos.
python3 -u scripts/check_visible_dataset.py --data "$OUT" ${MAX_SOURCES:+--skip-amounts}

# A sheet of one photo with each type and its ground truth, to look at.
python3 -u scripts/render_label_check_sheet.py --data "$OUT" --out "$OUT/label_check_medium.jpg" --severity medium
python3 -u scripts/render_label_check_sheet.py --data "$OUT" --out "$OUT/label_check_light.jpg" --severity low
