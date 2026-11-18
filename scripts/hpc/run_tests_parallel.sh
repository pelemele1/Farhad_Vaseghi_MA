#!/bin/bash
# Runs the test suite with one pytest process per test file, N files at a time,
# so the many cores of an HPC node shorten it from hours (one core) to about the
# time of the slowest file. Each file's output goes to its own log in LOG_DIR; a
# summary is printed at the end and the exit status is 1 if any file failed.
#
#   bash scripts/hpc/run_tests_parallel.sh [workers] [log_dir]
#
# tests/test_capture_app.py is left out: it needs PySide6, the Vimba X camera
# software and a screen, which a compute node has none of. It runs on the
# capture computer (Windows) instead.
set -u
WORKERS=${1:-$(nproc)}
LOG_DIR=${2:-logs/jobs/tests_${SLURM_JOB_ID:-local}}
mkdir -p "$LOG_DIR"
PYTHON=${PYTHON:-python3}   # PYTHON=python on Windows, where python3 is only a Store stub
export LOG_DIR PYTHON KMP_DUPLICATE_LIB_OK=TRUE MPLBACKEND=Agg

run_one() {
    name=$(basename "$1" .py)
    if "$PYTHON" -m pytest -q -p no:cacheprovider -p no:warnings -rfE "$1" > "$LOG_DIR/$name.log" 2>&1; then
        echo "PASS  $name  $(tail -n 1 "$LOG_DIR/$name.log")"
    else
        echo "FAIL  $name  $(tail -n 1 "$LOG_DIR/$name.log")"
    fi
}
export -f run_one

echo "Running the tests, $WORKERS files at a time; logs in $LOG_DIR"
start=$(date +%s)
# TEST_FILES (space-separated) runs only those files, e.g. for a quick check.
FILES=${TEST_FILES:-$(ls tests/test_*.py | grep -v "tests/test_capture_app.py")}
printf '%s\n' $FILES \
    | xargs -P "$WORKERS" -I{} bash -c 'run_one "$@"' _ {} | tee "$LOG_DIR/summary.txt"

echo "Tests took $(( ($(date +%s) - start) / 60 )) min"
if grep -q "^FAIL" "$LOG_DIR/summary.txt"; then
    echo "SOME TESTS FAILED -- see the logs above; the failures in short:"
    for log in $(grep "^FAIL" "$LOG_DIR/summary.txt" | awk '{print $2}'); do
        grep -E "^(FAILED|ERROR)" "$LOG_DIR/$log.log" | sed "s/^/    /"
    done
    exit 1
fi
echo "ALL TESTS PASSED"
