#!/usr/bin/env bash
# Run the per-GPU concurrency benchmark on one H100 plus the CPU-only host factorisation benchmark,
# and write JSON + markdown under $BENCH_OUT. Runs after bootstrap_lambda.sh (same WORK layout).
#
#   BENCH_GPU=0 bash modern/tools/cloud/bench.sh            # ~30-45 min on GPU 0, CPUs shared
#
# Knobs: WORK (default $HOME/work), BENCH_GPU (default 0), BENCH_OUT (default $WORK/bench),
#        CONCURRENCY (default "1 2 4"), CONFIGS (default "channel-50um channel-33um plume-v2.0-50um"),
#        STEPS (2000), WARMUP (400), LOADS ("seed production"), FACT_CONCURRENCY ("1 4 8"),
#        FACT_CONFIGS ("channel-50um channel-33um plume-v2.0-50um plume-v2.1-50um"), SKIP_FACTORISE=1.
#
# The `seed` load times the launch-bound floor at the protocol seed (~0.26 M particles); the
# `production` load re-seeds every configuration to its recorded plateau particle count so the
# RTX 5090 anchors (1.98 ms/step channel-only, 7.0-7.15 ms/step plume v2.0) are like-for-like.
set -euo pipefail

WORK="${WORK:-$HOME/work}"
REPO_DIR="${REPO_DIR:-$WORK/uni-project}"
PY="${PY:-$REPO_DIR/.venv-pic/bin/python}"
BENCH_GPU="${BENCH_GPU:-0}"
BENCH_OUT="${BENCH_OUT:-$WORK/bench}"
CONCURRENCY="${CONCURRENCY:-1 2 4}"
CONFIGS="${CONFIGS:-channel-50um channel-33um plume-v2.0-50um}"
STEPS="${STEPS:-2000}"
WARMUP="${WARMUP:-400}"
LOADS="${LOADS:-seed production}"
FACT_CONCURRENCY="${FACT_CONCURRENCY:-1 4 8}"
FACT_CONFIGS="${FACT_CONFIGS:-channel-50um channel-33um plume-v2.0-50um plume-v2.1-50um}"
SKIP_FACTORISE="${SKIP_FACTORISE:-0}"

[ -x "$PY" ] || { echo "venv python not found at $PY (run bootstrap_lambda.sh first)" >&2; exit 1; }
mkdir -p "$BENCH_OUT"
cd "$REPO_DIR/modern"
export PYTHONPATH="src:."
export CUDA_DEVICE_ORDER=PCI_BUS_ID
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
HOST="$(hostname)"

# Refuse to share the benchmark GPU with a running job: the numbers would be meaningless.
if nvidia-smi --query-compute-apps=pid --format=csv,noheader --id="$BENCH_GPU" | grep -q '[0-9]'; then
    echo "GPU $BENCH_GPU already has compute processes; pick another BENCH_GPU" >&2
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv --id="$BENCH_GPU" >&2
    exit 1
fi

# shellcheck disable=SC2086  # word splitting of the knob lists is intended
for LOAD in $LOADS; do
    OUT_JSON="$BENCH_OUT/gpu-concurrency-${HOST}-gpu${BENCH_GPU}-${LOAD}-${STAMP}.json"
    echo "== GPU concurrency, load=${LOAD}, GPU ${BENCH_GPU}, N in {${CONCURRENCY}}, configs {${CONFIGS}} -> ${OUT_JSON}"
    "$PY" -m tools.cloud.bench_gpu_concurrency run --gpu "$BENCH_GPU" --concurrency $CONCURRENCY \
        --configs $CONFIGS --steps "$STEPS" --warmup "$WARMUP" --load "$LOAD" \
        --out "$OUT_JSON" --markdown "${OUT_JSON%.json}.md" --scratch "$BENCH_OUT/scratch-${LOAD}-${STAMP}" --keep-scratch
done

if [ "$SKIP_FACTORISE" != "1" ]; then
    OUT_JSON="$BENCH_OUT/host-factorisation-${HOST}-${STAMP}.json"
    echo "== host factorisation (CPU only), N in {${FACT_CONCURRENCY}}, configs {${FACT_CONFIGS}} -> ${OUT_JSON}"
    # shellcheck disable=SC2086
    "$PY" -m tools.cloud.bench_gpu_concurrency factorise --concurrency $FACT_CONCURRENCY --configs $FACT_CONFIGS \
        --out "$OUT_JSON" --markdown "${OUT_JSON%.json}.md"
fi

echo "== done; reports in $BENCH_OUT:"
ls -1 "$BENCH_OUT"/*.md
