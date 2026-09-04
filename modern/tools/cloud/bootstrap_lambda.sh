#!/usr/bin/env bash
# Bootstrap a Lambda Cloud 8x H100 (SXM) box for the PIC-MCC campaign. Idempotent: re-running
# fast-forwards the checkout, re-syncs the venv and re-runs the smoke checks; nothing is duplicated.
#
# Usage (as user ubuntu, once the IP is known; see PLAN.md for the full operator sequence):
#
#   export WORK=/home/ubuntu/<persistent-fs>/cft          # persistent filesystem mount
#   export REPO_URL=https://github.com/terrorproforma/open-cft.git
#   export GIT_TOKEN=...            # https: fine-grained token (read, or read/write for pushing results)
#   # or:  export DEPLOY_KEY=~/.ssh/open-cft-deploy   and  REPO_URL=git@github.com:terrorproforma/open-cft.git
#   bash bootstrap_lambda.sh
#
# Environment knobs (all optional):
#   WORK            work root on the persistent filesystem          (default $HOME/work)
#   REPO_URL        clone URL (https or ssh)                         (default the open-cft GitHub https URL)
#   REPO_REF        branch to check out                              (default feat/sota-foundation)
#   GIT_TOKEN       token for https remotes; handed to git through a credential helper that reads the
#                   variable at call time, so the secret is never written into .git/config or this log
#   DEPLOY_KEY      private key for ssh remotes (GIT_SSH_COMMAND with IdentitiesOnly)
#   EXPECTED_GPUS   number of GPUs nvidia-smi must report            (default 8)
#   PYTHON_VERSION  uv-managed interpreter                           (default 3.12; the local anchor is 3.12.10)
#   LFS_FULL        1 = pull every LFS object (914 MB), 0 = only the PIC field inputs (default 1)
#   SKIP_APT        1 = do not touch apt (image already has the packages)
#   SKIP_TESTS      1 = skip the pytest smoke
#   TEST_GPU        GPU index for the pytest smoke                   (default 0)
#   WARP_WHEEL_URL  optional explicit wheel (e.g. the +cu13 GitHub wheel); default = PyPI warp-lang==1.14.0
#
# Outputs: $WORK/uni-project (checkout), $WORK/uni-project/.venv-pic (venv), $WORK/provision.log.
set -euo pipefail

WORK="${WORK:-$HOME/work}"
REPO_URL="${REPO_URL:-https://github.com/terrorproforma/open-cft.git}"
REPO_REF="${REPO_REF:-feat/sota-foundation}"
EXPECTED_GPUS="${EXPECTED_GPUS:-8}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
LFS_FULL="${LFS_FULL:-1}"
SKIP_APT="${SKIP_APT:-0}"
SKIP_TESTS="${SKIP_TESTS:-0}"
TEST_GPU="${TEST_GPU:-0}"
WARP_WHEEL_URL="${WARP_WHEEL_URL:-}"

REPO_DIR="$WORK/uni-project"
VENV_DIR="$REPO_DIR/.venv-pic"
REQUIREMENTS_REL="modern/tools/cloud/requirements-pic.txt"
LOG="$WORK/provision.log"
# The PIC field inputs: the P2 authority checkpoints (.npz arrays are LFS objects, the .json descriptors are
# plain) and the design mini-sweep's four material-aware fields (json + npz, both LFS).
PIC_LFS_INCLUDE="modern/examples/fem_reference/artifacts/third-level/divergent-exit-stack/checkpoints/*,modern/experiments/pic2d_design_mini_sweep_v1/fields/**"

mkdir -p "$WORK"
touch "$LOG"

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG"; }
die() { log "FATAL: $*"; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

log "== bootstrap_lambda.sh start (host $(hostname), user $(id -un), WORK=$WORK)"

# ----------------------------------------------------------------------------- 1. GPUs / driver
have nvidia-smi || die "nvidia-smi not found: this is not a Lambda Stack image (or the driver is not loaded)"
GPU_COUNT="$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -n1 | tr -d '[:space:]')"
DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -n1 | tr -d '[:space:]')"
# nvidia-smi prints the driver's CUDA version in the banner (there is no --query field for it)
CUDA_DRIVER_VERSION="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -n1)"
log "nvidia-smi: ${GPU_COUNT} GPU(s), driver ${DRIVER}, driver CUDA ${CUDA_DRIVER_VERSION:-unknown}"
nvidia-smi --query-gpu=index,name,uuid,memory.total,pci.bus_id --format=csv,noheader | tee -a "$LOG"
[ "$GPU_COUNT" = "$EXPECTED_GPUS" ] || die "expected ${EXPECTED_GPUS} GPUs, nvidia-smi reports ${GPU_COUNT}"
# Warp 1.14.0 PyPI wheels are CUDA 12.9 builds -> driver >= 525 required (the +cu13 wheel would need >= 580)
DRIVER_MAJOR="${DRIVER%%.*}"
[ "$DRIVER_MAJOR" -ge 525 ] || die "driver ${DRIVER} is older than 525: the CUDA 12.x Warp wheel cannot load"

# ----------------------------------------------------------------------------- 2. system packages
if [ "$SKIP_APT" != "1" ]; then
    if have apt-get; then
        log "apt: git git-lfs build-essential cmake tmux jq curl ca-certificates"
        export DEBIAN_FRONTEND=noninteractive
        sudo -n apt-get update -qq
        # build-essential/cmake are only needed for the OPTIONAL pybind11 kernel (cft_revival._native);
        # the pic2d tests skip it when absent. They are cheap, so install them for completeness.
        sudo -n apt-get install -y -qq git git-lfs build-essential cmake tmux jq curl ca-certificates >/dev/null
    else
        log "apt-get not available; skipping system packages"
    fi
fi
git lfs install --skip-repo >/dev/null
log "git $(git --version | awk '{print $3}'), git-lfs $(git lfs version | awk '{print $1}' | cut -d/ -f2)"

# ----------------------------------------------------------------------------- 3. uv (+ managed python)
if ! have uv; then
    log "installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
have uv || die "uv is not on PATH after installation"
log "uv $(uv --version | awk '{print $2}')"
uv python install "$PYTHON_VERSION" >/dev/null 2>&1 || true   # no-op when already present

# ----------------------------------------------------------------------------- 4. repository
# Credentials: never written to disk. https -> credential helper that echoes $GIT_TOKEN at call time;
# ssh -> GIT_SSH_COMMAND with the deploy key. git-lfs reuses both.
GIT_CRED_ARGS=()
if [ -n "${GIT_TOKEN:-}" ]; then
    GIT_CRED_ARGS=(-c "credential.helper=!f() { echo username=x-access-token; echo \"password=\${GIT_TOKEN}\"; }; f")
    log "auth: https token from GIT_TOKEN (credential helper, not persisted)"
elif [ -n "${DEPLOY_KEY:-}" ]; then
    export GIT_SSH_COMMAND="ssh -i ${DEPLOY_KEY} -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"
    log "auth: ssh deploy key ${DEPLOY_KEY}"
else
    log "auth: none (public https or agent-forwarded ssh)"
fi
g() { git "${GIT_CRED_ARGS[@]}" "$@"; }

if [ ! -d "$REPO_DIR/.git" ]; then
    log "cloning ${REPO_URL} (${REPO_REF}) -> ${REPO_DIR} (LFS smudge deferred)"
    GIT_LFS_SKIP_SMUDGE=1 g clone --branch "$REPO_REF" "$REPO_URL" "$REPO_DIR"
else
    log "fetching ${REPO_DIR}"
    (cd "$REPO_DIR" && GIT_LFS_SKIP_SMUDGE=1 g fetch --prune origin && g checkout -q "$REPO_REF" && GIT_LFS_SKIP_SMUDGE=1 g pull --ff-only origin "$REPO_REF")
fi
cd "$REPO_DIR"
GIT_HEAD="$(git rev-parse HEAD)"
log "checkout ${REPO_REF} at ${GIT_HEAD}"
log "git lfs pull (PIC field inputs: ${PIC_LFS_INCLUDE})"
g lfs pull --include="$PIC_LFS_INCLUDE"
if [ "$LFS_FULL" = "1" ]; then
    log "git lfs pull (all objects, ~914 MB)"
    g lfs pull
fi
# fail closed if the PIC field authority is still a pointer file
P2_NPZ="modern/examples/fem_reference/artifacts/third-level/divergent-exit-stack/checkpoints/divergent-exit-stack.level-1.json.arrays.npz"
[ "$(stat -c %s "$P2_NPZ")" -gt 1000000 ] || die "${P2_NPZ} is still an LFS pointer (LFS pull failed?)"

# ----------------------------------------------------------------------------- 5. venv from the pins
[ -f "$REQUIREMENTS_REL" ] || die "${REQUIREMENTS_REL} missing at ${GIT_HEAD}; the pins live in the repo"
if [ ! -x "$VENV_DIR/bin/python" ]; then
    log "creating ${VENV_DIR} (python ${PYTHON_VERSION})"
    uv venv --python "$PYTHON_VERSION" "$VENV_DIR" >/dev/null
fi
PY="$VENV_DIR/bin/python"
log "installing pins from ${REQUIREMENTS_REL} (warp-lang 1.14.0 = PyPI CUDA 12.9 build)"
uv pip install --python "$PY" -q -r "$REQUIREMENTS_REL"
if [ -n "$WARP_WHEEL_URL" ]; then
    log "overriding warp-lang with ${WARP_WHEEL_URL}"
    uv pip install --python "$PY" -q --reinstall --no-deps "$WARP_WHEEL_URL"
fi
PY_VERSION="$("$PY" -c 'import sys; print(sys.version.split()[0])')"
NUMPY_VERSION="$("$PY" -c 'import numpy; print(numpy.__version__)')"
WARP_VERSION="$("$PY" -c 'import warp; print(warp.config.version)')"
log "python ${PY_VERSION}, numpy ${NUMPY_VERSION}, warp-lang ${WARP_VERSION}"

# CUDA indices == nvidia-smi indices for every process spawned from here on (the scheduler sets it too)
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# ----------------------------------------------------------------------------- 6. smoke: Warp sees the GPUs
log "smoke: warp.init() + CUDA device enumeration"
WARP_SMOKE="$("$PY" - <<'PYEOF' | tail -n 1
import json
import warp as wp
if hasattr(wp, "LOG_WARNING"):  # keep the init banner out of the JSON line (config.quiet is deprecated in 1.14)
    wp.config.log_level = wp.LOG_WARNING
else:
    wp.config.quiet = True
wp.init()
try:                            # the runtime object is private API; report what it knows, never fail on it
    from warp._src import context as _context
    rt = _context.runtime
    toolkit, driver = rt.toolkit_version, rt.driver_version
except Exception:               # noqa: BLE001
    toolkit = driver = None
devices = wp.get_cuda_devices()
print(json.dumps({
    "warp": wp.config.version,
    "toolkit_built_with": toolkit,
    "driver_cuda": driver,
    "cuda_devices": [{"alias": str(d), "name": d.name, "arch": d.arch,
                      "uuid": getattr(d, "uuid", None), "pci_bus_id": getattr(d, "pci_bus_id", None),
                      "total_memory_gib": round(d.total_memory / 2**30, 1)} for d in devices],
}))
PYEOF
)"
echo "$WARP_SMOKE" | tee -a "$LOG"
WARP_GPU_COUNT="$(echo "$WARP_SMOKE" | "$PY" -c 'import json,sys; print(len(json.load(sys.stdin)["cuda_devices"]))')"
[ "$WARP_GPU_COUNT" = "$EXPECTED_GPUS" ] || die "Warp enumerates ${WARP_GPU_COUNT} CUDA devices, expected ${EXPECTED_GPUS}"

# ----------------------------------------------------------------------------- 7. smoke: pic2d tests on one GPU
TEST_SECONDS="skipped"
TEST_RESULT="skipped"
if [ "$SKIP_TESTS" != "1" ]; then
    log "pytest modern/tests/pic2d -x -q on GPU ${TEST_GPU} (also warms the Warp kernel cache for sm_90)"
    START=$SECONDS
    set +e
    (cd modern && CUDA_VISIBLE_DEVICES="$TEST_GPU" PYTHONPATH="src:." OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
        "$PY" -m pytest tests/pic2d -x -q >"$WORK/pytest-pic2d.log" 2>&1)
    TEST_STATUS=$?
    set -e
    TEST_SECONDS=$((SECONDS - START))
    tail -n 5 "$WORK/pytest-pic2d.log" | tee -a "$LOG"
    if [ "$TEST_STATUS" -eq 0 ]; then TEST_RESULT="passed"; else TEST_RESULT="FAILED (exit ${TEST_STATUS})"; fi
    log "pytest tests/pic2d: ${TEST_RESULT} in ${TEST_SECONDS} s"
fi

# ----------------------------------------------------------------------------- 8. provenance record
{
    echo "---- provision record $(date -u +%Y-%m-%dT%H:%M:%SZ) ----"
    echo "host:            $(hostname)  ($(nproc) vCPU, $(awk '/MemTotal/ {printf "%.0f GiB", $2/1048576}' /proc/meminfo) RAM)"
    echo "os:              $(. /etc/os-release && echo "$PRETTY_NAME") $(uname -r)"
    echo "repo:            ${REPO_URL} ${REPO_REF}"
    echo "git head:        ${GIT_HEAD}"
    echo "git head subject: $(git log -1 --format=%s)"
    echo "nvidia driver:   ${DRIVER} (driver CUDA ${CUDA_DRIVER_VERSION:-unknown}); ${GPU_COUNT} GPUs"
    echo "cuda toolkit:    $(command -v nvcc >/dev/null 2>&1 && nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p' || echo 'nvcc not on PATH (not needed: Warp embeds NVRTC)')"
    echo "warp-lang:       ${WARP_VERSION} (smoke: ${WARP_SMOKE})"
    echo "python:          ${PY_VERSION} at ${PY}"
    echo "numpy:           ${NUMPY_VERSION}"
    echo "uv:              $(uv --version | awk '{print $2}')"
    echo "pins:            ${REQUIREMENTS_REL} sha256 $(sha256sum "$REQUIREMENTS_REL" | awk '{print $1}')"
    echo "pytest pic2d:    ${TEST_RESULT} (${TEST_SECONDS} s, GPU ${TEST_GPU})"
    echo "pip freeze:"
    uv pip freeze --python "$PY" | sed 's/^/                 /'
} | tee -a "$LOG"

log "== bootstrap done. Next: PLAN.md step 3 (bench.sh), then schedule.py plan/launch."
