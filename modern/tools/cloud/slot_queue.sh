#!/bin/bash
# slot_queue.sh - a CHAINED slot-waiter for the Lambda H100 box: launch the listed preregistered jobs ONE AT A TIME as
# scheduler slots free, in the given order, strictly AFTER a predecessor queue has finished (never two launchers racing for
# one slot).
#
#   bash slot_queue.sh <name> [--after <predecessor queue.log>] <job-id> [<job-id> ...]
#
# * the queue log is $WORK/<name>/queue.log; the predecessor is done when its log carries "== queue done" (the r1-queue
#   convention); while it is alive (tmux session <name-of-predecessor> or no "queue done" line) this queue only polls -
#   it NEVER calls the scheduler, so it cannot take a slot the predecessor is waiting for; a predecessor that ended for
#   another reason ("queue stopped" / "giving up") HOLDS this queue (a human decides; nothing is skipped)
# * launches with `schedule.py launch --only <id>` (refuses while the slots are busy; never --force; a job with a state
#   is never relaunched); retries every 10 min only on the "no free GPU slot" refusal; any other refusal stops the queue
# * reads jobs.yaml from the shared checkout at launch time: pull the jobs commit BEFORE the queue reaches a job
# * never signals any process (Xid-31 lesson); no `timeout` wrapper
set -u
export CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps}
export CUDA_MPS_LOG_DIRECTORY=${CUDA_MPS_LOG_DIRECTORY:-/tmp/nvidia-log}
WORK=${WORK:-/lambda/nfs/h100-files/cft}
NAME="$1"; shift
AFTER=""
if [ "${1:-}" = "--after" ]; then AFTER="$2"; shift 2; fi
if [ $# -lt 1 ]; then echo "usage: slot_queue.sh <name> [--after <predecessor queue.log>] <job-id> ..."; exit 64; fi
mkdir -p "$WORK/$NAME"
LOG=$WORK/$NAME/queue.log
cd "$WORK/uni-project/modern" || exit 1
export PYTHONPATH=src:.
PY=../.venv-pic/bin/python
echo "== queue $NAME start $(date -u +%FT%TZ) head $(git rev-parse --short HEAD) jobs: $*" | tee -a "$LOG"
if [ -n "$AFTER" ]; then
  echo "== waiting for the predecessor queue ($AFTER) to finish" | tee -a "$LOG"
  WAITED=0
  while true; do
    if [ -f "$AFTER" ] && grep -q "^== queue done" "$AFTER"; then
      echo "== predecessor done $(date -u +%FT%TZ); proceeding" | tee -a "$LOG"
      break
    fi
    if [ -f "$AFTER" ] && grep -qE "^== (giving up|.* REFUSED for another reason; queue stopped)" "$AFTER"; then
      if [ $((WAITED % 12)) -eq 0 ]; then echo "   [$(date -u +%H:%M)] predecessor ended WITHOUT 'queue done' - holding (a human decides); nothing launched" | tee -a "$LOG"; fi
    elif [ $((WAITED % 12)) -eq 0 ]; then
      echo "   [$(date -u +%H:%M)] predecessor still running ($WAITED polls)" | tee -a "$LOG"
    fi
    WAITED=$((WAITED+1))
    sleep 300
  done
fi
for JOB in "$@"; do
  echo "== waiting for a slot for $JOB $(date -u +%FT%TZ)" | tee -a "$LOG"
  TRIES=0
  while true; do
    OUT=$($PY tools/cloud/schedule.py launch --only "$JOB" 2>&1)
    RC=$?
    if [ $RC -eq 0 ]; then
      echo "$OUT" | tail -6 | tee -a "$LOG"
      echo "== launched $JOB $(date -u +%FT%TZ)" | tee -a "$LOG"
      sleep 180
      $PY tools/cloud/schedule.py status 2>&1 | grep -F -e "id " -e "$JOB" | tee -a "$LOG"
      break
    fi
    if echo "$OUT" | grep -q "no free GPU slot"; then
      TRIES=$((TRIES+1))
      if [ $((TRIES % 6)) -eq 1 ]; then echo "   [$(date -u +%H:%M)] slots busy ($TRIES); waiting" | tee -a "$LOG"; fi
      if [ $TRIES -gt 400 ]; then echo "== giving up on $JOB after $TRIES tries" | tee -a "$LOG"; exit 2; fi
      sleep 600
      continue
    fi
    echo "== $JOB REFUSED for another reason; queue stopped:" | tee -a "$LOG"
    echo "$OUT" | tail -8 | tee -a "$LOG"
    exit 3
  done
done
echo "== queue done $(date -u +%FT%TZ)" | tee -a "$LOG"
