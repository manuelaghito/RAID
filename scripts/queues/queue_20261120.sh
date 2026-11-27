#!/usr/bin/env bash
## run queue of 2026-11-20, all with seed_study.sh (20 epochs, folds val2024 + val2025 + ensemble, no full model):
##   GPU0: u+v trained on A04+A03+A05 (regions), seeds 0 1
##   GPU1: waits for the edge study (edge_study.sh) to finish, then
##         u+v on A04 only (a04, seeds 0 1 2; seed 0 = the linked 20261119 run, only re-evaluated)
##         salt on A04 with the cubic 800m input (cubic, seeds 0 1 2)
##         u+v regions seed 2
## results: <O>/u_v/20261120_regions/{a04,regions}_seed<s>, <O>/salt/20261120_cubic/cubic_seed<s>
## finished seeds are skipped, so after a crash just start the same thing again.
##
## usage (from anywhere; uses this worktree's code and the repo venv):
##   bash scripts/queues/queue_20261120.sh          # start both queues in detached screen sessions raid_gpu0 / raid_gpu1
##   screen -r raid_gpu0                     # watch one (detach again with ctrl-a d)
##   bash scripts/queues/queue_20261120.sh gpu1     # or run one queue in the foreground (what the screens run)
## logs: <O>/queue_20261120_gpu<N>.log
set -euo pipefail

here="$(cd "$(dirname "$0")/../.." && pwd)"
VENV=${VENV:-/gpfs/gpfs0/osea/manuel/RAID/.venv}
O=${O:-/data/prosjekt/15915_RAID/train/step1B}

gpu0() {
    LABEL=regions SEEDS="0 1" CUDA_VISIBLE_DEVICES=0 \
        bash scripts/seed_study.sh configs/step1b_unet_uv_regions.yaml "$O/u_v/20261120_regions"
}

gpu1() {
    while pgrep -f "edge_study.sh" > /dev/null; do
        echo "$(date '+%F %T') edge study still running, waiting"
        sleep 600
    done
    LABEL=a04 SEEDS="0 1 2" CUDA_VISIBLE_DEVICES=1 \
        bash scripts/seed_study.sh configs/step1b_unet_uv.yaml "$O/u_v/20261120_regions"
    LABEL=cubic SEEDS="0 1 2" CUDA_VISIBLE_DEVICES=1 \
        bash scripts/seed_study.sh configs/step1b_unet_salt.yaml "$O/salt/20261120_cubic"
    LABEL=regions SEEDS="2" CUDA_VISIBLE_DEVICES=1 \
        bash scripts/seed_study.sh configs/step1b_unet_uv_regions.yaml "$O/u_v/20261120_regions"
}

case "${1:-}" in
    gpu0|gpu1)
        cd "$here"
        source "$VENV/bin/activate"
        export PYTHONPATH="$here/src"          # this branch's raid package, not the one installed from dev
        echo "=== $(date '+%F %T') queue $1, code $(git -C "$here" rev-parse --short HEAD) in $here"
        "$1"
        echo "=== $(date '+%F %T') queue $1 done"
        ;;
    "")
        for q in gpu0 gpu1; do
            if screen -ls | grep -q "\.raid_$q[[:space:]]"; then
                echo "screen raid_$q already exists, not starting it again"
                continue
            fi
            screen -dmS "raid_$q" bash -c "bash '$here/scripts/queues/queue_20261120.sh' $q 2>&1 | tee -a '$O/queue_20261120_$q.log'"
            echo "started screen raid_$q (log $O/queue_20261120_$q.log)"
        done
        ;;
    *)
        echo "usage: $0 [gpu0|gpu1]" >&2
        exit 1
        ;;
esac
