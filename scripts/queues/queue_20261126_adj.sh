#!/usr/bin/env bash
## run queue of 2026-11-26: u+v on A04 trained on target_adj (configs/step1b_unet_uv_adj.yaml, data/processed/160m_adj:
## the 160m target minus the day's large-scale mismatch to the 800m input), seed_study.sh (20 epochs, folds
## val2024 + val2025 + ensemble), then every model scored on BOTH targets, all cells and interior (trim 10):
##   adj_seed<s>  (new)                        on 160m_adj: eval_test[_trim10]   on 160m: eval_test[_trim10]_target160m
##   a04_seed<s>  (20261120_regions, existing) on 160m:     eval_test[_trim10]   on 160m_adj: ..._target160m_adj
## both queues first wait for the cross-region evaluation (queue_20261126_crosseval.sh) to finish.
##   GPU0: adj seeds 0 1;  GPU1: adj seed 2, then the a04 models on 160m_adj
## results: <O>/u_v/20261126_adj/adj_seed<s>; finished seeds are skipped, so after a crash just start it again.
##
## usage (from anywhere; uses this worktree's code and the repo venv):
##   bash scripts/queues/queue_20261126_adj.sh          # start both queues in detached screens raid_adj_gpu0 / _gpu1
##   bash scripts/queues/queue_20261126_adj.sh gpu1     # or run one queue in the foreground
## logs: <O>/queue_20261126_adj_gpu<N>.log
## tables:
##   python3 scripts/summarize_runs.py <O>/u_v/20261126_adj --eval eval_test_trim10_target160m     # adj models, real target
##   python3 scripts/summarize_runs.py <O>/u_v/20261120_regions --eval eval_test_trim10            # a04 models, real target
set -euo pipefail

here="$(cd "$(dirname "$0")/../.." && pwd)"
VENV=${VENV:-/gpfs/gpfs0/osea/manuel/RAID/.venv}
O=${O:-/data/prosjekt/15915_RAID/train/step1B}
OUT="$O/u_v/20261126_adj"
A04="$O/u_v/20261120_regions"
export FOLDS="val2024 val2025"

wait_crosseval() {
    while pgrep -f "queue_20261126_crosseval.sh" > /dev/null; do
        echo "$(date '+%F %T') cross-region evaluation still running, waiting"
        sleep 300
    done
}

score_on() {   # score_on <config> <run> <target dir>: test scores against another target, all cells + interior
    local config=$1 run=$2 tdir=$3
    if [ -f "$run/u_v/ensemble/eval_test_trim10_target${tdir}/v/daily_metrics.npy" ]; then
        echo "$run already scored on $tdir, skipping"
        return
    fi
    for fold in val2024 val2025 ensemble; do
        for trim in 0 10; do
            args=(--config "$config" --fold "$fold" --output_dir "$run" --target_dir "$tdir" --edge_pixels "$trim")
            [ "$fold" = ensemble ] || args+=(--split test)
            python3 scripts/evaluate.py "${args[@]}"
        done
    done
}

adj() {   # adj <gpu> <seeds>
    CUDA_VISIBLE_DEVICES=$1 LABEL=adj SEEDS="$2" bash scripts/seed_study.sh configs/step1b_unet_uv_adj.yaml "$OUT"
    for s in $2; do
        CUDA_VISIBLE_DEVICES=$1 score_on configs/step1b_unet_uv_adj.yaml "$OUT/adj_seed$s" 160m
    done
}

gpu0() { wait_crosseval; adj 0 "0 1"; }
gpu1() {
    wait_crosseval
    adj 1 "2"
    for s in 0 1 2; do
        CUDA_VISIBLE_DEVICES=1 score_on configs/step1b_unet_uv.yaml "$A04/a04_seed$s" 160m_adj
    done
}

case "${1:-}" in
    gpu0|gpu1)
        cd "$here"
        source "$VENV/bin/activate"
        export PYTHONPATH="$here/src"          # this branch's raid package, not the one installed from dev
        echo "=== $(date '+%F %T') queue adj $1, code $(git -C "$here" rev-parse --short HEAD) in $here"
        "$1"
        echo "=== $(date '+%F %T') queue adj $1 done"
        ;;
    "")
        for q in gpu0 gpu1; do
            if screen -ls | grep -q "\.raid_adj_$q[[:space:]]"; then
                echo "screen raid_adj_$q already exists, not starting it again"
                continue
            fi
            screen -dmS "raid_adj_$q" bash -c "bash '$here/scripts/queues/queue_20261126_adj.sh' $q 2>&1 | tee -a '$O/queue_20261126_adj_$q.log'"
            echo "started screen raid_adj_$q (log $O/queue_20261126_adj_$q.log)"
        done
        ;;
    *)
        echo "usage: $0 [gpu0|gpu1]" >&2
        exit 1
        ;;
esac
