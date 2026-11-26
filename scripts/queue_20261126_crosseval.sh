#!/usr/bin/env bash
## cross-region evaluation of 2026-11-26 (no training): the u+v models of u_v/20261120_regions scored on
## test 2026 of A03 and A05, interior only (--edge_pixels 10), folds val2024 + val2025 + ensemble:
##   a04_seed<s>      trained on A04 only          -> how well it transfers to a region it never saw
##   regions_seed<s>  trained on A03+A04+A05 (2023, 2024, 2025) -> reference: the region seen in training
##   GPU0 scores on A03, GPU1 on A05; GPU_A03 / GPU_A05 pick other GPUs, e.g. both on GPU 1 (they can share it:
##   evaluation is batch 1 without gradients, roughly 2 GB on A03 and 3 GB on A05; check nvidia-smi for free memory):
##   GPU_A03=1 GPU_A05=1 bash scripts/queue_20261126_crosseval.sh
## results: <run>/u_v/<fold>/eval_test_trim10_processed_A0x (ensemble: <run>/u_v/ensemble/...); runs already scored
## are skipped, so after a crash just start it again. Table:
##   python3 scripts/summarize_runs.py <O>/u_v/20261120_regions --eval eval_test_trim10_processed_A03
##
## usage (from anywhere; uses this worktree's code and the repo venv):
##   bash scripts/queue_20261126_crosseval.sh          # start both queues in detached screens raid_xeval_gpu0 / _gpu1
##   bash scripts/queue_20261126_crosseval.sh gpu1     # or run one queue in the foreground
## logs: <O>/queue_20261126_crosseval_gpu<N>.log
set -euo pipefail

here="$(cd "$(dirname "$0")/.." && pwd)"
VENV=${VENV:-/gpfs/gpfs0/osea/manuel/RAID/.venv}
O=${O:-/data/prosjekt/15915_RAID/train/step1B}
D=${D:-/data/prosjekt/15915_RAID/data}
RUNS="$O/u_v/20261120_regions"
TRIM=10

crosseval() {   # crosseval <region folder name, e.g. processed_A03>
    local region=$1
    for label in a04 regions; do
        if [ "$label" = a04 ]; then config=configs/step1b_unet_uv.yaml; else config=configs/step1b_unet_uv_regions.yaml; fi
        for s in 0 1 2; do
            run="$RUNS/${label}_seed${s}"
            echo "########## $label seed $s on $region"
            for fold in val2024 val2025 ensemble; do
                if [ -f "$run/u_v/$fold/eval_test_trim${TRIM}_${region}/v/daily_metrics.npy" ]; then
                    echo "$fold already scored on $region, skipping"
                    continue
                fi
                args=(--config "$config" --fold "$fold" --output_dir "$run" --processed_dir "$D/$region" --edge_pixels "$TRIM")
                [ "$fold" = ensemble ] || args+=(--split test)
                python3 scripts/evaluate.py "${args[@]}"
            done
        done
    done
}

gpu0() { CUDA_VISIBLE_DEVICES=${GPU_A03:-0} crosseval processed_A03; }
gpu1() { CUDA_VISIBLE_DEVICES=${GPU_A05:-1} crosseval processed_A05; }

case "${1:-}" in
    gpu0|gpu1)
        cd "$here"
        source "$VENV/bin/activate"
        export PYTHONPATH="$here/src"          # this branch's raid package, not the one installed from dev
        echo "=== $(date '+%F %T') crosseval $1, code $(git -C "$here" rev-parse --short HEAD) in $here"
        "$1"
        echo "=== $(date '+%F %T') crosseval $1 done"
        ;;
    "")
        for q in gpu0 gpu1; do
            if screen -ls | grep -q "\.raid_xeval_$q[[:space:]]"; then
                echo "screen raid_xeval_$q already exists, not starting it again"
                continue
            fi
            screen -dmS "raid_xeval_$q" bash -c "bash '$here/scripts/queue_20261126_crosseval.sh' $q 2>&1 | tee -a '$O/queue_20261126_crosseval_$q.log'"
            echo "started screen raid_xeval_$q (log $O/queue_20261126_crosseval_$q.log)"
        done
        ;;
    *)
        echo "usage: $0 [gpu0|gpu1]" >&2
        exit 1
        ;;
esac
