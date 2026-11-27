#!/usr/bin/env bash
## edge-trimming study: train and evaluate every (edge_pixels, seed) pair with all other settings identical.
## for each pair: train_folds.sh (val2024, val2025, full), then evaluate_folds.sh twice:
## on all sea cells (eval_*) and on the interior only (eval_*_trim<TRIM>).
## runs go to <OUT>/edge<e>_seed<s>/<var>/<fold>; a pair whose full/best.pt already exists is not
## retrained, so after a crash just start the script again.
##
## usage (activate the venv first: source .venv/bin/activate):
## CUDA_VISIBLE_DEVICES=1 bash scripts/edge_study.sh configs/step1b_unet.yaml /data/prosjekt/15915_RAID/train/step1B/temp/edge_study
## settings via environment variables, e.g. a cheaper first pass:
## EPOCHS=12 SEEDS="0 1" CUDA_VISIBLE_DEVICES=1 bash scripts/edge_study.sh configs/step1b_unet.yaml <OUT>
## then the results table:
## python3 scripts/summarize_runs.py <OUT>
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "usage: $0 <config.yaml> <output_root>" >&2
    exit 1
fi
config=$1
out=$2

EDGES=${EDGES:-"0 5 10"}         # train.edge_pixels values to compare (training loss only)
SEEDS=${SEEDS:-"0 1 2"}
EPOCHS=${EPOCHS:-20}             # identical for every run: it also sets the cosine LR schedule
VAL_EDGE=${VAL_EDGE:-0}          # val loss cells (picks best.pt); identical for every run
TRIM=${TRIM:-10}                 # interior-only test scoring

here="$(dirname "$0")"
var=$(python3 -c "import sys, yaml; print('_'.join(yaml.safe_load(open(sys.argv[1]))['data']['target_vars']))" "$config")
echo "edge study: edges [$EDGES]  seeds [$SEEDS]  epochs $EPOCHS  val_edge_pixels $VAL_EDGE  trim $TRIM  -> $out"

for e in $EDGES; do
    for s in $SEEDS; do
        run="$out/edge${e}_seed${s}"
        echo "########## edge_pixels $e  seed $s  -> $run"
        if [ -f "$run/$var/full/best.pt" ]; then
            echo "already trained, skipping training"
        else
            bash "$here/train_folds.sh" "$config" --output_dir "$run" \
                --epochs "$EPOCHS" --seed "$s" --train_edge_pixels "$e" --val_edge_pixels "$VAL_EDGE"
        fi
        bash "$here/evaluate_folds.sh" "$config" --output_dir "$run"
        bash "$here/evaluate_folds.sh" "$config" --output_dir "$run" --edge_pixels "$TRIM"
    done
done

echo "done. results table: python3 $here/summarize_runs.py $out"
