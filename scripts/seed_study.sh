#!/usr/bin/env bash
## train and evaluate one config with several seeds, everything else identical:
## for each seed: train_folds.sh (FOLDS), then evaluate_folds.sh on all sea cells and on the interior (TRIM).
## runs go to <output_root>/<LABEL>_seed<s>/<var>/<fold>; a seed whose last fold already has best.pt is not
## retrained, so after a crash just start the script again. Extra arguments go to every train.py run.
##
## usage (activate the venv first: source .venv/bin/activate), e.g. multi-region u+v vs A04 only:
## LABEL=regions CUDA_VISIBLE_DEVICES=0 bash scripts/seed_study.sh configs/step1b_unet_uv_regions.yaml /data/prosjekt/15915_RAID/train/step1B/u_v/20261120_regions
## LABEL=a04 SEEDS="1 2" CUDA_VISIBLE_DEVICES=1 bash scripts/seed_study.sh configs/step1b_unet_uv.yaml /data/prosjekt/15915_RAID/train/step1B/u_v/20261120_regions
## then the results table (pools seeds per LABEL):
## python3 scripts/summarize_runs.py <output_root> --eval eval_test_trim10
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "usage: $0 <config.yaml> <output_root> [train.py args...]" >&2
    exit 1
fi
config=$1
out=$2
shift 2

LABEL=${LABEL:-run}
SEEDS=${SEEDS:-"0 1 2"}
EPOCHS=${EPOCHS:-20}             # identical for every run: it also sets the cosine LR schedule
FOLDS=${FOLDS:-"val2024 val2025"}   # the full model has no validation and is unreliable; add "full" if wanted
TRIM=${TRIM:-10}                 # interior-only test scoring
export FOLDS

here="$(dirname "$0")"
var=$(python3 -c "import sys, yaml; print('_'.join(yaml.safe_load(open(sys.argv[1]))['data']['target_vars']))" "$config")
last=$(echo $FOLDS | awk '{print $NF}')
echo "seed study $LABEL: config $config  seeds [$SEEDS]  epochs $EPOCHS  folds [$FOLDS]  trim $TRIM  -> $out"

for s in $SEEDS; do
    run="$out/${LABEL}_seed${s}"
    echo "########## $LABEL seed $s -> $run"
    if [ -f "$run/$var/$last/best.pt" ] && [ -f "$run/$var/$last/train.log" ] && \
       [ "$(grep -c '^epoch' "$run/$var/$last/train.log")" -ge "$EPOCHS" ]; then
        echo "already trained, skipping training"
    else
        bash "$here/train_folds.sh" "$config" --output_dir "$run" --epochs "$EPOCHS" --seed "$s" "$@"
    fi
    bash "$here/evaluate_folds.sh" "$config" --output_dir "$run"
    bash "$here/evaluate_folds.sh" "$config" --output_dir "$run" --edge_pixels "$TRIM"
done

echo "done. results table: python3 $here/summarize_runs.py $out --eval eval_test_trim10"
