#!/usr/bin/env bash
## preprocess several NorFjords regions (configs/base_<R>.yaml, cubic, temp/salt/u/v + static) into
## data/processed_<R>/, one job per region and year, JOBS of them side by side, 4 threads each
## (more threads per job only oversubscribes the node: same speed, much higher load).
## finished days are skipped and daily files are written atomically, so after a crash just start it again.
##
## usage (from anywhere; uses this worktree's code and the repo venv):
##   bash scripts/preprocess_regions.sh                 # all regions not done yet, in a detached screen raid_prep
##   screen -r raid_prep                                # watch (detach with ctrl-a d)
##   REGIONS="A01 A02" JOBS=8 bash scripts/preprocess_regions.sh run   # a subset, in the foreground
## logs: <data>/processed_<R>/preprocess_<year>.log, overview in <data>/preprocess_regions.log
set -euo pipefail

here="$(cd "$(dirname "$0")/.." && pwd)"
VENV=${VENV:-/gpfs/gpfs0/osea/manuel/RAID/.venv}
D=${D:-/data/prosjekt/15915_RAID/data}
REGIONS=${REGIONS:-"A01 A02 A06 A07 A08 A09 A10 A11 A12 A13"}   # A04 = data/processed, A03 and A05 are done
YEARS=${YEARS:-"2023 2024 2025 2026"}
JOBS=${JOBS:-12}

job() {   # job <region> <year>
    local r=$1 y=$2
    mkdir -p "$D/processed_$r"
    echo "$(date '+%F %T') start $r $y"
    if OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
       python3 -u scripts/preprocess.py --config "configs/base_$r.yaml" --years "$y" \
       >> "$D/processed_$r/preprocess_$y.log" 2>&1; then
        echo "$(date '+%F %T') done  $r $y"
    else
        echo "$(date '+%F %T') FAILED $r $y (see $D/processed_$r/preprocess_$y.log)"
    fi
}
export -f job
export D

case "${1:-}" in
    run)
        cd "$here"
        source "$VENV/bin/activate"
        export PYTHONPATH="$here/src"          # this branch's raid package
        echo "=== $(date '+%F %T') preprocess [$REGIONS] years [$YEARS], $JOBS jobs, code $(git -C "$here" rev-parse --short HEAD)"
        for r in $REGIONS; do [ -f "configs/base_$r.yaml" ] || { echo "no configs/base_$r.yaml" >&2; exit 1; }; done
        ## larger years first, so the 12-day 2023 jobs fill the gaps at the end
        { for y in $YEARS; do [ "$y" = 2023 ] || for r in $REGIONS; do echo "$r $y"; done; done
          for y in $YEARS; do [ "$y" != 2023 ] || for r in $REGIONS; do echo "$r $y"; done; done; } \
            | xargs -P "$JOBS" -L 1 bash -c 'job "$0" "$1"'
        echo "=== $(date '+%F %T') preprocess done"
        ;;
    "")
        if screen -ls | grep -q "\.raid_prep[[:space:]]"; then
            echo "screen raid_prep already exists, not starting it again"
            exit 0
        fi
        screen -dmS raid_prep bash -c "REGIONS='$REGIONS' YEARS='$YEARS' JOBS=$JOBS bash '$here/scripts/preprocess_regions.sh' run 2>&1 | tee -a '$D/preprocess_regions.log'"
        echo "started screen raid_prep (overview log $D/preprocess_regions.log)"
        ;;
    *)
        echo "usage: $0 [run]" >&2
        exit 1
        ;;
esac
