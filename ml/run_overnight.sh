#!/bin/sh
# The overnight training run, supervised: if the trainer dies (memory, a
# crash), carry on from the last checkpoint rather than lose the night.
# usage: run_overnight.sh OUT_DIR STEPS INIT_MODEL MIX [extra train_mlx args]
set -u
here=$(cd "$(dirname "$0")" && pwd -P)
py=${ML_PYTHON:-$HOME/ml/venv/bin/python}
out=$1; steps=$2; init=$3; mix=$4; shift 4
mkdir -p "$out"
for attempt in 1 2 3 4 5 6 7 8; do
    if [ -f "$out/latest/step.json" ]; then
        start=$(sed 's/[^0-9]//g' "$out/latest/step.json")
        from="$out/latest"
    else
        start=0; from=$init
    fi
    echo "== attempt $attempt from step $start ($from) $(date)"
    caffeinate -dimsu "$py" -u "$here/train_mlx.py" "$out" --init "$from" --mix "$mix" \
        --steps "$steps" --start "$start" "$@" >> "$out/train.log" 2>&1
    grep -q DONE "$out/train.log" && exit 0
    sleep 30
done
