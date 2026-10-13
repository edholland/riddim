#!/bin/sh
# After the overnight run: judge it, render it, and get it off the machine.
# Waits for models/v2/train.log to say DONE (or for a deadline, then uses
# the best checkpoint so far), evaluates base / v1 / v2 on riddim songs,
# and uploads v2's best checkpoint with the report as a GitHub release.
# usage: morning.sh [DEADLINE_HHMM]   (default 0645)
set -u
here=$(cd "$(dirname "$0")" && pwd -P)
H=${ML_HOME:-$HOME/ml}; py="$H/venv/bin/python"; gh=${GH:-$HOME/.local/bin/gh}
deadline=${1:-0645}
cd "$H"
while ! grep -q DONE models/v2/train.log 2>/dev/null; do
    [ "$(date +%H%M)" -ge "$deadline" ] && [ "$(date +%H%M)" -lt 1200 ] && {
        echo "deadline: stopping training at $(date)"; pkill -f run_overnight.sh; pkill -f train_mlx.py; sleep 20; break; }
    sleep 120
done
echo "== eval $(date)"
caffeinate -dimsu "$py" "$here/eval_leads.py" out/eval_v2 models/music-medium-800k models/ft_dmz/final \
    models/v2/best --songs 24 --audio 6
echo "== release $(date)"
tar cf release/riddim-lead-v2.tar -C models/v2 best
cp out/eval_v2/report.md release/eval_v2.md
grep valid models/v2/train.log > release/v2_valid_curve.txt
best=$(cat models/v2/best.json 2>/dev/null)
"$gh" release create models-v2 --repo edholland/riddim --target show \
    --title "Lead model v2 (role-normalised data, lead-task training, MLX)" \
    --notes "Trained overnight with ml/pipeline.sh (see ml/README.md): 12,327 deduplicated Lakh songs (CC-BY 4.0) normalised onto riddim's band, the inference task as training data, plus riddim's own band grammar; from v1. Best checkpoint: $best

Install: tar xf riddim-lead-v2.tar -C ~/ml/models/v2  (riddim prefers models/v2/best automatically).

$(cat out/eval_v2/report.md)" \
    release/riddim-lead-v2.tar release/eval_v2.md release/v2_valid_curve.txt
echo "== done $(date)"
