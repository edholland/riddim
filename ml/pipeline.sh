#!/bin/sh
# The whole training pipeline, from nothing to a lead-writing model riddim
# uses. Every stage skips itself when its output already exists, so it
# resumes after an interruption. Runs on Apple silicon (MLX); ~16h on an
# M5 Pro 48GB, most of it the training run.
#
#   ML_HOME=~/ml ml/pipeline.sh
#
# Needs: $ML_HOME/venv with torch, transformers, mlx, mido, numpy, scipy and
# the anticipation package (pip install --no-deps git+https://github.com/jthickstun/anticipation).
set -eu
here=$(cd "$(dirname "$0")" && pwd -P)
H=${ML_HOME:-$HOME/ml}
py="$H/venv/bin/python"
cd "$H"
mkdir -p data out logs models
LMD=http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz

# 1. The base model: the Anticipatory Music Transformer, 360M (Apache-2.0)
[ -f models/music-medium-800k/config.json ] || "$py" -c "
from huggingface_hub import snapshot_download
snapshot_download('stanford-crfm/music-medium-800k', local_dir='models/music-medium-800k')"

# 2. Data: Lakh MIDI, full (CC-BY 4.0), fetched as parallel ranges
if [ ! -f data/lmd_full/extract.done ]; then
    mkdir -p data/lmd_full
    size=$(curl -sI "$LMD" | tr -d '\r' | awk 'tolower($1)=="content-length:"{print $2}')
    "$py" "$here/rangefetch.py" "$LMD" data/lmd_full/lmd_full.tar.gz 16 "$size"
    (cd data/lmd_full && tar xzf lmd_full.tar.gz && touch extract.done)
fi

# 3. Score every file for the early-dubstep / dub sound by what it plays
[ -f out/dmz_scores_full.jsonl ] || "$py" "$here/dmz_filter.py" data/lmd_full/lmd_full out/dmz_scores_full.jsonl

# 4. Role-normalised, tempo-moved, transposed, deduplicated - two tiers
[ -f data/v2_top/stats.json ] || "$py" "$here/build_data.py" out/dmz_scores_full.jsonl data/v2_top \
    --min-score 3.5 --variants 4 --full-augment 2
[ -f data/v2_broad/stats.json ] || "$py" "$here/build_data.py" out/dmz_scores_full.jsonl data/v2_broad \
    --min-score 2.0 --max-score 3.5 --variants 2 --full-augment 1

# 5. riddim's own songs: the band grammar, never its leads
[ -f data/v2_riddim/full/valid.npy ] || "$py" "$here/riddim_corpus.py" data/v2_riddim --songs 4000

# 6. Train (MLX, compiled, supervised: resumes from the last checkpoint)
MIX=lead_top=data/v2_top/lead:0.3,lead_broad=data/v2_broad/lead:0.3,full_top=data/v2_top/full:0.15,full_broad=data/v2_broad/full:0.15,riddim=data/v2_riddim/full:0.1
INIT=${INIT:-models/music-medium-800k}
[ -f models/v2/best/config.json ] && grep -q DONE models/v2/train.log 2>/dev/null || \
    "$here/run_overnight.sh" models/v2 "${STEPS:-32000}" "$INIT" "$MIX" --batch 8 --accum 1 --lr 1.5e-5 --eval 1000

# 7. Judge it the way riddim uses it, against riddim's own leads
"$py" "$here/eval_leads.py" out/eval_v2 models/music-medium-800k models/v2/best --songs 24 --audio 6
