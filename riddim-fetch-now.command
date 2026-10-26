#!/bin/bash
# Double-click: fetch the queued lists named on the line below right now,
# alongside the running queue (which is busy with the long radio list).
cd "$(cd "$(dirname "$0")" && pwd)" || exit 1
exec > >(tee -a references/fetch-now.log) 2>&1
for f in references/queue/a_samples.txt references/queue/c_fma.txt references/queue/d_streams.txt; do
    [ -e "$f" ] || continue
    mv "$f" "references/queue/now_$(basename "$f").run"
    g="references/queue/now_$(basename "$f").run"
    echo "=== LIST $g $(date) ==="
    caffeinate -dimsu .venv/bin/python -u ml/collect_refs.py "$g" --out references --prefer-mp3 --jobs 4
    mv "$g" references/queue/done/
    echo "=== LIST DONE $g $(date) ==="
done
