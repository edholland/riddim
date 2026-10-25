#!/bin/bash
# Double-click: a download queue. Every list dropped into references/queue/
# is fetched into references/ (archive.org items directly, everything else
# through yt-dlp), then moved to references/queue/done/. Keeps watching for
# new lists until the window is closed. Logged to references/fetch.log.
cd "$(cd "$(dirname "$0")" && pwd)" || exit 1
mkdir -p references/queue/done
exec > >(tee -a references/fetch.log) 2>&1
echo "=== fetch queue $(date) ==="
[ -x .venv/bin/python ] || python3 -m venv .venv
.venv/bin/python -m pip install --quiet --upgrade pip yt-dlp
while true; do
    for f in references/queue/*.txt; do
        [ -e "$f" ] || continue
        echo "=== LIST $f $(date) ==="
        caffeinate -dimsu .venv/bin/python -u ml/collect_refs.py "$f" --out references --prefer-mp3 --jobs 4
        mv "$f" references/queue/done/
        echo "=== LIST DONE $f $(date) ==="
    done
    sleep 60
done
