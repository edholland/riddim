#!/bin/bash
# Double-click in Finder: push riddim (and the Deluge firmware branch) to
# GitHub, then fetch every reference list (refs*.txt) into references/.
# Safe to run again: pushed branches are no-ops, fetched audio is skipped.
# Everything is logged to references/sync.log; close the window to stop.
here=$(cd "$(dirname "$0")" && pwd)
cd "$here" || exit 1
mkdir -p references
log="$here/references/sync.log"
exec > >(tee -a "$log") 2>&1
echo "=== riddim sync $(date) ==="
export GIT_TERMINAL_PROMPT=0

# -- GitHub ---------------------------------------------------------------
branch=$(git branch --show-current)
if git push -u origin "$branch"; then echo "PUSH riddim $branch: ok"; else echo "PUSH riddim $branch: FAILED"; fi

fw_bundle="$here/deluge-firmware/ai-kit-automation.bundle"
if [ -f "$fw_bundle" ]; then
    fw="$HOME/repos/DelugeFirmware"
    if [ ! -d "$fw/.git" ]; then
        fw="$HOME/repos/.riddim-fw-push"
        [ -d "$fw/.git" ] || git clone --filter=blob:none --no-checkout https://github.com/edholland/DelugeFirmware.git "$fw"
    fi
    if git -C "$fw" fetch "$fw_bundle" "ai-kit-automation:refs/remotes/bundle/ai-kit-automation" \
        && git -C "$fw" push https://github.com/edholland/DelugeFirmware.git "refs/remotes/bundle/ai-kit-automation:refs/heads/ai-kit-automation"; then
        echo "PUSH firmware ai-kit-automation: ok"
    else
        echo "PUSH firmware ai-kit-automation: FAILED"
    fi
fi

# -- reference audio ------------------------------------------------------
[ -x .venv/bin/python ] || python3 -m venv .venv
.venv/bin/python -m pip install --quiet --upgrade pip yt-dlp
lists=$(ls refs_more.txt refs_*.txt refs.txt 2>/dev/null | awk '!seen[$0]++' | grep -v summary)
echo "FETCH lists: $lists"
caffeinate -dimsu .venv/bin/python -u ml/collect_refs.py $lists --out references --prefer-mp3 --jobs 4
echo "=== FETCH DONE $(date) ==="
