#!/bin/sh
# Double-click in Finder to open the riddim UI (studio sound included).
#
# First run makes a project venv in .venv with what the UI needs - mido and
# python-rtmidi for the Deluge, numpy and scipy for studio sound - and
# riddim-cli uses it from then on. Close this window to stop the server.
here=$(cd "$(dirname "$0")" && pwd)
cd "$here" || exit 1

if ! { [ -x .venv/bin/python ] && .venv/bin/python -c "import mido, rtmidi, numpy, scipy" 2>/dev/null; }; then
    echo "setting up .venv (first run only)..."
    [ -x .venv/bin/python ] || python3 -m venv .venv || { echo "python3 -m venv failed"; read -r _; exit 1; }
    .venv/bin/python -m pip install --quiet --upgrade pip
    .venv/bin/python -m pip install --quiet "mido>=1.3" "python-rtmidi>=1.5" "numpy>=1.21" "scipy>=1.7" \
        || { echo "pip install failed - see above"; read -r _; exit 1; }
fi

exec ./riddim-cli ui "$@"
