"""Measure the judge's training and test material: reference records, clean
and with known flaws on them (riddim/defects.py), through riddim/ears.py.

    python ml/judge_data.py refs  --meta refs.json --audio refs/ --out judge_feats.npz [--flaws 12]

refs.json lists the records: [{"id", "family", "artist", "excerpts": [file, ...]}].
Each excerpt is measured clean, with `--flaws` random (flaw, size) pairs, and
with one "nudge" (a 0.5 dB wiggle at a random frequency: a pair nobody could
tell apart). Everything is per window, so pairs line up window by window.
"""

from __future__ import annotations

import argparse
import json
import zlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from riddim import defects, ears  # noqa: E402


def nudge(x, rng):
    f0 = float(np.exp(rng.uniform(np.log(40), np.log(10000))))
    return defects._filt(x, defects._biquad("peak", f0, float(rng.choice([-0.5, 0.5])), 1.0))


def one(job):
    path, seed, n_flaws = job
    rng = np.random.default_rng(seed)
    try:
        x = ears.load(path)
    except Exception:
        return None
    if len(x) < ears.SR * ears.WIN_S:
        return None
    rows = [("clean", 0.0, ears.windows(x))]
    cat = defects.CATALOGUE
    for k in rng.choice(len(cat), size=min(n_flaws, len(cat)), replace=False):
        name, _, sizes = cat[k]
        si = int(rng.integers(len(sizes)))
        y = ears.normalize(defects.apply(x, name, sizes[si]))
        rows.append((name, float(si), ears.windows(y)))
    rows.append(("nudge", 0.0, ears.windows(ears.normalize(nudge(x, rng)))))
    return rows


def refs(a):
    meta = json.load(open(a.meta))
    jobs, info = [], []
    for r in meta:
        for e in r["excerpts"]:
            jobs.append((os.path.join(a.audio, e), zlib.crc32(f"{r['id']}:{e}".encode()), a.flaws))
            info.append((r["id"], r["family"], r["artist"], e))
    F, rec, fam, art, exc, flaw, size, win = [], [], [], [], [], [], [], []
    with ProcessPoolExecutor(a.jobs) as ex:
        for i, rows in enumerate(ex.map(one, jobs, chunksize=4)):
            if i % 100 == 0:
                print(f"{i}/{len(jobs)}", flush=True)
            if not rows:
                continue
            rid, fm, ar, e = info[i]
            for name, si, w in rows:
                for j, v in enumerate(w):
                    F.append(v); rec.append(rid); fam.append(fm); art.append(ar); exc.append(e)
                    flaw.append(name); size.append(si); win.append(j)
    np.savez_compressed(a.out, F=np.array(F, np.float32), names=np.array(ears.NAMES), rec=np.array(rec),
                        family=np.array(fam), artist=np.array(art), excerpt=np.array(exc),
                        flaw=np.array(flaw), size=np.array(size, np.float32), win=np.array(win))
    print("wrote", a.out, len(F), "windows")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("refs")
    r.add_argument("--meta", required=True)
    r.add_argument("--audio", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--flaws", type=int, default=12)
    r.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    a = p.parse_args(argv)
    if a.cmd == "refs":
        refs(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
