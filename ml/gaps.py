"""Where riddim stands on every measured target, per profile: the mean over a
render folder's songs against the records' figure, in units of the target's
tolerance (ml/targets.py), largest gap first.

    python ml/gaps.py goals6            # goal_check.py's --out folder
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]

import targets  # noqa: E402

# Gaps that are the measurement's, not the music's (docs/goals.md).
ARTIFACT = {
    # Demucs files a 140 sub under drums: the drum and bass stems' shares
    # in the 140 profiles say more about the split than the kit.
    ("dubstep_140", "kick_share"), ("dubstep_140", "snare_share"), ("dubstep_140", "hat_share"),
    ("dark_garage", "kick_share"), ("dark_garage", "snare_share"), ("dark_garage", "hat_share"),
    ("dubstep_140", "bass_sub"), ("dark_garage", "bass_sub"),
}
ADVISORY = targets.ADVISORY


def measure(folder: str, profile: str) -> dict[str, float]:
    rows = []
    for f in sorted(glob.glob(os.path.join(folder, f"{profile}_*.wav"))):
        cache = f[:-4] + ".targets.json"
        if os.path.exists(cache):
            x = json.load(open(cache))
        else:
            x = targets.measure(f)
            json.dump(x, open(cache, "w"))
        rows.append({**x, **targets.stem_measure(f)})
    keys = targets.TARGETS + targets.STEM
    return {k: float(np.nanmean([r[k] if r.get(k) is not None else np.nan for r in rows])) for k in keys} | \
        {"n": len(rows)}


def table(folder: str, profiles) -> list[dict]:
    out = []
    for p in profiles:
        ours = measure(folder, p)
        rec = {**targets.records(p), **targets.stem_records(p)}
        for k, r in rec.items():
            if k not in ours or np.isnan(ours[k]):
                continue
            tol = targets.TOLERANCE.get(k)
            if not tol:
                continue
            note = "artifact" if (p, k) in ARTIFACT else ("advisory" if k in ADVISORY else "")
            out.append({"profile": p, "target": k, "ours": ours[k], "records": r,
                        "units": (ours[k] - r) / tol, "note": note})
    return sorted(out, key=lambda g: -abs(g["units"]))


def main(argv=None) -> int:
    import goal_check
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folder")
    ap.add_argument("--top", type=int, default=40)
    a = ap.parse_args(argv)
    rows = table(a.folder, goal_check.PROFILES)
    json.dump(rows, open(os.path.join(a.folder, "gaps.json"), "w"), indent=1)
    print("| profile | target | ours | records | gap (tolerances) | note |\n|---|---|---|---|---|---|")
    for g in rows[:a.top]:
        print(f"| {g['profile']} | {g['target']} | {g['ours']:.3f} | {g['records']:.3f} | {g['units']:+.1f} | {g['note']} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
