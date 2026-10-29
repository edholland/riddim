"""Measured targets: what the records do, and whether a change moves riddim
toward it. This is the rule since the judges failed Ed's blind test
(docs/goals.md).

Every target is measured the same way on riddim's renders and on the
records:
- drum feel (ml/drum_feel.py): onset jitter and lean, four-bar repeat,
  hit-strength spread
- band shares, RMS and crest (riddim.listen)

The records' figure for a profile is the median over its families (stats in
references/stats/stats.jsonl, drum feel on 30 records).

ab() renders a change both ways on the same seeds and gives each target's
mean paired difference with a 95% bootstrap interval, alongside where the
records sit. A change is kept when it moves its target toward the records
beyond the interval and pushes no other target away beyond its own.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [REPO, HERE, os.path.join(HERE, "critic")]

import drum_feel  # noqa: E402
import library_stats as ls  # noqa: E402

FEEL = ("jitter_ms", "lean_ms", "repeat", "cv_kick", "cv_snare", "cv_hat")
BANDS = ("sub", "bass", "low_mid", "high_mid", "air")
LEVEL = ("rms_db", "crest_db", "sub_held")
TARGETS = FEEL + BANDS + LEVEL
STEM = ("kick_share", "snare_share", "hat_share", "hat_centroid_hz", "hat_decay_ms", "snare_body_db")


def stem_measure(path: str) -> dict[str, float]:
    """The drum stem's character (ml/drum_stem.py), split as the records'
    were; cached beside the render."""
    import shutil
    import tempfile
    import drum_stem
    import mine_cells
    cache = path[:-4] + ".drumstem.json"
    if os.path.exists(cache):
        return json.load(open(cache))
    tmp = tempfile.mkdtemp(prefix="tstem_")
    try:
        mine_cells.split(path, os.path.join(tmp, "s"))
        m = drum_stem.metrics(os.path.join(tmp, "s", "drums.wav"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    json.dump(m, open(cache, "w"))
    return m


def stem_records(profile: str) -> dict[str, float]:
    import goal_check as gc
    rec = json.load(open(os.path.join(REPO, "references", "stats", "drum_stem_records.json")))
    fams = [f for f in gc.families_for(profile) if f in rec]
    return {k: float(np.mean([rec[f][k] for f in fams])) for k in STEM}


def measure(path: str) -> dict[str, float]:
    from riddim import listen
    f = drum_feel.feel(path) or {}
    rep = listen.analyze(path)
    out = {k: f.get(k) for k in FEEL}
    out.update({b: rep.band_share.get(b) for b in BANDS})
    out.update(rms_db=rep.rms_db, crest_db=rep.crest_db, sub_held=rep.sub_held)
    return out


_RECORDS: dict = {}


def records(profile: str, n: int = 30) -> dict[str, float]:
    """The records' median per target for a profile's families."""
    if profile not in _RECORDS:
        import goal_check as gc
        fams = gc.families_for(profile)
        st = [ls.fold(r) for r in ls._load(os.path.join(REPO, "references", "stats", "stats.jsonl"))
              if r.get("family") in fams]
        out = {b: float(np.median([r["band_share"][b] for r in st])) for b in BANDS}
        out.update({k: float(np.median([r[k] for r in st if r.get(k) is not None]))
                    for k in ("rms_db", "crest_db", "jitter_ms", "lean_ms", "sub_held")})
        feels = [x for x in (drum_feel.feel(f) for f in drum_feel.records(fams, n, os.path.join(REPO, "references")))
                 if x]
        for k in ("repeat", "cv_kick", "cv_snare", "cv_hat"):
            out[k] = float(np.median([x[k] for x in feels if x.get(k) is not None]))
        _RECORDS[profile] = out
    return _RECORDS[profile]


# Advisory targets: reported, but a move away does not block a change. The
# kick's onset-strength spread on the full mix follows the arrangement's
# dynamics (drops, pulls, the kick thinned in pulls) far more than the kit -
# ours is about double the records' whatever the kit does.
ADVISORY = {"cv_kick"}

# The smallest move that matters, per target: a move away from the records
# smaller than this is collateral, whatever its interval (16 seeds resolve
# band-share changes of 0.001, which nobody hears).
TOLERANCE = {"jitter_ms": 0.5, "lean_ms": 0.5, "repeat": 0.02, "cv_kick": 0.02, "cv_snare": 0.02, "cv_hat": 0.02,
             "sub": 0.005, "bass": 0.005, "low_mid": 0.005, "high_mid": 0.005, "air": 0.005,
             "rms_db": 0.5, "crest_db": 0.5, "sub_held": 0.02, "kick_share": 0.01, "snare_share": 0.01, "hat_share": 0.01,
             "hat_centroid_hz": 100.0, "hat_decay_ms": 5.0, "snare_body_db": 1.0}


def interval(v: np.ndarray, draws: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) < 3:
        return float("nan"), float("nan"), float("nan")
    means = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(draws)]
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def ab(profile: str, a_changes: list, b_changes: list, seeds, out_dir: str,
       watch: tuple[str, ...] = (), stems: bool = False) -> dict:
    """B against A on `seeds`: per target, A's and B's means, the records',
    the paired difference with its interval, and whether B moved toward the
    records (+1), away (-1) or neither (0) beyond the interval. `watch`
    names the targets the change aims at; the verdict is keep when they move
    toward the records and nothing moves away."""
    from blind_ab import patched
    from riddim.compose import compose
    from riddim.render import audio
    rec = records(profile)
    m = {}
    for side, changes in (("a", a_changes), ("b", b_changes)):
        os.makedirs(os.path.join(out_dir, side), exist_ok=True)
        m[side] = []
        for s in seeds:
            f = os.path.join(out_dir, side, f"{profile}_{s}.wav")
            if not os.path.exists(f):
                with patched(changes):
                    audio.write(compose(profile, seed=s), f)
            cache = f[:-4] + ".targets.json"
            if os.path.exists(cache):
                x = json.load(open(cache))
            else:
                x = measure(f)
                json.dump(x, open(cache, "w"))
            if stems:
                x = {**x, **stem_measure(f)}
            m[side].append(x)
    if stems:
        rec = {**rec, **stem_records(profile)}
    res = {}
    for t in TARGETS + (STEM if stems else ()):
        a = np.array([x.get(t) if x.get(t) is not None else np.nan for x in m["a"]], float)
        b = np.array([x.get(t) if x.get(t) is not None else np.nan for x in m["b"]], float)
        d = b - a
        mean, lo, hi = interval(d)
        r = rec.get(t)
        am, bm = float(np.nanmean(a)), float(np.nanmean(b))
        move = 0
        if r is not None and not np.isnan(lo):
            sure = lo > 0 or hi < 0
            if sure:
                closer = abs(bm - r) < abs(am - r)
                # Away counts only beyond the target's tolerance.
                move = 1 if closer else (-1 if abs(abs(bm - r) - abs(am - r)) > TOLERANCE.get(t, 0) else 0)
        res[t] = {"a": am, "b": bm, "records": r, "diff": (mean, lo, hi), "move": move}
    aimed = [t for t in watch if t in res]
    ok = bool(aimed) and all(res[t]["move"] == 1 for t in aimed) and \
        not any(v["move"] == -1 for t, v in res.items() if t not in ADVISORY)
    res["verdict"] = "keep" if ok else ("revert" if any(res[t]["move"] == -1 for t in aimed) else "can't tell")
    return res


def table(res: dict) -> str:
    lines = ["| target | A | B | records | B - A (95% CI) | toward records |", "|---|---|---|---|---|---|"]
    for t in TARGETS + tuple(k for k in STEM if k in res):
        r = res[t]
        rec = f"{r['records']:.3f}" if r["records"] is not None else "-"
        d = r["diff"]
        lines.append(f"| {t} | {r['a']:.3f} | {r['b']:.3f} | {rec} | {d[0]:+.3f} ({d[1]:+.3f} to {d[2]:+.3f}) | "
                     f"{ {1: 'yes', -1: 'AWAY', 0: '-'}[r['move']] } |")
    return "\n".join(lines) + f"\n\nVerdict: {res['verdict']}\n"
