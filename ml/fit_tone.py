"""Fit a profile's master tone (TONE's five bands) and its kick's high-pass
jointly to the records' band shares and drum-stem kick/snare/hat shares.

    python ml/fit_tone.py roots_dub            # seeds 1-8, ~40 evaluations

One knob at a time fails on band shares: they sum to one, so a cut that
brings one band to the records pushes a neighbour past its tolerance. Here
the bands move together. Each band stays within +-3 dB of its current
value and the kick high-pass within 30-60 Hz (off below 31). The objective
is the sum of squared gaps in tolerance units (ml/targets.py) over the
mix's five bands, with a penalty when any band or any of the drum stem's
three shares ends further from the records than it starts (as the A/B rule
would reject it); the drum shares taken on
the clean drums bus (which tracks Demucs' drum stem within a few points:
docs/goals.md). Every evaluation is logged to references/stats/fit_tone_<profile>.jsonl.

Confirm what it finds with ml/targets.py on other seeds (101-116).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [REPO, HERE]

import targets  # noqa: E402

BANDS = targets.BANDS
DRUMS = ("kick_share", "snare_share", "hat_share")
SPAN = 3.0
HP = (30.0, 60.0)
DECAY = (60.0, 800.0)   # kick_sub_decay_ms, with --decay


def shares(x: np.ndarray, sr: int) -> dict[str, float]:
    from riddim import listen
    mono = x.mean(axis=1)
    e = {b: float(np.mean(listen._band(mono, lo, hi, sr) ** 2)) for b, (lo, hi) in listen.BANDS.items()}
    tot = sum(e.values()) or 1.0
    return {b: v / tot for b, v in e.items()}


def evaluate(profile: str, songs: list, base_tone: dict, base_shape: dict, v: np.ndarray) -> dict:
    import soundfile as sf
    import drum_stem
    from riddim.render import audio
    tone = {b: base_tone.get(b, 0.0) + float(d) for b, d in zip(BANDS, v[:5])}
    hp = float(v[5])
    shape = {**base_shape, **({"kick_hp": hp} if hp > HP[0] + 1 else {})}
    shape.pop("kick_hp", None) if hp <= HP[0] + 1 else None
    if len(v) > 6:
        shape["kick_sub_decay_ms"] = float(v[6])
    rows = []
    f = os.path.join(tempfile.gettempdir(), f"fit_{profile}_drums.wav")
    old_t, old_s = audio.TONE.get(profile), audio.DRUM_SHAPE.get(profile)
    audio.TONE[profile], audio.DRUM_SHAPE[profile] = tone, shape
    try:
        for song in songs:
            so: dict = {}
            y = audio.render(song, stems_out=so)
            m = shares(y, audio.SR)
            sf.write(f, so["drums"], audio.SR)
            m.update({k: drum_stem.metrics(f)[k] for k in DRUMS})
            rows.append(m)
    finally:
        audio.TONE[profile] = old_t
        if old_s is None:
            audio.DRUM_SHAPE.pop(profile, None)
        else:
            audio.DRUM_SHAPE[profile] = old_s
    return {k: float(np.mean([r[k] for r in rows])) for k in BANDS + DRUMS} | {"tone": tone, "shape": shape}


def main(argv=None) -> int:
    from scipy.optimize import minimize
    from riddim.compose import compose
    from riddim.render import audio
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("profile")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--evals", type=int, default=40)
    ap.add_argument("--decay", type=float, default=0.0,
                    help="also fit the kick's sub decay (DRUM_SHAPE kick_sub_decay_ms), from this many ms")
    a = ap.parse_args(argv)
    p = a.profile
    rec = {**targets.records(p), **targets.stem_records(p)}
    songs = [compose(p, seed=s) for s in range(1, a.seeds + 1)]
    base_tone = dict(audio.TONE.get(p, {}))
    base_shape = dict(audio.DRUM_SHAPE.get(p, {}))
    hp0 = base_shape.get("kick_hp", HP[0])
    log = os.path.join(REPO, "references", "stats", f"fit_tone_{p}.jsonl")
    lo = np.array([-SPAN] * 5 + [HP[0]] + ([DECAY[0]] if a.decay else []))
    hi = np.array([SPAN] * 5 + [HP[1]] + ([DECAY[1]] if a.decay else []))
    best = {"loss": np.inf}

    def loss(v):
        v = np.clip(v, lo, hi)
        m = evaluate(p, songs, base_tone, base_shape, v)
        gaps = {k: (m[k] - rec[k]) / targets.TOLERANCE[k] for k in BANDS + DRUMS}
        if "gaps0" not in best:
            best["gaps0"] = {k: abs(g) for k, g in gaps.items()}
        # The bands are fitted, but no target - band or drum share (the kit's
        # balance, which the tone can't fix) - may end further from the
        # records than it starts by more than its tolerance: the A/B rule
        # (ml/targets.py) rejects any trade of one band for another.
        L = float(sum(gaps[k] ** 2 for k in BANDS) +
                  25 * sum(max(0.0, abs(gaps[k]) - best["gaps0"][k] - 0.8) ** 2 for k in gaps))
        row = {"v": [round(float(x), 2) for x in v], "loss": round(L, 2),
               "gaps": {k: round(g, 2) for k, g in gaps.items()}, "tone": m["tone"], "shape": m["shape"]}
        with open(log, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"{L:9.1f}  " + " ".join(f"{x:+.2f}" for x in v[:5]) + f"  hp {v[5]:.0f}" + (f" decay {v[6]:.0f}" if len(v) > 6 else "") + "  " +
              " ".join(f"{k} {g:+.1f}" for k, g in gaps.items()), flush=True)
        if L < best["loss"]:
            best.update(loss=L, row=row)
        return L

    x0 = np.array([0.0] * 5 + [hp0] + ([a.decay] if a.decay else []))
    n = len(x0)
    step = [1.5] * 5 + [10.0] + [0.4 * (a.decay or 0)]
    simplex = [x0] + [x0 + np.eye(n)[i] * step[i] for i in range(n)]
    minimize(loss, x0, method="Nelder-Mead",
             options={"maxfev": a.evals, "initial_simplex": np.array(simplex), "xatol": 0.1, "fatol": 1.0})
    print("BEST", json.dumps(best["row"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
