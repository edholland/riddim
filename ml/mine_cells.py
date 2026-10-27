"""Mine the records' bassline and skank rhythms, per family, for riddim's
rhythm pools - what bass players on real records play, instead of only what
the generator's knobs can reach.

Runs where the audio is (the Mac), after `library_stats.py plan`. Each record
is split once (htdemucs on the middle of the track) and its bass, drums and
other stems kept small under references/stats/_cells/, so a later step (the
hook model's transcriptions) can reuse them.

    python ml/mine_cells.py split   --per-family 40      # Demucs, kept stems
    python ml/mine_cells.py read                          # stems -> notes on the grid
                                                          #   (downbeats: pip install beat-this)
    python ml/mine_cells.py build                         # riddim/theory/cells.json + report

`read` writes references/stats/cells.jsonl: per record, its tempo, which
beat is the one, and every bar's bass notes (step, length, pitch, how far off
the grid) and the other stem's onsets. `build` keeps, per riddim profile, the
bar rhythms that more than one record plays, weighted by how many records
play them - so one looped riddim can't fill the pool - and the numbers that
say what a family's bass does: how often each step is struck, how many
notes a bar, how long a note is held, the offbeat swing. The step shares are
what riddim draws from (theory/cells.py); exact bar rhythms repeat across
records too rarely to draw from alone. The rhythms are counts of grid
positions, not audio: the records stay study-only.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import library_stats as ls  # noqa: E402

CELLS_JSON = os.path.join(REPO, "riddim", "theory", "cells.json")
# Dub techno is a 4/4 at ~120 that FOR_PROFILES files under the half-time dub
# profile; read on that profile's grid its bars are the wrong length.
SKIP = {"dub_techno"}


def _target_bpm(family: str) -> float:
    from riddim import profile
    return float(profile.get(ls.FOR_PROFILES[family]).bpm)


def split(path: str, folder: str, seconds: float = 40.0) -> None:
    """htdemucs on the middle `seconds`; keeps bass, drums and other, mono at
    riddim.listen's rate."""
    from riddim import listen
    work = folder + "_work"
    os.makedirs(work, exist_ok=True)
    x = listen.load(path)
    dur = len(x) / listen.SR
    clip = os.path.join(work, "clip.wav")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{max(0.0, dur / 2 - seconds / 2):.2f}",
                    "-t", f"{seconds:.2f}", "-i", path, "-ac", "2", "-ar", "44100", clip], check=True)
    subprocess.run([sys.executable, "-m", "demucs", "-n", "htdemucs", "-o", work, clip],
                   check=True, capture_output=True)
    os.makedirs(folder, exist_ok=True)
    for stem in ("bass", "drums", "other"):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i",
                        os.path.join(work, "htdemucs", "clip", stem + ".wav"),
                        "-ac", "1", "-ar", str(listen.SR), os.path.join(folder, stem + ".wav")],
                       check=True)
    shutil.rmtree(work, ignore_errors=True)


def _mono(path: str) -> np.ndarray:
    from riddim import listen
    return listen.load(path).mean(axis=1).astype(float)


_TRACKER = []


def downbeats(mix: np.ndarray, sr: int) -> np.ndarray | None:
    """Downbeat times (s) from beat_this (CPJKU), a trained beat and downbeat
    tracker - independent of the bass, so the bar's one isn't chosen by where
    the bass happens to start. None if it isn't installed."""
    if not _TRACKER:
        try:
            from beat_this.inference import Audio2Beats
        except ImportError:
            return None
        import torch
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        _TRACKER.append(Audio2Beats(checkpoint_path="final0", device=dev, dbn=False))
    _, down = _TRACKER[0](mix.astype(np.float32), sr)
    return np.asarray(down, dtype=float)


def phase_from(beats: np.ndarray, down: np.ndarray, tol: float = 0.07) -> tuple[int | None, float]:
    """Which of the grid's first four beats the tracker's downbeats fall on,
    and how clearly: the best rotation's share of downbeats matched, less the
    runner-up's. A tracker at double the felt tempo marks every half bar, so
    two rotations tie and the answer is None."""
    if down is None or not len(down) or len(beats) < 8:
        return None, 0.0
    shares = []
    for r in range(4):
        downs = beats[r::4]
        shares.append(float(np.mean([np.min(np.abs(downs - d)) < tol for d in down])))
    order = np.argsort(shares)[::-1]
    margin = shares[order[0]] - shares[order[1]]
    return (int(order[0]) if margin >= 0.25 else None), round(margin, 2)


def read(folder: str, family: str, tracker: bool = True) -> dict:
    """One record's stems onto the grid."""
    from riddim import transcribe as T
    bass, drums, other = (_mono(os.path.join(folder, s + ".wav")) for s in ("bass", "drums", "other"))
    beats = T.beat_grid(drums, _target_bpm(family))
    if len(beats) < 9:
        return {"error": "no beat"}
    notes = T.bass_notes(bass)
    guess = T.bar_phase(beats, notes, drums)
    phase, margin = phase_from(beats, downbeats(bass + drums + other, T.SR) if tracker else None)
    if tracker and phase is None:
        return {"error": f"downbeat unclear (margin {margin})", "guess": guess}
    phase = guess if phase is None else phase
    bars = T.bars(notes, beats, phase)
    hits = [(t, 0.0, 0.0) for t in T.onset_times(other, 300, 3000)]
    stabs = [T.mask(b) for b in T.bars(hits, beats, phase)]
    offs = [n[3] for b in bars for n in b if n[0] % 2]
    return {"bpm": round(60.0 / float(np.median(np.diff(beats))), 2), "phase": phase,
            "guess": guess, "margin": margin, "bars": bars, "stabs": stabs,
            "swing": round(float(np.median(offs)), 3) if len(offs) >= 8 else None}


def cmd_split(a) -> int:
    try:
        import demucs  # noqa: F401
    except ImportError:
        sys.exit("split needs demucs: pip install demucs")
    plan = [p for p in ls._plan(os.path.join(a.out, "plan.jsonl")) if p["kind"] == "record"]
    by: dict = defaultdict(list)
    for p in plan:
        by[p["family"]].append(p)
    fams = set(a.families.split(",")) if a.families else set(by)
    pick = [p for fam in by if fam in fams for p in by[fam][: a.per_family]]
    todo = [p for p in pick if not os.path.exists(os.path.join(a.out, "_cells", p["id"], "other.wav"))]
    print(f"{len(todo)} of {len(pick)} records to split", flush=True)
    for i, p in enumerate(todo):
        try:
            split(os.path.join(a.refs, p["file"]), os.path.join(a.out, "_cells", p["id"]))
            print(f"  {i + 1}/{len(todo)} {p['family']} {p['id']}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  {i + 1}/{len(todo)} {p['family']} {p['id']} failed: {str(e)[:120]}", flush=True)
    return 0


def cmd_read(a) -> int:
    plan = {p["id"]: p for p in ls._plan(os.path.join(a.out, "plan.jsonl"))}
    path = os.path.join(a.out, "cells.jsonl")
    done = set() if a.redo else ls._done(path)
    if a.redo and os.path.exists(path):
        os.remove(path)
    root = os.path.join(a.out, "_cells")
    ids = sorted(i for i in os.listdir(root) if i in plan and i not in done) if os.path.isdir(root) else []
    print(f"{len(ids)} records to read -> {path}", flush=True)
    for i, rid in enumerate(ids):
        fam = plan[rid]["family"]
        try:
            rec = read(os.path.join(root, rid), fam, tracker=not a.no_tracker)
        except Exception as e:  # noqa: BLE001
            rec = {"error": str(e)[:200]}
        rec.update(id=rid, family=fam)
        ls._append(path, rec)
        n = len(rec.get("bars", []))
        print(f"  {i + 1}/{len(ids)} {fam} {rid} {rec.get('bpm')} bpm, {n} bars", flush=True)
    return 0


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------

def pool(records: list[dict], key, keep: int = 24, min_hits: int = 2, max_hits: int = 10):
    """The bar rhythms more than one record plays, by how many records play
    them (then by bars): [[mask, records], ...]."""
    recs, bars = Counter(), Counter()
    for r in records:
        seen = set()
        for m in key(r):
            if min_hits <= bin(m).count("1") <= max_hits:
                bars[m] += 1
                seen.add(m)
        recs.update(seen)
    ranked = sorted((m for m in recs if recs[m] >= 2), key=lambda m: (-recs[m], -bars[m], m))
    return [[m, recs[m]] for m in ranked[:keep]]


def summary(records: list[dict]) -> dict:
    bars = [b for r in records for b in r["bars"] if b]
    if not bars:
        return {}
    swing = [r["swing"] for r in records if r.get("swing") is not None]
    # Each record counts once: its bars' step and note-count shares, averaged
    # over records, so one long loop can't speak for the family.
    steps, counts = np.zeros(16), np.zeros(17)
    n = 0
    for r in records:
        rb = [b for b in r["bars"] if b]
        if not rb:
            continue
        n += 1
        steps += np.mean([[any(x[0] == k for x in b) for k in range(16)] for b in rb], axis=0)
        counts += np.bincount([min(16, len(b)) for b in rb], minlength=17) / len(rb)
    return {"records": len(records), "bars": len(bars),
            "steps": [round(float(v), 3) for v in steps / max(1, n)],
            "notes": [round(float(v), 3) for v in counts / max(1, n)],
            "notes_per_bar": round(float(np.median([len(b) for b in bars])), 2),
            "one": round(float(np.mean([any(n[0] == 0 for n in b) for b in bars])), 2),
            "and_of_4": round(float(np.mean([any(n[0] == 14 for n in b) for b in bars])), 2),
            "hold_steps": round(float(np.median([n[1] for b in bars for n in b])), 2),
            "swing": round(float(np.median(swing)), 3) if swing else None}


def build(records: list[dict]) -> dict:
    by: dict = defaultdict(list)
    for r in records:
        if r.get("bars") and ls.FOR_PROFILES.get(r["family"]) and r["family"] not in SKIP:
            by[ls.FOR_PROFILES[r["family"]]].append(r)
    from riddim import transcribe as T
    out = {}
    for prof, recs in sorted(by.items()):
        out[prof] = {"families": sorted({r["family"] for r in recs}),
                     "bass": pool(recs, lambda r: [T.mask(b) for b in r["bars"] if b]),
                     "stab": pool(recs, lambda r: r.get("stabs", []), min_hits=1, max_hits=8),
                     "stats": summary(recs)}
    return out


def report(cells: dict) -> str:
    from riddim import transcribe as T
    lines = ["# Mined rhythm cells", "",
             "Bar rhythms the records play, per riddim profile, by how many records play them.", ""]
    for prof, c in cells.items():
        s = c["stats"]
        lines += [f"## {prof} ({', '.join(c['families'])})", "",
                  f"{s.get('records')} records, {s.get('bars')} bars; notes a bar {s.get('notes_per_bar')}, "
                  f"the one {s.get('one')}, the & of 4 {s.get('and_of_4')}, held {s.get('hold_steps')} steps, "
                  f"swing {s.get('swing')}", "",
                  "Share of bars struck, step by step: " + " ".join(f"{v:.2f}" for v in s.get("steps", [])),
                  "", "| bass | records | other | records |", "|---|---|---|---|"]
        for i in range(min(8, max(len(c["bass"]), len(c["stab"])))):
            b = c["bass"][i] if i < len(c["bass"]) else None
            o = c["stab"][i] if i < len(c["stab"]) else None
            lines.append(f"| {'`' + T.mask_string(b[0]) + '`' if b else ''} | {b[1] if b else ''} | "
                         f"{'`' + T.mask_string(o[0]) + '`' if o else ''} | {o[1] if o else ''} |")
        lines.append("")
    return "\n".join(lines)


def cmd_build(a) -> int:
    records = [r for r in ls._load(os.path.join(a.out, "cells.jsonl")) if not r.get("error")]
    cells = build(records)
    with open(a.cells, "w") as f:
        json.dump(cells, f, indent=1)
        f.write("\n")
    text = report(cells)
    with open(os.path.join(a.out, "cells.md"), "w") as f:
        f.write(text + "\n")
    print(text)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", choices=("split", "read", "build"))
    ap.add_argument("--refs", default=ls.REFS)
    ap.add_argument("--out", default=ls.OUT)
    ap.add_argument("--per-family", type=int, default=40)
    ap.add_argument("--families", default="", help="split: only these, comma-separated")
    ap.add_argument("--cells", default=CELLS_JSON)
    ap.add_argument("--redo", action="store_true", help="read: start cells.jsonl again")
    ap.add_argument("--no-tracker", action="store_true",
                    help="read: pick the one from bass and kick (riddim.transcribe.bar_phase) "
                         "instead of beat_this's downbeats")
    a = ap.parse_args(argv)
    return {"split": cmd_split, "read": cmd_read, "build": cmd_build}[a.step](a)


if __name__ == "__main__":
    sys.exit(main())
