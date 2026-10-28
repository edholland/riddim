"""How a drummer plays it: timing, touch and how much each four bars repeat,
measured the same way on records and on riddim's renders.

    python ml/drum_feel.py --records roots_dub,roots_reggae      # 40 records
    python ml/drum_feel.py goals4/roots_dub_30?.wav              # renders
    python ml/drum_feel.py --records steppers_dub goals4/steppers_dub_30?.wav

Per track, on the middle 120 s of the full mix (no stems):

    repeat     the median Jaccard between consecutive four-bar onset grids:
               kick (35-120 Hz), snare (180-2500 Hz) and hat (6-10.5 kHz)
               onsets on the sixteenth grid, as riddim.listen finds them. 1 is
               the same four bars over and over; roots records sit near 0.54,
               steppers near 0.82.
    cv_*       the spread of onset strengths per band (std / mean): how much
               the touch varies from hit to hit.
    jitter_ms  riddim.listen's timing spread around the grid, and lean_ms its
               mean offset (negative is ahead of the beat).

Detection on full mixes is noisy, so compare medians over many tracks made the
same way, not single tracks. References are study-only: measured, never
rendered into riddim's output.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [REPO, HERE]

from riddim import listen  # noqa: E402

SR = listen.SR
FPS = SR / 256
BANDS = ((35, 120, {}), (180, 2500, {}), (6000, 10500, {"min_gap": 0.05}))


def _middle(path: str, seconds: float) -> np.ndarray:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                         capture_output=True, text=True).stdout.strip()
    start = max(0.0, float(out or 0) / 2 - seconds / 2)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{start:.1f}", "-t", f"{seconds}", "-i", path,
                          "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def feel(path: str, seconds: float = 120.0) -> dict | None:
    x = _middle(path, seconds)
    ons = [listen._onsets(listen._envelope(listen._band(x, lo, hi)), FPS, **kw) for lo, hi, kw in BANDS]
    bpm = listen._tempo(np.sort(np.concatenate([t for t, _ in ons])), length=len(x) / SR)
    if bpm <= 0 or len(ons[0][0]) < 8:
        return None
    six = 60.0 / bpm / 4
    # The grid's phase from the kicks, as riddim.listen fits it.
    phases = np.linspace(0, six, 48, endpoint=False)
    kicks = ons[0][0]
    phase = phases[int(np.argmin([np.mean(np.abs(((kicks - p + six / 2) % six) - six / 2)) for p in phases]))]
    bars: dict[int, set] = {}
    for band, (t, _) in enumerate(ons):
        for step in np.round((t - phase) / six).astype(int):
            bars.setdefault(int(step) // 16, set()).add((band, int(step) % 16))
    n = int((len(x) / SR - phase) / six) // 16
    chunks = [{(i, b, s) for i in range(4) for b, s in bars.get(c * 4 + i, ())} for c in range(n // 4)]
    jac = [len(a & b) / len(a | b) for a, b in zip(chunks, chunks[1:]) if a | b]
    cv = [float(np.std(s) / np.mean(s)) if len(s) > 8 else None for _, s in ons]
    rep = listen.analyze(path, seconds=None) if path.endswith(".wav") else None
    out = {"bpm": round(float(bpm), 1), "repeat": round(float(np.median(jac)), 3) if jac else None,
           "cv_kick": cv[0], "cv_snare": cv[1], "cv_hat": cv[2]}
    if rep is not None:
        out.update(jitter_ms=rep.jitter_ms, lean_ms=rep.lean_ms)
    return out


def records(families: list[str], n: int, refs: str) -> list[str]:
    """`n` records of the families from the library plan, without the ones
    the stats could not read."""
    stats = os.path.join(refs, "stats")
    plan = [json.loads(line) for line in open(os.path.join(stats, "plan.jsonl"))]
    bad = {json.loads(line)["id"] for line in open(os.path.join(stats, "stats.jsonl")) if '"error"' in line}
    files = [os.path.join(refs, p["file"]) for p in plan
             if p["kind"] == "record" and p["family"] in families and p["id"] not in bad]
    random.Random(5).shuffle(files)
    return files[:n]


def summary(name: str, rows: list[dict], stats: dict | None = None) -> str:
    def med(k):
        v = np.array([r[k] for r in rows if r.get(k) is not None], float)
        return f"{np.median(v):.2f} ({np.percentile(v, 25):.2f}-{np.percentile(v, 75):.2f})" if len(v) else "-"
    line = (f"{name}: n={len(rows)}  repeat {med('repeat')}  strength cv kick {med('cv_kick')} "
            f"snare {med('cv_snare')} hat {med('cv_hat')}")
    if any("jitter_ms" in r for r in rows):
        line += f"  jitter {med('jitter_ms')} ms  lean {med('lean_ms')} ms"
    if stats:
        line += f"  (stats.jsonl: jitter {stats['jitter_ms']:.1f} ms, lean {stats['lean_ms']:.1f} ms)"
    return line


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="*", help="renders (or any audio) to measure")
    ap.add_argument("--records", default="", help="families to sample references from, comma-separated")
    ap.add_argument("--n", type=int, default=40, help="records to measure")
    ap.add_argument("--refs", default=os.path.join(REPO, "references"))
    a = ap.parse_args(argv)
    if a.records:
        fams = a.records.split(",")
        rows = [r for r in (feel(f) for f in records(fams, a.n, a.refs)) if r]
        st = [json.loads(line) for line in open(os.path.join(a.refs, "stats", "stats.jsonl"))]
        st = [r for r in st if r.get("family") in fams and "error" not in r]
        print(summary("records " + a.records, rows,
                      {k: float(np.median([r[k] for r in st])) for k in ("jitter_ms", "lean_ms")}))
    if a.files:
        print(summary("files", [r for r in (feel(f) for f in a.files) if r]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
