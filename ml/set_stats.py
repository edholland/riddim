"""Measure the sets: how a DJ or a sound system runs an hour, per family, so
riddim's live sets are shaped against the tapes and not only the records.

Runs where the audio is (the Mac), on the sets library_stats.py planned
(kind "set" in references/stats/plan.jsonl: radio shows, DJ mixes, sound
system tapes). Each set appends a line to references/stats/sets.jsonl and is
skipped when re-run, so it can be stopped and resumed.

    python ml/set_stats.py run    [--per-family 40] [--minutes 60]
    python ml/set_stats.py report                     # references/stats/sets.md

What it measures, on the first hour of each set:

    mix points  where one tune hands over to the next: the CLAP embedding
                (ml/judge.py) of every 10 s window, and the cosine distance
                between the 30 s before and the 30 s after each window edge.
                A peak that stands over the set's own novelty (median + 2.5
                MAD) and is the largest within a minute is a mix point.
                Calibrated on mixes made from the records (15 tunes a family,
                2-5 minutes of each, 16 s crossfades): 19 of 23 mix points
                found within 30 s, 7 false ones - most in roots dub, where a
                version dropping its parts in and out reads as a new tune.
                Talk, a rewind or a dub plate's version side can also make
                one, so tune lengths are an upper bound on how often the
                selector changes something, not a tracklist.
    tempo       per ten minutes, from the onsets (riddim.listen), folded to
                the felt tempo for the half-time families as the record stats
                are (library_stats.fold).
    bass pulls  per 32 bars, as library_stats.arrangement counts them for
                records, on ten-minute chunks.
    energy      RMS per minute in dB against the set's median minute: how an
                hour rises and falls.

References are study-only: measured, never rendered into riddim's output.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [REPO, HERE]

import library_stats as ls  # noqa: E402

REFS = ls.REFS
OUT = ls.OUT
WIN = 10.0          # seconds per CLAP window
SIDE = 3            # windows either side of an edge: 30 s
MIN_GAP = 6         # windows: mix points at least a minute apart
CHUNK = 600.0       # seconds per tempo / bass-pull chunk
MIN_SET = 20 * 60   # anything shorter is not a set


def _decode(path: str, sr: int, seconds: float) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-t", f"{seconds:.0f}", "-f", "s16le",
                          "-ac", "1", "-ar", str(sr), "-"], check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.int16)


def embed_windows(x16: np.ndarray, batch: int = 16) -> np.ndarray:
    """CLAP embeddings of consecutive 10 s windows (48 kHz int16 in)."""
    import torch
    import judge
    model, proc = judge._load_model()
    n = int(WIN * judge.SR)
    starts = range(0, len(x16) - n + 1, n)
    out = []
    ws = [x16[s:s + n].astype(np.float32) / 32768.0 for s in starts]
    for i in range(0, len(ws), batch):
        inp = proc(audio=ws[i:i + batch], sampling_rate=judge.SR, return_tensors="pt")
        with torch.no_grad():
            e = model.get_audio_features(**inp)
        out.append(getattr(e, "pooler_output", e).numpy())
    return np.concatenate(out) if out else np.zeros((0, 512), dtype=np.float32)


def novelty(emb: np.ndarray, side: int = SIDE) -> np.ndarray:
    """Cosine distance between the mean of the `side` windows before each
    edge and the `side` after it (index i = the edge before window i)."""
    e = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    nov = np.zeros(len(e))
    for i in range(side, len(e) - side + 1):
        a, b = e[i - side:i].mean(0), e[i:i + side].mean(0)
        nov[i] = 1.0 - float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    return nov


def mix_points(nov: np.ndarray, k: float = 2.5, gap: int = MIN_GAP) -> list[int]:
    """Edges whose novelty stands k MADs over the set's median and is the
    largest within `gap` windows either side."""
    body = nov[SIDE:len(nov) - SIDE + 1]
    if len(body) < 2 * gap:
        return []
    med = float(np.median(body))
    mad = float(np.median(np.abs(body - med))) or 1e-6
    out = []
    for i in range(SIDE, len(nov) - SIDE + 1):
        lo, hi = max(0, i - gap), min(len(nov), i + gap + 1)
        if nov[i] >= med + k * mad and nov[i] == nov[lo:hi].max():
            out.append(i)
    return out


def tempo(mono: np.ndarray, sr: int) -> float:
    """Tempo from the onsets of a stretch of audio, as riddim.listen reads it."""
    from riddim import listen
    fps = sr / 256
    ons = [listen._onsets(listen._envelope(listen._band(mono, lo, hi, sr), sr=sr), fps, **kw)[0]
           for lo, hi, kw in ((35, 120, {}), (180, 2500, {}), (6000, 10500, {"min_gap": 0.05}))]
    allo = np.sort(np.concatenate(ons))
    bpm = listen._tempo(allo, length=len(mono) / sr)
    if 60 <= bpm < 95 and len(ons[2]) / max(1.0, len(mono) / sr) > 3.5:
        bpm *= 2
    return float(bpm)


def measure(path: str, family: str, minutes: float) -> dict:
    import judge
    from riddim import listen
    x48 = _decode(path, judge.SR, minutes * 60)
    secs = len(x48) / judge.SR
    if secs < MIN_SET:
        return {"error": f"only {secs / 60:.1f} min - not a set"}
    emb = embed_windows(x48)
    del x48
    nov = novelty(emb)
    mp = mix_points(nov)
    edges = [0] + mp + [len(emb)]
    tunes = [WIN * (b - a) for a, b in zip(edges, edges[1:])]
    # The first and last stretches are cut by the hour, not by a mix.
    inner = tunes[1:-1]

    sr = listen.SR
    mono = _decode(path, sr, minutes * 60).astype(np.float32) / 32768.0
    c = int(CHUNK * sr)
    bpms, pulls, subs, weights = [], [], [], []
    for s in range(0, len(mono) - c // 2, c):
        x = mono[s:s + c]
        bpm = tempo(x[: 90 * sr], sr)
        if family in ls.HALF_TIME and bpm > 110:
            bpm /= 2
        if bpm <= 0:
            continue
        bpms.append(bpm)
        arr = ls.arrangement(x, sr, bpm)
        if arr:
            pulls.append(arr["pulls_per_32"])
            subs.append(arr["sub_bars"])
            weights.append(arr["bars"])
    m = int(60 * sr)
    per_min = np.array([10 * np.log10(np.mean(mono[i:i + m] ** 2) + 1e-12)
                        for i in range(0, len(mono) - m + 1, m)])
    energy = per_min - np.median(per_min)
    bpm = float(np.median(bpms)) if bpms else 0.0
    w = np.array(weights, float)
    return {"minutes": round(secs / 60, 1), "bpm": round(bpm, 1),
            "mix_points_s": [int(WIN * i) for i in mp],
            "tunes_s": [round(t) for t in inner],
            "tune_bars": [round(t * bpm / 240.0, 1) for t in inner] if bpm else [],
            "novelty_median": round(float(np.median(nov[SIDE:len(nov) - SIDE + 1])), 4),
            "pulls_per_32": round(float(np.average(pulls, weights=w)), 2) if pulls else None,
            "sub_bars": round(float(np.average(subs, weights=w)), 3) if subs else None,
            "energy_db": [round(float(v), 1) for v in energy]}


def cmd_run(a) -> int:
    plan = [p for p in ls._plan(os.path.join(a.out, "plan.jsonl")) if p["kind"] == "set"]
    by: dict[str, list] = {}
    for p in plan:
        by.setdefault(p["family"], []).append(p)
    plan = [p for f in ls.FAMILIES for p in by.get(f, [])[: a.per_family]]
    path = os.path.join(a.out, "sets.jsonl")
    done = ls._done(path)
    todo = [p for p in plan if p["id"] not in done]
    print(f"{len(todo)} of {len(plan)} sets to measure -> {path}", flush=True)
    for i, p in enumerate(todo):
        try:
            rec = measure(os.path.join(a.refs, p["file"]), p["family"], a.minutes)
        except Exception as e:  # noqa: BLE001 - one bad file never stops the run
            rec = {"error": str(e)[:200]}
        rec.update(id=p["id"], family=p["family"], title=p.get("title", "")[:80])
        ls._append(path, rec)
        brief = rec.get("error") or (f"{rec['minutes']} min, {rec['bpm']} bpm, {len(rec['mix_points_s'])} mix "
                                     f"points, pulls {rec['pulls_per_32']}")
        print(f"  {i + 1}/{len(todo)} {p['family']} {p['id']}: {brief}", flush=True)
    return 0


def _curve(rs: list[dict], step: int = 5) -> str:
    """The family's median energy, minute by minute, in `step`-minute steps."""
    n = max((len(r["energy_db"]) for r in rs), default=0)
    cols = []
    for m in range(0, n, step):
        vals = [np.mean(r["energy_db"][m:m + step]) for r in rs if len(r["energy_db"]) > m]
        if len(vals) >= max(3, len(rs) // 2):
            cols.append(f"{np.median(vals):+.1f}")
    return " ".join(cols)


def summarise(sets: list[dict], stats: list[dict]) -> str:
    stats = [ls.fold(r) for r in stats]
    out = ["# The sets, measured", "",
           "The first hour of each set (radio shows, DJ mixes, sound-system tapes). Medians per family, "
           "interquartile range in brackets. Tune lengths are the stretches between mix points found "
           "from CLAP novelty; the first and last stretch of each hour are left out. Half-time families "
           "are read at their felt tempo, as the record stats are.", "",
           "## Tunes and mixing", "",
           "| family | sets | minutes | bpm | mix points / hour | tune length (min) | tune length (bars) |",
           "|---|---|---|---|---|---|---|"]
    fams = [f for f in ls.FAMILIES if any(r["family"] == f for r in sets)]
    for fam in fams:
        rs = [r for r in sets if r["family"] == fam]
        per_hour = [60.0 * len(r["mix_points_s"]) / r["minutes"] for r in rs]
        tunes = [t / 60.0 for r in rs for t in r["tunes_s"]]
        bars = [b for r in rs for b in r["tune_bars"]]
        out.append(f"| {fam} | {len(rs)} | {ls._med([r['minutes'] for r in rs])} | "
                   f"{ls._med([r['bpm'] for r in rs if r['bpm']])} | {ls._med(per_hour)} | "
                   f"{ls._med(tunes)} | {ls._med(bars)} |")
    out += ["", "## Bass pulls (per 32 bars)", "",
            "| family | sets | records | sub plays, sets (share of bars) | sub plays, records |",
            "|---|---|---|---|---|"]
    for fam in fams:
        rs = [r for r in sets if r["family"] == fam]
        rec = [r for r in stats if r["family"] == fam and r.get("bars")]
        out.append(f"| {fam} | {ls._med([r.get('pulls_per_32') for r in rs])} | "
                   f"{ls._med([r.get('pulls_per_32') for r in rec])} | "
                   f"{ls._med([r.get('sub_bars') for r in rs])} | {ls._med([r.get('sub_bars') for r in rec])} |")
    out += ["", "## Energy over the hour", "",
            "RMS per minute in dB against each set's median minute; the family's median, five minutes "
            "a step (0-5, 5-10, ...). Spread: the 90th less the 10th percentile minute.", "",
            "| family | spread (dB) | median curve, 5 min steps |", "|---|---|---|"]
    for fam in fams:
        rs = [r for r in sets if r["family"] == fam]
        spread = [float(np.percentile(r["energy_db"], 90) - np.percentile(r["energy_db"], 10)) for r in rs]
        out.append(f"| {fam} | {ls._med(spread)} | {_curve(rs)} |")
    out += ["", "Which riddim profile each family is the reference for: "
            + ", ".join(f"{f} -> {p}" for f, p in ls.FOR_PROFILES.items()) + "."]
    return "\n".join(out) + "\n"


def cmd_report(a) -> int:
    text = summarise(ls._load(os.path.join(a.out, "sets.jsonl")), ls._load(os.path.join(a.out, "stats.jsonl")))
    path = os.path.join(a.out, "sets.md")
    with open(path, "w") as f:
        f.write(text)
    print(text)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", choices=("run", "report"))
    ap.add_argument("--refs", default=REFS)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--per-family", type=int, default=40)
    ap.add_argument("--minutes", type=float, default=60.0, help="measure at most this much of each set")
    a = ap.parse_args(argv)
    return {"run": cmd_run, "report": cmd_report}[a.step](a)


if __name__ == "__main__":
    sys.exit(main())
