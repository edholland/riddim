"""Measure the reference library: what the records do, per family, so riddim
is tuned against ~11,000 tracks instead of the 439 the judge was built on.

Runs where the audio is (the Mac). Each step appends to a JSONL under
references/stats/ and skips what is already there, so it can be stopped and
re-run. Records and sets are kept apart: a two-hour radio show is not a
record, and its levels and lengths would skew the targets.

    python ml/library_stats.py plan   --per-family 300      # pick the tracks
    python ml/library_stats.py stats                         # levels, bands, tempo, sections
    python ml/library_stats.py stems  --per-family 50        # Demucs stem levels (needs demucs)
    python ml/library_stats.py embed  --per-family 300       # the judge's references (needs torch,
                                                             #   transformers; ml/judge.py)
    python ml/library_stats.py report                        # references/stats/report.md

What each step measures, per track:

    stats   loudness (RMS, crest, dynamic range), the five bands' share, stereo
            width, tempo and swing (riddim.listen), and the arrangement at bar
            resolution: section lengths (where the band balance moves), how
            often the bass is pulled (bars with the sub gone and the music
            still playing, per 32 bars), and the bar the sub first lands.
    stems   htdemucs on the middle 40 s: drums / bass / other / vocals, each
            in dB against the mix - the numbers riddim's mix is set against.
    embed   CLAP embeddings (ml/judge.py) per family into
            references/judge/<family>.npz, and each family's own baseline: the
            median closeness of its tracks to their centroid, and the FAD of
            one half against the other.

References are study-only: measured, never rendered into riddim's output.
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
sys.path.insert(0, REPO)

REFS = os.path.join(REPO, "references")
OUT = os.path.join(REFS, "stats")

# Family by tag, first match wins: the narrow scenes before the broad ones
# (a speed garage record is tagged uk_garage too).
FAMILIES = ("speed_garage", "dark_garage", "uk_garage", "grime", "dub_techno", "digital_dub",
            "steppers_dub", "roots_dub", "roots_reggae", "dubstep")
SET_TAGS = {"radio", "live", "mix", "soundsystem", "dj_set", "set"}
SET_SECONDS = 15 * 60

# Which riddim profiles each family is the reference for.
FOR_PROFILES = {"dubstep": "dubstep_140", "dark_garage": "dark_garage", "grime": "dark_garage",
                "uk_garage": "uk_garage", "speed_garage": "uk_garage",
                "roots_dub": "roots_dub", "roots_reggae": "roots_dub",
                "steppers_dub": "steppers_dub", "digital_dub": "steppers_dub",
                "dub_techno": "dubby_halftime"}


def family_of(row: dict) -> str | None:
    tags = set(row.get("tags", []))
    return next((f for f in FAMILIES if f in tags), None)


def kind_of(row: dict) -> str:
    """'set' for radio shows, live recordings and mixes; 'record' otherwise."""
    tags = set(row.get("tags", []))
    if tags & SET_TAGS or (row.get("duration") or 0) > SET_SECONDS:
        return "set"
    return "record"


def manifest(refs: str = REFS) -> list[dict]:
    rows = []
    with open(os.path.join(refs, "manifest.jsonl")) as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("silent") or not r.get("file"):
                continue
            if not os.path.exists(os.path.join(refs, r["file"])):
                continue
            r["family"], r["kind"] = family_of(r), kind_of(r)
            if r["family"]:
                rows.append(r)
    return rows


def _done(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    out = set()
    for line in open(path):
        try:
            out.add(json.loads(line)["id"])
        except (ValueError, KeyError):
            pass
    return out


def _append(path: str, rec: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def _plan(path: str) -> list[dict]:
    if not os.path.exists(path):
        sys.exit(f"no plan at {path}: run `plan` first")
    return [json.loads(line) for line in open(path)]


# --------------------------------------------------------------------------
# plan
# --------------------------------------------------------------------------

def cmd_plan(a) -> int:
    rows = manifest(a.refs)
    rng = random.Random(a.seed)
    os.makedirs(a.out, exist_ok=True)
    plan = []
    for fam in FAMILIES:
        for kind in ("record", "set"):
            pool = [r for r in rows if r["family"] == fam and r["kind"] == kind]
            rng.shuffle(pool)
            n = a.per_family if kind == "record" else a.sets_per_family
            plan += [{"id": r["id"], "file": r["file"], "family": fam, "kind": kind,
                      "source": r.get("source"), "title": r.get("title")} for r in pool[:n]]
    path = os.path.join(a.out, "plan.jsonl")
    with open(path, "w") as f:
        for p in plan:
            f.write(json.dumps(p) + "\n")
    counts: dict = {}
    for r in rows:
        counts.setdefault(r["family"], {"record": 0, "set": 0})[r["kind"]] += 1
    print(f"{len(rows)} usable tracks; plan of {len(plan)} -> {path}")
    for fam in FAMILIES:
        c = counts.get(fam, {"record": 0, "set": 0})
        print(f"  {fam:14s} records {c['record']:5d}  sets {c['set']:4d}")
    return 0


# --------------------------------------------------------------------------
# stats
# --------------------------------------------------------------------------

def arrangement(mono: np.ndarray, sr: int, bpm: float) -> dict:
    """The arrangement at bar resolution: section lengths, bass pulls, and
    where the sub first lands."""
    from riddim import listen
    if bpm <= 0:
        return {}
    bar = int(4 * 60.0 / bpm * sr)
    n = len(mono) // bar
    if n < 16:
        return {}
    bands = {}
    for name, (lo, hi) in (("sub", (25, 70)), ("bass", (70, 250)), ("mid", (250, 4000)),
                           ("high", (4000, 10000))):
        e = (listen._band(mono, lo, hi, sr)[: n * bar].reshape(n, bar) ** 2).mean(axis=1)
        db = 10 * np.log10(e + 1e-12)
        # Below 40 dB under its loud level a band is silence, whatever the
        # numbers do down there.
        bands[name] = np.maximum(db, np.percentile(db, 95) - 40)
    loud = 10 * np.log10((mono[: n * bar].reshape(n, bar) ** 2).mean(axis=1) + 1e-12)
    playing = loud > np.percentile(loud, 90) - 25
    sub_on = bands["sub"] > np.median(bands["sub"][playing]) - 12
    mid_on = bands["mid"] > np.median(bands["mid"][playing]) - 12
    # A pull: the sub gone for at least a bar while the music plays on.
    pulls = int(np.sum(np.diff(np.concatenate([[0], (~sub_on & mid_on & playing).astype(int)])) == 1))
    body = int(playing.sum())
    # Sections: on the four-bar grid, a move of 6 dB or more in any band
    # between one four-bar block and the next.
    m = n // 4
    blocks = np.stack([bands[k][: m * 4].reshape(m, 4).mean(axis=1) for k in bands], axis=1)
    edges = [0] + [i for i in range(1, m) if np.max(np.abs(blocks[i] - blocks[i - 1])) >= 6.0] + [m]
    lengths = [4 * (b - a) for a, b in zip(edges, edges[1:])]
    first_sub = int(np.argmax(sub_on & playing)) if (sub_on & playing).any() else -1
    return {"bars": int(n), "sections": lengths, "pulls_per_32": round(32.0 * pulls / max(1, body), 2),
            "sub_bars": round(float(sub_on[playing].mean()), 3), "first_sub_bar": first_sub}


def cmd_stats(a) -> int:
    from riddim import listen
    plan = [p for p in _plan(os.path.join(a.out, "plan.jsonl")) if p["kind"] == "record"]
    path = os.path.join(a.out, "stats.jsonl")
    done = _done(path)
    todo = [p for p in plan if p["id"] not in done]
    print(f"{len(todo)} of {len(plan)} records to measure -> {path}", flush=True)
    for i, p in enumerate(todo):
        f = os.path.join(a.refs, p["file"])
        try:
            rep = listen.analyze(f, seconds=a.max_seconds)
            x = listen.load(f, seconds=a.max_seconds).mean(axis=1)
            rec = {k: v for k, v in rep.__dict__.items() if k not in ("path", "energy_curve")}
            rec.update(arrangement(x, listen.SR, rep.bpm))
        except Exception as e:  # noqa: BLE001 - one bad file never stops the run
            rec = {"error": str(e)[:200]}
        rec.update(id=p["id"], family=p["family"], kind=p["kind"])
        _append(path, rec)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(todo)}", flush=True)
    return 0


# --------------------------------------------------------------------------
# stems
# --------------------------------------------------------------------------

def stem_levels(path: str, work: str, start: float | None = None, seconds: float = 40.0) -> dict:
    """htdemucs on `seconds` from the middle of the track: each stem's level
    in dB against the mix."""
    from riddim import listen
    os.makedirs(work, exist_ok=True)
    x = listen.load(path)
    dur = len(x) / listen.SR
    s = max(0.0, dur / 2 - seconds / 2) if start is None else start
    clip = os.path.join(work, "clip.wav")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{s:.2f}", "-t", f"{seconds:.2f}",
                    "-i", path, "-ac", "2", "-ar", "44100", clip], check=True)
    subprocess.run([sys.executable, "-m", "demucs", "-n", "htdemucs", "-o", work, clip],
                   check=True, capture_output=True)
    folder = os.path.join(work, "htdemucs", "clip")
    mix = listen.load(clip).mean(axis=1)
    ref = float(np.mean(mix ** 2)) + 1e-12
    out = {}
    for stem in ("drums", "bass", "other", "vocals"):
        y = listen.load(os.path.join(folder, stem + ".wav")).mean(axis=1)
        out[stem] = round(10 * np.log10((float(np.mean(y ** 2)) + 1e-12) / ref), 1)
    return out


def cmd_stems(a) -> int:
    try:
        import demucs  # noqa: F401
    except ImportError:
        sys.exit("stems needs demucs: pip install demucs")
    plan = [p for p in _plan(os.path.join(a.out, "plan.jsonl")) if p["kind"] == "record"]
    by: dict = {}
    for p in plan:
        by.setdefault(p["family"], []).append(p)
    pick = [p for fam in by for p in by[fam][: a.per_family]]
    path = os.path.join(a.out, "stems.jsonl")
    done = _done(path)
    todo = [p for p in pick if p["id"] not in done]
    print(f"{len(todo)} of {len(pick)} records to split -> {path}", flush=True)
    work = os.path.join(a.out, "_demucs")
    for i, p in enumerate(todo):
        try:
            rec = stem_levels(os.path.join(a.refs, p["file"]), work)
        except Exception as e:  # noqa: BLE001
            rec = {"error": str(e)[:200]}
        rec.update(id=p["id"], family=p["family"])
        _append(path, rec)
        print(f"  {i + 1}/{len(todo)} {p['family']} {rec}", flush=True)
    return 0


# --------------------------------------------------------------------------
# embed (the judge's references)
# --------------------------------------------------------------------------

def cmd_embed(a) -> int:
    sys.path.insert(0, HERE)
    import judge
    plan = [p for p in _plan(os.path.join(a.out, "plan.jsonl")) if p["kind"] == "record"]
    jdir = os.path.join(a.refs, "judge")
    os.makedirs(jdir, exist_ok=True)
    rows = []
    fams = sorted({p["family"] for p in plan})
    for fam in fams:
        files = [os.path.join(a.refs, p["file"]) for p in plan if p["family"] == fam][: a.per_family]
        cache = os.path.join(jdir, f"{fam}.npz")
        print(f"{fam}: embedding {len(files)} tracks -> {cache}", flush=True)
        emb = {k: v for k, v in judge.embed_many(files, cache).items() if len(v)}
        if len(emb) < 4:
            continue
        keys = sorted(emb)
        res = judge.score(emb, emb)
        close = [r["closeness"] for r in res["tracks"].values()]
        real = [r["real"] for r in res["tracks"].values()]
        half = judge.fad(np.concatenate([emb[k] for k in keys[::2]]),
                         np.concatenate([emb[k] for k in keys[1::2]]))
        rows.append({"family": fam, "tracks": len(emb), "median_closeness": round(float(np.median(close)), 3),
                     "median_real": round(float(np.median(real)), 3), "half_fad": round(half, 3)})
        print(f"  {rows[-1]}", flush=True)
    with open(os.path.join(jdir, "baselines.json"), "w") as f:
        json.dump(rows, f, indent=1)
    return 0


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def _load(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    return [r for r in (json.loads(line) for line in open(path)) if "error" not in r]


def _med(xs) -> str:
    xs = [x for x in xs if x is not None]
    if not xs:
        return "-"
    q1, m, q3 = np.percentile(xs, [25, 50, 75])
    return f"{m:.2f} ({q1:.2f} to {q3:.2f})" if abs(m) < 10 else f"{m:.1f} ({q1:.1f} to {q3:.1f})"


def summarise(stats: list[dict], stems: list[dict]) -> str:
    out = ["# Reference library, measured", "",
           "Medians per family, interquartile range in brackets. Records only; sets are measured "
           "separately (step 4).", "",
           "## Levels and bands", "",
           "| family | n | RMS dB | crest dB | dyn range | sub | bass | low mid | high mid | air | bpm |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    fams = [f for f in FAMILIES if any(r["family"] == f for r in stats)]
    for fam in fams:
        rs = [r for r in stats if r["family"] == fam]
        band = lambda k: _med([r.get("band_share", {}).get(k) for r in rs])  # noqa: E731
        out.append(f"| {fam} | {len(rs)} | {_med([r.get('rms_db') for r in rs])} | "
                   f"{_med([r.get('crest_db') for r in rs])} | {_med([r.get('dynamic_range_db') for r in rs])} | "
                   f"{band('sub')} | {band('bass')} | {band('low_mid')} | {band('high_mid')} | {band('air')} | "
                   f"{_med([r.get('bpm') for r in rs])} |")
    out += ["", "## Arrangement", "",
            "| family | length (bars) | section length (bars) | bass pulls / 32 bars | sub plays (share of bars) | first sub bar | swing |",
            "|---|---|---|---|---|---|---|"]
    for fam in fams:
        rs = [r for r in stats if r["family"] == fam and r.get("bars")]
        secs = [s for r in rs for s in r.get("sections", [])]
        out.append(f"| {fam} | {_med([r['bars'] for r in rs])} | {_med(secs)} | "
                   f"{_med([r.get('pulls_per_32') for r in rs])} | {_med([r.get('sub_bars') for r in rs])} | "
                   f"{_med([r.get('first_sub_bar') for r in rs])} | {_med([r.get('swing') for r in rs])} |")
    if stems:
        out += ["", "## Stems (dB against the mix, htdemucs, middle 40 s)", "",
                "| family | n | drums | bass | other | vocals |", "|---|---|---|---|---|---|"]
        for fam in FAMILIES:
            rs = [r for r in stems if r["family"] == fam]
            if rs:
                out.append(f"| {fam} | {len(rs)} | " + " | ".join(_med([r.get(s) for r in rs])
                                                             for s in ("drums", "bass", "other", "vocals")) + " |")
    out += ["", "Which riddim profile each family is the reference for: "
            + ", ".join(f"{f} -> {p}" for f, p in FOR_PROFILES.items()) + "."]
    return "\n".join(out) + "\n"


def cmd_report(a) -> int:
    text = summarise(_load(os.path.join(a.out, "stats.jsonl")), _load(os.path.join(a.out, "stems.jsonl")))
    base = os.path.join(a.refs, "judge", "baselines.json")
    if os.path.exists(base):
        text += "\n## The judge's references\n\n| family | tracks | median closeness | median real | half-vs-half FAD |\n|---|---|---|---|---|\n"
        for r in json.load(open(base)):
            text += (f"| {r['family']} | {r['tracks']} | {r['median_closeness']} | {r['median_real']} | "
                     f"{r['half_fad']} |\n")
    path = os.path.join(a.out, "report.md")
    with open(path, "w") as f:
        f.write(text)
    print(text)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", choices=("plan", "stats", "stems", "embed", "report"))
    ap.add_argument("--refs", default=REFS)
    ap.add_argument("--out", default=None, help="default references/stats")
    ap.add_argument("--per-family", type=int, default=None)
    ap.add_argument("--sets-per-family", type=int, default=40)
    ap.add_argument("--max-seconds", type=float, default=600.0,
                    help="measure at most this much of each record")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    a.out = a.out or os.path.join(a.refs, "stats")
    if a.per_family is None:
        a.per_family = {"plan": 300, "stems": 50, "embed": 300}.get(a.step, 300)
    return {"plan": cmd_plan, "stats": cmd_stats, "stems": cmd_stems, "embed": cmd_embed,
            "report": cmd_report}[a.step](a)


if __name__ == "__main__":
    sys.exit(main())
