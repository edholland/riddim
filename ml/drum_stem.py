"""The drums' character, measured on Demucs drum stems: records against
riddim's renders split the same way (mine_cells.split).

    python ml/drum_stem.py records                # the records' medians per family (852 cell splits)
    python ml/drum_stem.py ours critic_head       # a render folder, split and measured

Per drum stem, with everything under 60 Hz left out (Demucs files a 140
sub under drums, so the sub is no part of the kit's character here):
- **kick / snare / hat share**: energy in 60-150 Hz, 150-2500 Hz and
  2.5-11 kHz, as shares of the three
- **hat brightness**: the spectral centroid of the 2.5-11 kHz band, Hz
- **hat decay**: median time (ms) for a hat onset's envelope to fall 20 dB
- **snare body**: at snare onsets, 150-400 Hz energy against 2-8 kHz, dB

and per bass stem (bass_metrics): energy shares under 60 Hz, 60-120,
120-300 and 300-2000, how much of the time it sounds, and the spread of its
note-onset strengths.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [REPO, HERE]

from riddim import listen  # noqa: E402

KEYS = ("kick_share", "snare_share", "hat_share", "hat_centroid_hz", "hat_decay_ms", "snare_body_db")


def metrics(path: str) -> dict[str, float]:
    sr = listen.SR
    x = listen.load(path).mean(axis=1)
    x = listen._band(x, 60, sr * 0.45, sr)
    e = {name: float(np.mean(listen._band(x, lo, hi, sr) ** 2))
         for name, (lo, hi) in (("kick", (60, 150)), ("snare", (150, 2500)), ("hat", (2500, 11000)))}
    tot = sum(e.values()) or 1.0
    out = {f"{k}_share": v / tot for k, v in e.items()}
    hat = listen._band(x, 2500, 11000, sr)
    spec = np.abs(np.fft.rfft(hat[: min(len(hat), sr * 30)]))
    freqs = np.fft.rfftfreq(min(len(hat), sr * 30), 1 / sr)
    band = (freqs >= 2500) & (freqs <= 11000)
    out["hat_centroid_hz"] = float((spec[band] * freqs[band]).sum() / (spec[band].sum() or 1.0))
    hop = 128
    fps = sr / hop
    env = listen._envelope(hat, sr, hop)
    on, _ = listen._onsets(env, fps, min_gap=0.05)
    decays = []
    for t in on[:400]:
        i = int(t * fps)
        seg = env[i:i + int(0.4 * fps)]
        if len(seg) < 4:
            continue
        peak = seg[:3].max()
        below = np.where(seg < peak * 0.1)[0]
        if len(below):
            decays.append(below[0] / fps * 1000)
    out["hat_decay_ms"] = float(np.median(decays)) if decays else float("nan")
    body = listen._band(x, 150, 400, sr)
    crack = listen._band(x, 2000, 8000, sr)
    s_env = listen._envelope(listen._band(x, 180, 2500, sr), sr, hop)
    s_on, _ = listen._onsets(s_env, fps)
    ratios = []
    for t in s_on[:300]:
        i = int(t * sr)
        w = slice(i, i + int(0.08 * sr))
        b, c = float(np.mean(body[w] ** 2)), float(np.mean(crack[w] ** 2))
        if b > 0 and c > 0:
            ratios.append(10 * np.log10(b / c))
    out["snare_body_db"] = float(np.median(ratios)) if ratios else float("nan")
    return out


BASS_KEYS = ("bass_sub", "bass_low", "bass_mid", "bass_upper", "bass_held", "bass_cv")


def bass_metrics(path: str) -> dict[str, float]:
    """The bass stem's character: energy shares under 60 Hz, 60-120, 120-300
    and 300-2000; the share of 50 ms frames sounding within 20 dB of the
    stem's loud level (how long the bass holds); and the spread (std/mean)
    of its note onsets' strengths."""
    sr = listen.SR
    x = listen.load(path).mean(axis=1)
    e = {name: float(np.mean(listen._band(x, lo, hi, sr) ** 2))
         for name, (lo, hi) in (("sub", (25, 60)), ("low", (60, 120)), ("mid", (120, 300)), ("upper", (300, 2000)))}
    tot = sum(e.values()) or 1.0
    out = {f"bass_{k}": v / tot for k, v in e.items()}
    hop = int(0.05 * sr)
    env = listen._envelope(x, sr, hop)
    db = 20 * np.log10(env + 1e-9)
    out["bass_held"] = float(np.mean(db > np.percentile(db, 95) - 20))
    on, strength = listen._onsets(listen._envelope(listen._band(x, 30, 300, sr), sr, 256), sr / 256)
    out["bass_cv"] = float(np.std(strength) / np.mean(strength)) if len(strength) > 8 else float("nan")
    return out


OTHER_KEYS = ("other_lowmid", "other_mid", "other_high", "other_centroid_hz", "other_onsets_s", "other_held")


def other_metrics(path: str) -> dict[str, float]:
    """The "other" stem (keys, guitars, pads, leads, effects): energy shares
    in 150-500 Hz, 500-2500 and 2.5-10 kHz; its spectral centroid; onsets a
    second (how busy); and the share of 50 ms frames sounding within 20 dB
    of its loud level."""
    sr = listen.SR
    x = listen.load(path).mean(axis=1)
    e = {name: float(np.mean(listen._band(x, lo, hi, sr) ** 2))
         for name, (lo, hi) in (("lowmid", (150, 500)), ("mid", (500, 2500)), ("high", (2500, 10000)))}
    tot = sum(e.values()) or 1.0
    out = {f"other_{k}": v / tot for k, v in e.items()}
    n = min(len(x), sr * 30)
    spec = np.abs(np.fft.rfft(x[:n]))
    freqs = np.fft.rfftfreq(n, 1 / sr)
    band = (freqs >= 100) & (freqs <= 10000)
    out["other_centroid_hz"] = float((spec[band] * freqs[band]).sum() / (spec[band].sum() or 1.0))
    on, _ = listen._onsets(listen._envelope(listen._band(x, 200, 6000, sr), sr, 256), sr / 256)
    out["other_onsets_s"] = len(on) / (len(x) / sr)
    env = listen._envelope(x, sr, int(0.05 * sr))
    db = 20 * np.log10(env + 1e-9)
    out["other_held"] = float(np.mean(db > np.percentile(db, 95) - 20))
    return out


def records() -> dict[str, dict[str, float]]:
    cells = os.path.join(REPO, "references", "stats", "_cells")
    fam = {json.loads(line)["id"]: json.loads(line)["family"]
           for line in open(os.path.join(REPO, "references", "stats", "plan.jsonl"))}
    by = {}
    for rid in sorted(os.listdir(cells)):
        f = os.path.join(cells, rid, "drums.wav")
        b = os.path.join(cells, rid, "bass.wav")
        o = os.path.join(cells, rid, "other.wav")
        if rid in fam and os.path.exists(f) and os.path.exists(b) and os.path.exists(o):
            by.setdefault(fam[rid], []).append({**metrics(f), **bass_metrics(b), **other_metrics(o)})
    return {f: {k: float(np.nanmedian([m[k] for m in ms])) for k in KEYS + BASS_KEYS + OTHER_KEYS} | {"n": len(ms)}
            for f, ms in by.items()}


def ours(folder: str) -> dict[str, dict[str, float]]:
    """Each render in `folder` split as the records were and measured;
    medians per profile."""
    import mine_cells
    by = {}
    tmp = tempfile.mkdtemp(prefix="drumstem_")
    try:
        for f in sorted(glob.glob(os.path.join(folder, "*.wav"))):
            prof = os.path.basename(f).rsplit("_", 1)[0]
            cache = f[:-4] + ".stems3.json"
            if os.path.exists(cache):
                m = json.load(open(cache))
            else:
                mine_cells.split(f, os.path.join(tmp, "s"))
                m = {**metrics(os.path.join(tmp, "s", "drums.wav")), **bass_metrics(os.path.join(tmp, "s", "bass.wav")),
                     **other_metrics(os.path.join(tmp, "s", "other.wav"))}
                shutil.rmtree(os.path.join(tmp, "s"), ignore_errors=True)
                json.dump(m, open(cache, "w"))
            by.setdefault(prof, []).append(m)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return {p: {k: float(np.nanmedian([m[k] for m in ms])) for k in KEYS + BASS_KEYS + OTHER_KEYS} | {"n": len(ms)}
            for p, ms in by.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", choices=("records", "ours"))
    ap.add_argument("folder", nargs="?")
    a = ap.parse_args(argv)
    res = records() if a.step == "records" else ours(a.folder)
    out = os.path.join(REPO, "references", "stats", "drum_stem_records.json") if a.step == "records" else \
        os.path.join(a.folder, "drum_stem.json")
    json.dump(res, open(out, "w"), indent=1)
    keys = [k for k in KEYS + BASS_KEYS + OTHER_KEYS if all(k in v for v in res.values())]
    print("| set | n | " + " | ".join(keys) + " |")
    for k, v in res.items():
        print(f"| {k} | {v['n']} | " + " | ".join(f"{v[x]:.3f}" for x in keys) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
