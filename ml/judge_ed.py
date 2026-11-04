"""Ed's picks as judge test material: measure both sides of every pair he
judged through riddim/ears.py, for ml/judge_fit.py --ed.

    python ml/judge_ed.py --blind ab-blind --library ab --out ed_feats.json

--blind: a blind A/B folder (pairs.json, picks.jsonl, clips/) from ml/blind_ab.py;
"unsure" picks are kept as can't-tell pairs. --library: the sample library
A/B folder (<song>_library.* and <song>_synth.*), where Ed picked the library
side every time. Audio may be .wav, .flac or .opus (same stem name).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from riddim import ears  # noqa: E402


def _find(folder, base):
    stem = os.path.splitext(base)[0]
    for ext in (".wav", ".flac", ".opus", ".mp3"):
        for d in (folder, os.path.join(folder, "clips")):
            p = os.path.join(d, stem + ext)
            if os.path.exists(p):
                return p
    raise FileNotFoundError(f"{stem} in {folder}")


def pairs(blind=None, library=None):
    out = []
    if blind:
        meta = {m["id"]: m for m in json.load(open(os.path.join(blind, "pairs.json")))}
        last = {}
        for line in open(os.path.join(blind, "picks.jsonl")):
            r = json.loads(line)
            last[r["id"]] = r
        for i, m in sorted(meta.items()):
            if i not in last:
                continue
            k, o = os.path.basename(m["files"]["kept"]), os.path.basename(m["files"]["other"])
            pk = last[i]
            ed = None if pk["pick"] == "unsure" else (
                "kept" if os.path.splitext(os.path.basename(pk["clip"]))[0] == os.path.splitext(k)[0] else "other")
            out.append(dict(src="blind", id=i, profile=m["profile"], change=m.get("change", m.get("item", "")),
                            a=_find(blind, k), b=_find(blind, o), ed=ed))
    if library:
        songs = sorted({os.path.basename(f).rsplit("_", 1)[0] for f in glob.glob(os.path.join(library, "*_library.*"))})
        for s in songs:
            out.append(dict(src="library", id=s, profile=s.rsplit("_", 1)[0], change="sample library vs synths",
                            a=_find(library, s + "_library"), b=_find(library, s + "_synth"), ed="kept"))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blind")
    ap.add_argument("--library")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = []
    for p in pairs(a.blind, a.library):
        xa, xb = ears.load(p["a"]), ears.load(p["b"])
        n = min(len(xa), len(xb))
        p["wa"], p["wb"] = ears.windows(xa[:n]).tolist(), ears.windows(xb[:n]).tolist()
        print(p["src"], p["id"], p["profile"], len(p["wa"]), p["ed"], flush=True)
        res.append(p)
    json.dump(res, open(a.out, "w"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
