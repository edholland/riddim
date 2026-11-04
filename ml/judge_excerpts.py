"""Cut the judge's reference excerpts: two 20 s excerpts (at 35% and 65% of
the way through) of up to N records a family, at most 4 a family per artist,
from references/stats/plan.jsonl (ml/library_stats.py plan). Writes Opus
excerpts to references/judge2/refs/ and their list to references/judge2/refs.json,
the --meta and --audio for ml/judge_data.py.

    python ml/judge_excerpts.py --per-family 120
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-family", type=int, default=120)
    ap.add_argument("--refs", default=os.path.join(REPO, "references"))
    ap.add_argument("--out", default=None, help="default: <refs>/judge2")
    a = ap.parse_args(argv)
    R = a.refs
    out_dir = a.out or os.path.join(R, "judge2")
    audio = os.path.join(out_dir, "refs")
    os.makedirs(audio, exist_ok=True)
    man = {}
    for line in open(os.path.join(R, "manifest.jsonl")):
        try:
            m = json.loads(line)
            man[m["id"]] = m
        except Exception:
            pass
    rows = [json.loads(line) for line in open(os.path.join(R, "stats", "plan.jsonl"))]
    rng = random.Random(7)
    by = collections.defaultdict(list)
    for r in rows:
        if r["kind"] == "record":
            by[r["family"]].append(r)
    pick = []
    for fam, rs in by.items():
        rng.shuffle(rs)
        seen, chosen = collections.Counter(), []
        for r in rs:
            art = man.get(r["id"], {}).get("artist") or r.get("title", "")
            art = (", ".join(art) if isinstance(art, list) else str(art)).strip().lower()
            if seen[art] >= 4:
                continue
            seen[art] += 1
            chosen.append((r, art))
            if len(chosen) >= a.per_family:
                break
        pick += [(fam, r, art) for r, art in chosen]

    def dur(p):
        o = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p],
                           capture_output=True, text=True).stdout
        try:
            return float(o.strip())
        except ValueError:
            return 0.0

    def job(t):
        fam, r, art = t
        src = os.path.join(R, r["file"])
        d = dur(src)
        if d < 60:
            return None
        outs = []
        for k, frac in enumerate((0.35, 0.65)):
            o = os.path.join(audio, f"{r['id']}_{k}.opus")
            if not os.path.exists(o):
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{d * frac:.2f}", "-t", "20", "-i", src,
                                "-ac", "2", "-ar", "48000", "-c:a", "libopus", "-b:a", "160k", o], check=False)
            if os.path.exists(o):
                outs.append(os.path.basename(o))
        return {"id": r["id"], "family": fam, "artist": art, "title": r.get("title"), "excerpts": outs, "dur": d}

    with ThreadPoolExecutor(os.cpu_count() or 4) as ex:
        res = [x for x in ex.map(job, pick) if x]
    json.dump(res, open(os.path.join(out_dir, "refs.json"), "w"), indent=0)
    print(len(res), dict(collections.Counter(x["family"] for x in res)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
