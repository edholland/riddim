"""The critic against Ed's 19 picks: his 9 decided blind A/B pairs
(ab-blind/) and his 10 library-over-synth picks (ab/). It's the same test the
bake-off ran (ml/critic/bakeoff.py), where the judge's CLAP agreed on 11 and
AST's centroid on 13.

    python ml/critic/ed_agree.py

A pair agrees when the side with the higher mix log-odds is the side Ed
picked. The stem critics' scores are reported beside it.
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.dirname(HERE)
REPO = os.path.dirname(ML)
sys.path[:0] = [REPO, ML, HERE]

import bakeoff  # noqa: E402
import score  # noqa: E402


def main() -> int:
    pairs = bakeoff.ed_pairs()
    files = sorted({p for t in pairs for p in t[1:3]})
    sc = score.song_scores(files, os.path.join(REPO, "ab-blind", "_critic_cache"))
    lines = ["| source | profile | Ed's side mix / other side mix | agrees |", "|---|---|---|---|"]
    agree = {"blind": [0, 0], "library": [0, 0]}
    stem_agree = {s: [0, 0] for s in score.STEMS}
    for prof, mine, other, src in pairs:
        a, b = sc[mine], sc[other]
        ok = a["mix"] > b["mix"]
        agree[src][0] += ok
        agree[src][1] += 1
        for s in score.STEMS:
            if s in a and s in b:
                stem_agree[s][0] += a[s] > b[s]
                stem_agree[s][1] += 1
        lines.append(f"| {src} | {prof} | {a['mix']:+.2f} / {b['mix']:+.2f} | {'yes' if ok else 'no'} |")
    tot = agree["blind"][0] + agree["library"][0], agree["blind"][1] + agree["library"][1]
    lines += ["", f"Mix critic: blind {agree['blind'][0]}/{agree['blind'][1]}, library "
              f"{agree['library'][0]}/{agree['library'][1]}, total {tot[0]}/{tot[1]} "
              "(the judge's CLAP 11/19, AST's centroid 13/19).",
              "Stem critics alone: " + ", ".join(f"{s} {v[0]}/{v[1]}" for s, v in stem_agree.items()) + "."]
    out = os.path.join(REPO, "references", "critic", "ed_agree.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
