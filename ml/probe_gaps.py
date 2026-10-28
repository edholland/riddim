"""What the records have that riddim's songs don't, in words: score a battery
of descriptions ("shuffled 2-step hi-hats", "a live drummer", "a sung vocal
hook" ...) against each family's references and against the goal panel's
renders with the judge's CLAP model, and list where ours sit furthest from
the records.

    python ml/probe_gaps.py --renders goals3

Each score is a song's mean cosine similarity, over its 10-second windows,
to the description's text embedding; the gap is ours minus the records'
median, in units of the records' spread (so +2 is "far more than records
do", -2 "far less"). A guide to what to work on, not a target: CLAP's words
are loose.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import goal_check  # noqa: E402
import library_stats as ls  # noqa: E402

PROBES = (
    # drums
    "shuffled swung 2-step garage hi-hats", "a live drummer playing reggae", "a one drop reggae drum beat",
    "four on the floor steppers kick drum", "half-time dubstep drums with a heavy snare",
    "a drum machine loop", "rimshots and cross-stick", "crisp open hi-hats", "shakers and percussion",
    # bass
    "a deep sub bass", "a melodic reggae bass guitar line", "a wobble bass", "a reese bass",
    "a bouncy garage bassline",
    # keys and guitar
    "an offbeat reggae guitar skank", "a reggae organ bubble", "Rhodes electric piano chords",
    "organ stabs", "synth pads", "a piano",
    # voice and lead
    "a sung vocal hook", "chopped and pitched vocal samples", "an MC toasting over the beat",
    "a melodica melody", "horns and brass", "a synth lead melody",
    # production
    "dub echo and spring reverb throws", "a heavily compressed modern mix", "a warm analogue 1970s recording",
    "vinyl crackle", "a dry close mix", "lots of reverb and space",
    # energy and arrangement
    "a sparse minimal arrangement", "a busy dense arrangement", "a big drop", "a slow build",
)


def scores(emb: dict, text: np.ndarray) -> dict[str, np.ndarray]:
    import judge
    t = judge._unit(text)
    return {k: (judge._unit(v) @ t.T).mean(0) for k, v in emb.items() if len(v)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--renders", default="goals3", help="goal_check.py's --out folder")
    ap.add_argument("--refs", default=ls.REFS)
    ap.add_argument("--per-family", type=int, default=120, help="reference tracks a family")
    ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args(argv)
    import judge

    text = judge.embed_text(list(PROBES))
    jdir = os.path.join(a.refs, "judge")
    ref = {}
    for fam in sorted({f for p in goal_check.PROFILES for f in goal_check.families_for(p)}):
        path = judge.ref_files(jdir).get(fam, os.path.join(jdir, f"{fam}.npz"))
        if os.path.exists(path):
            d = dict(np.load(path, allow_pickle=True))
            ref[fam] = scores(dict(list(d.items())[: a.per_family]), text)
    files = sorted(glob.glob(os.path.join(a.renders, "*.wav")))
    ours = scores(judge.embed_many(files, os.path.join(a.renders, "goal_emb.npz")), text)

    lines = ["# What the records have that ours don't (CLAP probes)", "",
             f"Renders: {a.renders}. Gap = ours minus the records' median, in records' "
             "standard deviations; listed largest first.", ""]
    for prof in goal_check.PROFILES:
        rs = np.array([v for f in goal_check.families_for(prof) for v in ref.get(f, {}).values()])
        os_ = np.array([v for k, v in ours.items() if os.path.basename(k).startswith(prof)])
        if not len(rs) or not len(os_):
            continue
        med, sd = np.median(rs, 0), rs.std(0) + 1e-6
        gap = (np.median(os_, 0) - med) / sd
        order = np.argsort(-np.abs(gap))[: a.top]
        lines += [f"## {prof} ({len(os_)} songs against {len(rs)} records)", "",
                  "| description | gap | ours | records |", "|---|---|---|---|"]
        lines += [f"| {PROBES[i]} | {gap[i]:+.1f} | {np.median(os_, 0)[i]:.3f} | {med[i]:.3f} |"
                  for i in order]
        lines.append("")
    out = os.path.join(a.renders, "gaps.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
