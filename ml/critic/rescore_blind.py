"""The 12 blind A/B calls again, made by the critic at song level, and
checked against Ed's picks.

    python ml/critic/rescore_blind.py [--seeds 16]     # writes ab-blind/critic.md

Each pair's change (ml/blind_ab.py pairs()) is rendered both ways on the same
seeds (401 on), and each song is scored by the critic (ml/critic/score.py:
mix and Demucs stems). The critic's call is the rule's verdict on the paired
differences: keep, revert or can't tell. Ed's pick for the pair is his
blind choice (ab-blind/picks.jsonl). The old judge's call was "keep" on all
12.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.dirname(HERE)
REPO = os.path.dirname(ML)
sys.path[:0] = [REPO, ML, HERE]

import score  # noqa: E402

OUT = os.path.join(REPO, "ab-blind")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--first", type=int, default=401)
    ap.add_argument("--keep-audio", action="store_true")
    a = ap.parse_args(argv)
    from blind_ab import pairs, patched
    from riddim.compose import compose
    from riddim.render import audio

    ed = {}
    for line in open(os.path.join(OUT, "picks.jsonl")):
        r = json.loads(line)
        ed[r["id"]] = r
    meta = {m["id"]: m for m in json.load(open(os.path.join(OUT, "pairs.json")))}
    work = os.path.join(OUT, "_critic")
    rows, agree, decided, tell = [], 0, 0, 0
    for i, p in enumerate(pairs()):
        paths = {}
        for side in ("kept", "other"):
            folder = os.path.join(work, f"{i:02d}_{side}")
            os.makedirs(folder, exist_ok=True)
            paths[side] = []
            for s in range(a.first, a.first + a.seeds):
                f = os.path.join(folder, f"{s}.wav")
                if not os.path.exists(f):
                    with patched(p[side]):
                        audio.write(compose(p["profile"], seed=s), f)
                paths[side].append(f)
        sc = {side: score.song_scores(paths[side], os.path.join(work, "_cache")) for side in paths}
        # B = the side the judge kept, A = the other: "keep" means the
        # critic prefers what the judge kept.
        res = score.compare(sc["other"], sc["kept"], list(zip(paths["other"], paths["kept"])))
        r = ed.get(i)
        if r is None or r["pick"] == "unsure":
            ed_pick = "can't tell"
        else:
            ed_pick = "kept" if r["clip"] == meta[i]["files"]["kept"] else "other"
        critic_pick = {"keep": "kept", "revert": "other"}.get(res["verdict"], "can't tell")
        if ed_pick != "can't tell":
            decided += 1
            if critic_pick != "can't tell":
                tell += 1
                agree += critic_pick == ed_pick
        cells = " | ".join(f"{res[k][0]:+.2f} ({res[k][1]:+.2f} to {res[k][2]:+.2f})" if k in res else "-"
                           for k in ("mix", "drums", "bass", "other"))
        rows.append(f"| {i} | {p['profile']} | {p['change']} | {cells} | {critic_pick} | {ed_pick} |")
        print(rows[-1], flush=True)
        if not a.keep_audio:
            for side in paths:
                shutil.rmtree(os.path.join(work, f"{i:02d}_{side}"), ignore_errors=True)
    lines = ["# The blind calls, by the critic", "",
             f"{a.seeds} seeds a side (from {a.first}), paired by seed; the change's effect on the critic's "
             "log-odds (kept side minus the other), mean and 95% interval.", "",
             "| # | profile | change | mix | drums | bass | other | critic | Ed |",
             "|---|---|---|---|---|---|---|---|---|"] + rows
    lines += ["", f"Of the {decided} pairs Ed decided, the critic made a call on {tell} and agreed with him on "
              f"{agree}. The old judge's calls (keep, all 12) agreed on 5 of 9."]
    with open(os.path.join(OUT, "critic.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[-1:]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
