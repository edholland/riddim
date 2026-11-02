"""Re-score dmz_filter.py's features for the UK garage feel.

usage: garage_filter.py DMZ_SCORES.jsonl OUT.jsonl

dmz_filter scores files for the dub and half-time 140 sound; the garage
profiles' hooks come out better from the base model than from a dub
fine-tune (docs/goals.md, 1 November). Same features, a garage score:

tempo   128-140 (or felt at half)
drums   a backbeat on 2 and 4 with a broken, not four-to-the-floor, kick
bass    present
mode    minor
chords  stabs (half a chord a bar or more), not too busy

Writes each row with `score` replaced (the dub score kept as `dmz_score`),
for build_data.py.
"""
import json
import sys


def score(f: dict) -> float:
    b = f["bpm"]
    if 128 <= b <= 140:
        s = 1.5
    elif 64 <= b <= 70:
        s = 0.5
    else:
        return -9.0
    if not f.get("has_drums"):
        return -9.0
    if f["snare_24"] > 0.6:
        s += 1.5                    # the backbeat on 2 and 4
    if f["kick_4"] < 0.6:
        s += 1.0                    # a broken, 2-step kick
    elif f["kick_4"] > 0.85:
        s -= 1.0                    # four to the floor: house
    if f["kick_on1"] > 0.15:
        s += 0.5
    if f.get("has_bass"):
        s += 0.5
    if f["minorness"] > 0.05:
        s += 1.0
    elif f["minorness"] < -0.1:
        s -= 0.5
    if f.get("chords_per_bar", 0) >= 0.5:
        s += 0.5
    if f["density"] > 60:
        s -= 1.0
    return round(s, 2)


if __name__ == "__main__":
    src, out = sys.argv[1], sys.argv[2]
    with open(src) as fin, open(out, "w") as fout:
        for line in fin:
            f = json.loads(line)
            fout.write(json.dumps({**f, "dmz_score": f["score"], "score": score(f)}) + "\n")
