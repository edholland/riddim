"""Blind before/after pairs for Ed's ears: changes kept on the measured
targets that need his say, a candidate the targets couldn't settle, and
sounds only he picks (sample choice is his call).

    python ml/audition_ab.py build kept        # -> ab-kept/
    python ml/audition_ab.py build snares      # -> ab-snares/
    python ml/audition_ab.py serve kept --port 8780
    python ml/audition_ab.py score kept

Each pair is one song rendered two ways - "after" as named, "before" as it
was - cut to about 45 s from two bars before the first drop and matched in
loudness, with ml/blind_ab.py's page, server and picks: the page never says
which side is which. A pick of the "after" side is a keep.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [REPO, HERE]

import blind_ab  # noqa: E402

CLIP_SECONDS = 45.0
SEEDS = (501, 502)


def _kept() -> list[dict]:
    """Before = the change undone. 'After' is the branch as it is, except
    for uk's deeper sub, which is the candidate itself."""
    from riddim.profile import Register
    from riddim.render import audio
    out = []
    roots_kit = {"snare_gain": 0.55, "hat_gain": 1.15}
    old_lift = {**audio.LEAD_LIFT_DB, "roots_dub": 18.0, "steppers_dub": 23.0}
    uk = "uk_garage"
    for s in SEEDS:
        out.append(dict(item="roots' louder, shorter kick (dc59172)", profile="roots_dub", seed=s, after=[],
                        before=[("item", audio.DRUM_SHAPE, "roots_dub", roots_kit),
                                ("item", audio.DRIVE, "roots_dub", 2.4)]))
    for p, s in (("roots_dub", SEEDS[0]), ("uk_garage", SEEDS[0])):
        out.append(dict(item="the darker lead (8e2744b)", profile=p, seed=s, after=[],
                        before=[("attr", audio, "LEAD_LP", {}), ("attr", audio, "LEAD_PRESENCE", 0.5),
                                ("attr", audio, "LEAD_LIFT_DB", old_lift)]))
    for s in SEEDS:
        out.append(dict(item="uk's deeper sub (candidate, not kept)", profile=uk, seed=s, before=[],
                        after=[("profile", uk, {"sub": Register(24, 45)}),
                               ("item", audio.DRUM_SHAPE, uk, {**audio.DRUM_SHAPE.get(uk, {}),
                                                              "kick_sub_decay_ms": 250.0, "kick_gain": 1.2}),
                               ("item", audio.FAMILY_GAIN, "garage", {**audio.FAMILY_GAIN.get("garage", {}),
                                                                      "sub": 1.4}),
                               ("item", audio.DRIVE, uk, 5.3),
                               ("item", audio.TONE, uk, {**audio.TONE[uk],
                                                         "low_mid": audio.TONE[uk]["low_mid"] - 0.6,
                                                         "high_mid": audio.TONE[uk]["high_mid"] - 0.6})]))
    return out


# Thin, crackly one-shots from the library (body 150-400 Hz against crack
# 2-8 kHz: -11 to -13 dB, the records' dubstep snares sit at -12; ours -> +4).
SNARES = [
    "musicradar-bassline-samples/Drum_Hits/BL_Snare04",
    "Tape Treated/Cassette Drum Hits/Snare 05",
    "Hits/Snare Drum \\[SD\\]/E808_SD-06",
    "Nu Garage/Drum_hits/Snare11",
]


def _snares() -> list[dict]:
    out = []
    for pat in SNARES:
        for s in SEEDS:
            out.append(dict(item=f"dubstep snare: {pat.replace(chr(92), '')}", profile="dubstep_140", seed=s,
                            before=[], after=[("env", "RIDDIM_PICKS", f"dubstep.snare={pat}")]))
    return out


SETS = {"kept": (_kept, "ab-kept"), "snares": (_snares, "ab-snares")}


def cmd_build(a) -> int:
    from riddim.compose import compose
    from riddim.render import audio
    blind_ab.CLIP_SECONDS = CLIP_SECONDS
    make, _ = SETS[a.set]
    os.makedirs(os.path.join(a.out, "clips"), exist_ok=True)
    meta = []
    for i, p in enumerate(make()):
        sides = {}
        with blind_ab.patched(p["before"]):
            span = blind_ab._span(compose(p["profile"], seed=p["seed"]))
        for side in ("after", "before"):
            with blind_ab.patched(p[side]):
                sides[side] = audio.render(compose(p["profile"], seed=p["seed"]), span)
        x, y = blind_ab._match(sides["after"], sides["before"])
        files = {}
        for side, z in (("after", x), ("before", y)):
            name = f"clips/{i:02d}_{random.Random(f'{i}:{side}:{a.set}').getrandbits(32):08x}.wav"
            blind_ab._write(os.path.join(a.out, name), z)
            files[side] = name
        meta.append({"id": i, "item": p["item"], "profile": p["profile"], "seed": p["seed"],
                     "files": {"kept": files["after"], "other": files["before"]}})
        print(f"  {i:2d} {p['profile']:13s} {p['seed']}  {p['item']}", flush=True)
    with open(os.path.join(a.out, "pairs.json"), "w") as f:
        json.dump(meta, f, indent=1)
    with open(os.path.join(a.out, "index.html"), "w") as f:
        f.write(PAGE)
    print(f"{len(meta)} pairs -> {a.out}")
    return 0


def cmd_score(a) -> int:
    meta = {m["id"]: m for m in json.load(open(os.path.join(a.out, "pairs.json")))}
    path = os.path.join(a.out, "picks.jsonl")
    last = {}
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            last[r["id"]] = r
    by: dict[str, list[str]] = {}
    for i, m in sorted(meta.items()):
        r = last.get(i)
        pick = "-" if r is None else "can't tell" if r["pick"] == "unsure" else \
            "after" if r["clip"] == m["files"]["kept"] else "before"
        by.setdefault(m["item"], []).append(pick)
    for item, picks in by.items():
        print(f"{item}: {', '.join(picks)}")
    return 0


PAGE = blind_ab.PAGE.replace("<title>Blind A/B</title>", "<title>Audition</title>").replace(
    "<h1>Blind A/B</h1>", "<h1>Audition</h1>").replace(
    "Pick the one that sounds more like a record.", "Pick the one you'd rather play out.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", choices=("build", "serve", "score"))
    ap.add_argument("set", choices=tuple(SETS))
    ap.add_argument("--port", type=int, default=8780)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(argv)
    a.out = os.path.join(REPO, SETS[a.set][1])
    if a.step == "serve":
        return blind_ab.cmd_serve(a)
    return {"build": cmd_build, "score": cmd_score}[a.step](a)


if __name__ == "__main__":
    sys.exit(main())
