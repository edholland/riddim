"""A blind A/B of the drops' hook, for Ed's ears: the trained model's lead
(riddim/model_lead.py, --model-lead) against riddim's own.

    python ml/hook_ab.py build            # render the pairs (about 10 minutes)
    python ml/hook_ab.py serve            # open the listening page
    python ml/hook_ab.py score            # which side Ed picked, pair by pair

Each pair is one song (one profile, one seed, the same params) rendered with
riddim's lead and with the model's in the drops; nothing else differs. Both
sides are cut to about 45 s from two bars before the first drop and matched
in loudness, with the two leads at one level while they play (the model's
re-rendered to the rule's), and the page, server and picks are ml/blind_ab.py's, so which
side is which never reaches the page. RIDDIM_ML_MODEL picks the model (the
base music-medium-800k unless a fine-tune is under ~/ml/models).
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

OUT = os.path.join(REPO, "ab-hooks")
CLIP_SECONDS = 45.0
PROFILES = ("roots_dub", "steppers_dub", "dubstep_140", "dark_garage", "uk_garage")
PAIRS = [(p, s) for p in PROFILES for s in (401, 402)]


def _lead_db(lead, sr: int) -> float:
    """The lead's level while it plays: RMS over the 100 ms windows within
    30 dB of its loudest."""
    import numpy as np
    x = lead.mean(axis=1)
    w = int(0.1 * sr)
    n = len(x) // w
    r = np.sqrt((x[: n * w].reshape(n, w) ** 2).mean(axis=1)) + 1e-12
    on = r > r.max() * 10 ** (-30 / 20)
    return float(20 * np.log10(np.sqrt(np.mean(r[on] ** 2))))


def cmd_build(a) -> int:
    from riddim import model_lead
    from riddim.compose import compose
    from riddim.render import audio
    why = model_lead.unavailable()
    if why:
        print(f"error: {why}", file=sys.stderr)
        return 1
    blind_ab.CLIP_SECONDS = CLIP_SECONDS
    os.makedirs(os.path.join(a.out, "clips"), exist_ok=True)
    meta = []
    pairs = [(p, int(x)) for p in PROFILES for x in a.seeds.split(",")] if a.seeds else PAIRS
    for i, (prof, seed) in enumerate(pairs):
        rule = compose(prof, seed=seed)
        model = model_lead.write(compose(prof, seed=seed), take=0, style=a.style)
        span = blind_ab._span(rule)
        sides, lead_db = {}, {}
        for side, song in (("rule", rule), ("model", model)):
            so: dict = {}
            sides[side] = audio.render(song, span, stems_out=so)
            lead_db[side] = _lead_db(so["lead"], audio.SR)
        # The two leads at one level (while they play), so the louder hook
        # can't win on loudness: the model's re-rendered at the rule's.
        trim = lead_db["rule"] - lead_db["model"]
        if abs(trim) > 0.5:
            old = dict(audio.LEAD_LIFT_DB)
            audio.LEAD_LIFT_DB[prof] = old.get(prof, 0.0) + trim
            try:
                so = {}
                sides["model"] = audio.render(model, span, stems_out=so)
                lead_db["model"] = _lead_db(so["lead"], audio.SR)
            finally:
                audio.LEAD_LIFT_DB.clear()
                audio.LEAD_LIFT_DB.update(old)
        m, r = blind_ab._match(sides["model"], sides["rule"])
        files = {}
        for side, x in (("model", m), ("rule", r)):
            name = f"clips/{i:02d}_{random.Random(f'{i}:{side}:{a.salt}').getrandbits(32):08x}.wav"
            blind_ab._write(os.path.join(a.out, name), x)
            files[side] = name
        # The server hands the page files["kept"] and files["other"], unlabelled.
        meta.append({"id": i, "profile": prof, "seed": seed, "model": str(model_lead.default_model()),
                     "style": a.style, "lead_db": {k: round(v, 2) for k, v in lead_db.items()}, "files": {"kept": files["model"], "other": files["rule"]}})
        print(f"  {i:2d} {prof:13s} {seed}  lead rule {lead_db['rule']:.1f} model {lead_db['model']:.1f} dB",
              flush=True)
    with open(os.path.join(a.out, "pairs.json"), "w") as f:
        json.dump(meta, f, indent=1)
    with open(os.path.join(a.out, "index.html"), "w") as f:
        f.write(PAGE)
    print(f"{len(meta)} pairs -> {a.out}  (now: python ml/hook_ab.py serve)")
    return 0


def cmd_score(a) -> int:
    from math import comb
    meta = {m["id"]: m for m in json.load(open(os.path.join(a.out, "pairs.json")))}
    path = os.path.join(a.out, "picks.jsonl")
    if not os.path.exists(path):
        print("no picks yet")
        return 1
    last = {}
    for line in open(path):
        r = json.loads(line)
        last[r["id"]] = r
    model = rule = unsure = 0
    print("| pair | profile | seed | Ed picked |")
    print("|---|---|---|---|")
    for i, m in sorted(meta.items()):
        r = last.get(i)
        if r is None:
            pick = "-"
        elif r["pick"] == "unsure":
            pick, unsure = "can't tell", unsure + 1
        elif r["clip"] == m["files"]["kept"]:
            pick, model = "model", model + 1
        else:
            pick, rule = "riddim", rule + 1
        print(f"| {i} | {m['profile']} | {m['seed']} | {pick} |")
    n = model + rule
    if n:
        k = max(model, rule)
        p = min(1.0, 2 * sum(comb(n, j) for j in range(k, n + 1)) / 2 ** n)
        print(f"\nmodel {model}, riddim {rule}, can't tell {unsure}; two-sided sign test p = {p:.3f}")
    return 0


PAGE = blind_ab.PAGE.replace("<title>Blind A/B</title>", "<title>Hook A/B</title>").replace(
    "<h1>Blind A/B</h1>", "<h1>Hook A/B</h1>").replace(
    "Each pair is the same moment of one song, made two ways and matched in loudness. "
    "Pick the one that sounds more like a record.",
    "Each pair is the same drop of one song with two different leads, matched in loudness. "
    "Pick the hook you'd rather play out.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", choices=("build", "serve", "score"))
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--port", type=int, default=8778)
    ap.add_argument("--salt", default="hooks", help="build: varies the clip file names")
    ap.add_argument("--style", default="rhythm", choices=("rhythm", "free"))
    ap.add_argument("--seeds", default="", help="build: seeds per profile, e.g. 403,404 (default 401,402)")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(argv)
    if a.step == "serve":
        return blind_ab.cmd_serve(a)
    return {"build": cmd_build, "score": cmd_score}[a.step](a)


if __name__ == "__main__":
    sys.exit(main())
