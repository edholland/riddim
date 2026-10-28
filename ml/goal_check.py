"""Where riddim stands against the library plan's goals, measured the way the
references were: render a panel of songs as they are now, split each with
Demucs and set its stem levels beside the records', and ask the judge which
family each song sounds like.

    python ml/goal_check.py --out goals --per-profile 4

Needs what library_stats.py's stems and embed steps made (references/stats/
stems.jsonl and references/judge/<family>.npz) and runs where they are (the
Mac). Writes goals/report.md:

- **stems** - each profile's median drums / bass / other / vocals in dB
  against the mix, the references' median for the families it stands for,
  and the gap (the plan: dub's "other" within 3 dB of the records; a voice
  within -13 to -24 dB once there is one);
- **genre** - for each song, the reference family whose centroid it sits
  closest to, and where its own family ranks (the plan: garage heard as
  garage before dubstep).
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import library_stats as ls  # noqa: E402

PROFILES = ("roots_dub", "steppers_dub", "dubstep_140", "dark_garage", "uk_garage")
STEMS = ("drums", "bass", "other", "vocals")


def families_for(profile: str) -> list[str]:
    return [f for f, p in ls.FOR_PROFILES.items() if p == profile and f != "dub_techno"]


def ref_levels(stems: list[dict], profile: str) -> dict:
    rows = [r for r in stems if r.get("family") in families_for(profile)]
    return {s: float(np.median([r[s] for r in rows])) for s in STEMS} if rows else {}


def centroids(jdir: str, keep: dict[str, set[str]] | None = None) -> dict[str, np.ndarray]:
    """Each family's centroid; with `keep` (ml/instrumental.py), from those
    records only - a family with fewer than four left drops out."""
    import judge
    out = {}
    for fam, path in judge.ref_files(jdir).items():
        emb = [v for k, v in np.load(path, allow_pickle=True).items()
               if len(v) and (keep is None or k in keep.get(fam, ()))]
        if len(emb) >= 4:
            out[fam] = judge._unit(np.concatenate(emb).mean(0))
    return out


def ranks(emb: np.ndarray, cents: dict[str, np.ndarray]) -> list[tuple[str, float]]:
    import judge
    e = judge._unit(emb)
    return sorted(((f, float((e @ c).mean())) for f, c in cents.items()), key=lambda x: -x[1])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="goals")
    ap.add_argument("--per-profile", "--seeds", dest="per_profile", type=int, default=4,
                    help="songs a profile (16-32 to see past the noise)")
    ap.add_argument("--seed", type=int, default=301)
    ap.add_argument("--refs", default=ls.REFS)
    ap.add_argument("--no-stems", action="store_true", help="skip Demucs (genre only)")
    ap.add_argument("--instrumental", action="store_true",
                    help="genre against centroids of instrumental records only (ml/instrumental.py)")
    a = ap.parse_args(argv)

    from riddim.compose import compose
    from riddim.render import audio
    import judge

    os.makedirs(a.out, exist_ok=True)
    songs = []
    for prof in PROFILES:
        for k in range(a.per_profile):
            path = os.path.join(a.out, f"{prof}_{a.seed + k}.wav")
            if not os.path.exists(path):
                audio.write(compose(prof, seed=a.seed + k), path)
            songs.append((prof, path))
            print(f"rendered {path}", flush=True)

    lines = ["# Goal check", "", f"{a.per_profile} songs a profile, seeds from {a.seed}, as the "
             "branch renders them now (sample library on if present).", ""]

    if not a.no_stems:
        refs = ls._load(os.path.join(a.refs, "stats", "stems.jsonl"))
        ours = defaultdict(list)
        work = os.path.join(a.out, "_demucs")
        for prof, path in songs:
            ours[prof].append(ls.stem_levels(path, work))
            print(f"stems {path} {ours[prof][-1]}", flush=True)
        lines += ["## Stem levels (dB against the mix: ours / records, gap)", "",
                  "| profile | " + " | ".join(STEMS) + " |", "|---" * (len(STEMS) + 1) + "|"]
        for prof in PROFILES:
            ref = ref_levels(refs, prof)
            cells = []
            for s in STEMS:
                o = float(np.median([r[s] for r in ours[prof]]))
                cells.append(f"{o:.1f} / {ref[s]:.1f} ({o - ref[s]:+.1f})" if ref else f"{o:.1f}")
            lines.append(f"| {prof} | " + " | ".join(cells) + " |")
        lines.append("")

    keep = None
    if a.instrumental:
        import instrumental
        keep = instrumental.instrumental(a.refs)
    cents = centroids(os.path.join(a.refs, "judge"), keep)
    emb = judge.embed_many([p for _, p in songs], os.path.join(a.out, "goal_emb.npz"))
    lines += ["## Genre (closest reference family; where the profile's own families rank)"
              + (" - instrumental records only" if a.instrumental else ""), "",
              "| song | closest | own family rank | top three |", "|---|---|---|---|"]
    top1 = defaultdict(list)
    for prof, path in songs:
        if not len(emb[path]):
            continue
        r = ranks(emb[path], cents)
        names = [f for f, _ in r]
        own = min((names.index(f) + 1 for f in families_for(prof) if f in names), default=None)
        top1[prof].append(names[0] in families_for(prof))
        lines.append(f"| {os.path.basename(path)[:-4]} | {names[0]} | {own} | "
                     + ", ".join(f"{f} {s:.3f}" for f, s in r[:3]) + " |")
    lines += ["", "| profile | heard as its own family |", "|---|---|"]
    lines += [f"| {p} | {sum(v)}/{len(v)} |" for p, v in top1.items()]

    # The judge on each profile against its main family's records, with 95%
    # intervals over songs (judge.bootstrap): a change smaller than these
    # intervals is not one the judge can see.
    refs_by_fam = judge.ref_files(os.path.join(a.refs, "judge"))
    lines += ["", "## The judge, per profile (95% interval over songs)"
              + ("" if judge.NORM else " - level-sensitive"), "",
              "| profile | against | FAD | closeness | real |", "|---|---|---|---|---|"]
    for prof in PROFILES:
        fam = families_for(prof)[0]
        if fam not in refs_by_fam:
            continue
        ref = dict(np.load(refs_by_fam[fam], allow_pickle=True))
        mine = {p: emb[p] for q, p in songs if q == prof and len(emb[p])}
        res, ci = judge.score(ref, mine), judge.bootstrap(ref, mine)
        t = res["tracks"].values()
        c = float(np.median([x["closeness"] for x in t]))
        r = float(np.median([x["real"] for x in t]))
        lines.append(f"| {prof} | {fam} | {res['fad']:.3f} ({ci['fad'][0]:.3f}-{ci['fad'][1]:.3f}) | "
                     f"{c:.3f} ({ci['closeness'][0]:.3f}-{ci['closeness'][1]:.3f}) | "
                     f"{r:.2f} ({ci['real'][0]:.2f}-{ci['real'][1]:.2f}) |")

    # Loudness, its own target now that the judge is loudness-blind.
    from riddim import listen
    stats = [ls.fold(r) for r in ls._load(os.path.join(a.refs, "stats", "stats.jsonl"))]
    lines += ["", "## Loudness (ours / records' median)", "",
              "| profile | RMS dB | crest dB | dynamic range dB |", "|---|---|---|---|"]
    for prof in PROFILES:
        rep = [listen.analyze(p) for q, p in songs if q == prof]
        recs = [r for r in stats if r.get("family") in families_for(prof)]
        if not rep or not recs:
            continue
        cell = lambda k: (f"{np.median([getattr(x, k) for x in rep]):.1f} / "  # noqa: E731
                          f"{np.median([r[k] for r in recs if r.get(k) is not None]):.1f}")
        lines.append(f"| {prof} | {cell('rms_db')} | {cell('crest_db')} | {cell('dynamic_range_db')} |")
    with open(os.path.join(a.out, "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
