"""A/B the sample library: the same songs rendered with the synths and with
the recorded instruments from samples/packs, side by side for your ears,
and scored by the judge against each family's references if it can.

    python ml/library_ab.py --out ab --per-family 2
    python ml/library_ab.py --out ab --refs dub=references/dub.npz \\
        --refs dubstep=references/dubstep.npz --refs garage=references/garage.npz

Writes ab/<profile>_<seed>_synth.wav and ..._library.wav, and ab/report.md:
for each pair what the library changed (which organ, skank, sub, hats, siren)
and, with --refs, the judge's closeness and real-record score for both.
The judge is a guide; the pairs are for listening.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PANEL = {"dub": ("roots_dub", "steppers_dub"), "dubstep": ("dubstep_140", "dark_garage"),
         "garage": ("uk_garage",)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="ab")
    ap.add_argument("--per-family", type=int, default=2, help="songs per profile")
    ap.add_argument("--seed", type=int, default=201, help="first seed")
    ap.add_argument("--refs", action="append", default=[],
                    help="family=embeddings.npz from `ml/judge.py embed` (repeatable)")
    ap.add_argument("--library", default=None, help="packs folder (default samples/packs)")
    a = ap.parse_args(argv)

    from riddim import library, profile
    from riddim.compose import compose
    from riddim.render import audio

    os.makedirs(a.out, exist_ok=True)
    lib_root = a.library or os.environ.get("RIDDIM_LIBRARY") or library.DEFAULT_ROOT
    if library.get(lib_root) is None:
        print(f"no sample library at {lib_root}", file=sys.stderr)
        return 1
    pairs = []
    for fam, profs in PANEL.items():
        for prof in profs:
            for k in range(a.per_family):
                seed = a.seed + k
                song = compose(prof, seed=seed)
                base = os.path.join(a.out, f"{prof}_{seed}")
                os.environ["RIDDIM_LIBRARY"] = "off"
                library._LIB.clear()
                audio.write(song, base + "_synth.wav")
                os.environ["RIDDIM_LIBRARY"] = lib_root
                library._LIB.clear()
                inst = audio.instruments(profile.get(prof).family, song.seed or 0, song.bpm,
                                         song.bpm >= 120 or "dubstep" in prof, "garage" in prof)
                audio.write(song, base + "_library.wav")
                used = ", ".join(f"{v}: {_name(inst[v])}" for v in
                                 ("organ", "skank", "sub", "hat", "shaker", "siren") if v in inst)
                pairs.append((fam, prof, seed, base, used))
                print(f"{prof} {seed}: {used}", flush=True)

    scores = {}
    refs = dict(r.split("=", 1) for r in a.refs)
    if refs:
        import numpy as np
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import judge
        for fam, path in refs.items():
            ref = dict(np.load(path, allow_pickle=True))
            mine = [p for p in pairs if p[0] == fam]
            files = [b + s for _, _, _, b, _ in mine for s in ("_synth.wav", "_library.wav")]
            emb = judge.embed_many(files, os.path.join(a.out, f"ab_{fam}_tests.npz"))
            res = judge.score(ref, emb)
            for f, r in res["tracks"].items():
                scores[f] = r
            for side in ("_synth.wav", "_library.wav"):
                part = {f: e for f, e in emb.items() if f.endswith(side)}
                scores[(fam, side)] = judge.score(ref, part)["fad"] if len(part) > 1 else None

    lines = ["# Sample library A/B", "",
             "Same songs, synths against the library. Listen to the pairs; the judge's numbers "
             "(closeness to the family's references, real-record probability) are a guide.", "",
             "| song | library voices | synth close / real | library close / real |",
             "|---|---|---|---|"]
    for fam, prof, seed, base, used in pairs:
        s, l = scores.get(base + "_synth.wav"), scores.get(base + "_library.wav")
        fmt = (lambda r: f"{r['closeness']:.3f} / {r['real']:.2f}" if r else "-")
        lines.append(f"| {prof} {seed} | {used} | {fmt(s)} | {fmt(l)} |")
    if refs:
        lines += ["", "| family | FAD synth | FAD library |", "|---|---|---|"]
        for fam in refs:
            fs, fl = scores.get((fam, "_synth.wav")), scores.get((fam, "_library.wav"))
            lines.append(f"| {fam} | {fs if fs is None else round(fs, 3)} | "
                         f"{fl if fl is None else round(fl, 3)} |")
    with open(os.path.join(a.out, "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def _name(v) -> str:
    if isinstance(v, str):
        return os.path.splitext(os.path.basename(v))[0]
    return os.path.basename(getattr(v, "name", "")) or type(v).__name__


if __name__ == "__main__":
    sys.exit(main())
