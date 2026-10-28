"""A/B the sample library: the same songs rendered with the synths and with
the recorded instruments from samples/packs, side by side for your ears,
and scored by the judge against each family's references if it can.

    python ml/library_ab.py --out ab --per-family 2
    python ml/library_ab.py --out ab --refs dub=references/dub.npz \\
        --refs dubstep=references/dubstep.npz --refs garage=references/garage.npz

With --cells the pair is the library with the generator's basslines (A,
..._gen.wav) against the bass rhythms fitted to the records (B, ..._cells.wav,
riddim/theory/cells.json) on every walking bassline.

With --toggle sub_sixteenths (any Profile field) the pair is each profile
with that field off (A, ..._off.wav) against the profile as it is (B,
..._on.wav); profiles where it is already off are left out.

With --gain "dub.drums=0.7,dubstep.drums=0.8" the pair is instead the
library as it is (A) against the library with those mix trims (B, files
..._trim.wav) - for setting levels against the measured stems.

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
    ap.add_argument("--gain", default="", help="A/B mix trims instead of synth/library, "
                                               "e.g. dub.drums=0.7")
    ap.add_argument("--cells", action="store_true",
                    help="A/B the generator's bass rhythms against the mined fit")
    ap.add_argument("--toggle", default="", help="A/B a Profile field off against as it is")
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
    sides = (("_off.wav", "_on.wav") if a.toggle else ("_gen.wav", "_cells.wav") if a.cells else
             ("_library.wav", "_trim.wav") if a.gain else ("_synth.wav", "_library.wav"))
    for fam, profs in PANEL.items():
        for prof in profs:
            if a.toggle and not getattr(profile.get(prof), a.toggle):
                continue
            for k in range(a.per_family):
                seed = a.seed + k
                song = compose(prof, seed=seed)
                base = os.path.join(a.out, f"{prof}_{seed}")
                os.environ["RIDDIM_LIBRARY"] = lib_root
                library._LIB.clear()
                if a.toggle:
                    from dataclasses import replace
                    off = type(getattr(profile.get(prof), a.toggle))(0)
                    audio.write(compose(replace(profile.get(prof), **{a.toggle: off}), seed=seed),
                                base + "_off.wav")
                    audio.write(song, base + "_on.wav")
                elif a.cells:
                    from dataclasses import replace
                    for side, mined in (("_gen.wav", 0.0), ("_cells.wav", 1.0)):
                        song = compose(replace(profile.get(prof), mined_bass=mined), seed=seed)
                        audio.write(song, base + side)
                elif a.gain:
                    os.environ.pop("RIDDIM_FAMILY_GAIN", None)
                    audio.write(song, base + "_library.wav")
                    os.environ["RIDDIM_FAMILY_GAIN"] = a.gain
                    audio.write(song, base + "_trim.wav")
                    os.environ.pop("RIDDIM_FAMILY_GAIN", None)
                else:
                    os.environ["RIDDIM_LIBRARY"] = "off"
                    library._LIB.clear()
                    audio.write(song, base + "_synth.wav")
                    os.environ["RIDDIM_LIBRARY"] = lib_root
                    library._LIB.clear()
                    audio.write(song, base + "_library.wav")
                inst = audio.instruments(profile.get(prof).family, song.seed or 0, song.bpm,
                                         song.bpm >= 120 or "dubstep" in prof, "garage" in prof)
                used = ", ".join(f"{v}: {_name(inst[v])}" for v in
                                 ("organ", "skank", "sub", "hat", "shaker", "siren") if v in inst)
                if a.cells:
                    used = f"bass style: {song.params.bass.style}"
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
            files = [b + s for _, _, _, b, _ in mine for s in sides]
            emb = judge.embed_many(files, os.path.join(a.out, f"ab_{fam}_tests.npz"))
            res = judge.score(ref, emb)
            for f, r in res["tracks"].items():
                scores[f] = r
            for side in sides:
                part = {f: e for f, e in emb.items() if f.endswith(side)}
                scores[(fam, side)] = judge.score(ref, part)["fad"] if len(part) > 1 else None

    lines = ["# Sample library A/B", "",
             "Same songs, synths against the library. Listen to the pairs; the judge's numbers "
             "(closeness to the family's references, real-record probability) are a guide.", "",
             f"| song | library voices | {sides[0][1:-4]} close / real | {sides[1][1:-4]} close / real |",
             "|---|---|---|---|"]
    for fam, prof, seed, base, used in pairs:
        s, l = scores.get(base + sides[0]), scores.get(base + sides[1])
        fmt = (lambda r: f"{r['closeness']:.3f} / {r['real']:.2f}" if r else "-")
        lines.append(f"| {prof} {seed} | {used} | {fmt(s)} | {fmt(l)} |")
    if refs:
        lines += ["", f"| family | FAD {sides[0][1:-4]} | FAD {sides[1][1:-4]} |", "|---|---|---|"]
        for fam in refs:
            fs, fl = scores.get((fam, sides[0])), scores.get((fam, sides[1]))
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
