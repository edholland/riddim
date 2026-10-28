"""Which embedding should the critic stand on? A bake-off of the backbones in
ml/critic/embed.py, all loudness-normalised and over the same windows.

    python ml/critic/bakeoff.py        # writes references/critic/bakeoff.md

Three tests per backbone (and the concatenation of all three):

1. **Family.** Can a linear probe tell the reference families apart? Each
   record is the mean of its windows; logistic regression, 5-fold stratified
   cross-validation over records, macro (balanced) accuracy. Also on the
   instrumental records only (ml/instrumental.py), for families with at least
   10 of them.
2. **Noise.** The same code rendered twice: two disjoint 16-seed panels per
   profile (seeds 301-316 and 317-332, critic_panel/). FAD-infinity (the FAD
   extrapolated to infinitely many windows, fitted over window subsets) of
   each panel against the profile's main reference family. The noise is the
   panels' relative difference |A - B| / mean(A, B). Smaller means a smaller
   change can be believed.
3. **Ed.** Agreement with his picks:
   - the 9 blind A/B pairs he decided (ab-blind/);
   - his 10 library-over-synth picks (ab/, not blind).
   Each side is scored by its closeness to the family's reference centroid;
   the higher side is the backbone's pick.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.dirname(HERE)
REPO = os.path.dirname(ML)
sys.path[:0] = [REPO, ML, HERE]

import embed  # noqa: E402

REFS = os.path.join(REPO, "references")
CACHE = os.path.join(REFS, "critic")
PANEL = os.path.join(REPO, "critic_panel")
PROFILES = ("roots_dub", "steppers_dub", "dubstep_140", "dark_garage", "uk_garage")
FAMILIES = ("speed_garage", "dark_garage", "uk_garage", "grime", "dub_techno", "digital_dub",
            "steppers_dub", "roots_dub", "roots_reggae", "dubstep")


def reference_paths() -> dict[str, list[str]]:
    """The records library_stats.py embeds for the judge: the first 300 a
    family in plan order."""
    plan = [json.loads(line) for line in open(os.path.join(REFS, "stats", "plan.jsonl"))]
    out = {}
    for fam in FAMILIES:
        files = [os.path.join(REFS, p["file"]) for p in plan if p["kind"] == "record" and p["family"] == fam]
        out[fam] = [f for f in files[:300] if os.path.exists(f)]
    return out


def render_panel(seeds=range(301, 333)) -> dict[str, list[str]]:
    from riddim.compose import compose
    from riddim.render import audio
    os.makedirs(PANEL, exist_ok=True)
    out = {}
    for prof in PROFILES:
        out[prof] = []
        for s in seeds:
            f = os.path.join(PANEL, f"{prof}_{s}.wav")
            if not os.path.exists(f):
                audio.write(compose(prof, seed=s), f)
            out[prof].append(f)
    return out


def _unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def fad(a: np.ndarray, b: np.ndarray) -> float:
    from scipy import linalg
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean = np.real(linalg.sqrtm(s1.dot(s2)))
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


def fad_inf(ref: np.ndarray, test: np.ndarray, steps: int = 5, seed: int = 0) -> float:
    """FAD extrapolated to infinitely many test windows: FAD on random
    subsets of 50-100% of the windows, a line in 1/n, its intercept."""
    rng = np.random.default_rng(seed)
    ns = np.linspace(0.5, 1.0, steps) * len(test)
    xs, ys = [], []
    for n in ns:
        n = int(n)
        idx = rng.choice(len(test), n, replace=False)
        xs.append(1.0 / n)
        ys.append(fad(ref, test[idx]))
    slope, icept = np.polyfit(xs, ys, 1)
    return float(icept)


def family_probe(feats: dict[str, dict[str, np.ndarray]], keep: dict[str, set] | None = None,
                 min_n: int = 10) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X, y = [], []
    for fam, d in feats.items():
        rows = [_unit(v.mean(0)) for k, v in d.items() if len(v) and (keep is None or k in keep.get(fam, ()))]
        if len(rows) < min_n:
            continue
        X += rows
        y += [fam] * len(rows)
    X, y = np.array(X), np.array(y)
    pred = np.empty_like(y)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5))
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict(X[te])
    return float(balanced_accuracy_score(y, pred))


def ed_pairs() -> list[tuple[str, str, str, str]]:
    """(profile, Ed's side, the other side, source) for each pick Ed made."""
    out = []
    meta = {m["id"]: m for m in json.load(open(os.path.join(REPO, "ab-blind", "pairs.json")))}
    last = {}
    for line in open(os.path.join(REPO, "ab-blind", "picks.jsonl")):
        r = json.loads(line)
        last[r["id"]] = r
    for i, r in last.items():
        if r["pick"] == "unsure":
            continue
        m = meta[i]
        files = {s: os.path.join(REPO, "ab-blind", m["files"][s]) for s in ("kept", "other")}
        mine = os.path.join(REPO, "ab-blind", r["clip"])
        other = files["other"] if mine == files["kept"] else files["kept"]
        out.append((m["profile"], mine, other, "blind"))
    import glob
    for f in sorted(glob.glob(os.path.join(REPO, "ab", "*_library.wav"))):
        prof = os.path.basename(f).rsplit("_", 2)[0]
        out.append((prof, f, f.replace("_library.wav", "_synth.wav"), "library"))
    return out


def main() -> int:
    import goal_check as gc
    refs = reference_paths()
    print("embedding the references", {f: len(v) for f, v in refs.items()}, flush=True)
    ref_emb = {}
    for fam, paths in refs.items():
        print(f"  {fam}", flush=True)
        r = embed.embed_files(paths, CACHE, fam)
        for b, d in r.items():
            ref_emb.setdefault(b, {})[fam] = d
    print("rendering and embedding the panel", flush=True)
    panel = render_panel()
    panel_emb = {}
    for prof, paths in panel.items():
        r = embed.embed_files(paths, os.path.join(PANEL, "_emb"), prof)
        for b, d in r.items():
            panel_emb.setdefault(b, {})[prof] = d
    pairs = ed_pairs()
    pair_emb = embed.embed_files([p for t in pairs for p in t[1:3]], os.path.join(REPO, "ab-blind", "_emb"), "ed")

    # The concatenation: each backbone's window embeddings unit-normalised, side by side.
    def concat(d_by_b):
        out = {}
        for k in d_by_b[embed.BACKBONES[0]]:
            parts = [d_by_b[b][k] for b in embed.BACKBONES]
            if all(len(p) for p in parts):
                n = min(len(p) for p in parts)
                out[k] = np.concatenate([_unit(p[:n]) for p in parts], axis=1)
            else:
                out[k] = np.zeros((0, 1))
        return out
    ref_emb["concat"] = {fam: concat({b: ref_emb[b][fam] for b in embed.BACKBONES}) for fam in refs}
    panel_emb["concat"] = {p: concat({b: panel_emb[b][p] for b in embed.BACKBONES}) for p in panel}
    pair_emb["concat"] = concat(pair_emb)

    import instrumental
    import judge
    was = judge.NORM
    judge.NORM = False          # the instrumental keys as calibrated, on the level-sensitive references
    keep = instrumental.instrumental()
    judge.NORM = was
    # instrumental keys are judge cache keys (absolute path::size); map to ours
    keep_paths = {fam: {k.split("::")[0] for k in ks} for fam, ks in keep.items()}

    lines = ["# Embedding bake-off", "",
             "All loudness-normalised (-14 dBFS RMS a track), up to 12 windows of 10 s a recording. "
             "Higher family accuracy is better; lower panel noise is better; Ed agreement counts "
             "the pairs where the side nearer the family's references is the one Ed picked.", "",
             "| backbone | family acc (10 families) | family acc (instrumental) | panel noise "
             "\\|A-B\\|/mean, median over profiles | Ed blind | Ed library | Ed total |",
             "|---|---|---|---|---|---|---|"]
    for b in list(embed.BACKBONES) + ["concat"]:
        feats = {fam: {k.split("::")[0] if "::" in k else k: v for k, v in d.items()}
                 for fam, d in ref_emb[b].items()}
        acc = family_probe(feats)
        acc_i = family_probe(feats, keep_paths)
        noise = []
        for prof, d in panel_emb[b].items():
            fam = gc.families_for(prof)[0]
            ref = np.concatenate([v for v in ref_emb[b][fam].values() if len(v)])
            files = panel[prof]
            half_a = np.concatenate([d[f] for f in files[:16] if len(d[f])])
            half_b = np.concatenate([d[f] for f in files[16:] if len(d[f])])
            fa, fb = fad_inf(ref, half_a), fad_inf(ref, half_b)
            noise.append(abs(fa - fb) / ((fa + fb) / 2))
        cents = {fam: _unit(np.concatenate([v for v in d.values() if len(v)]).mean(0))
                 for fam, d in ref_emb[b].items()}
        agree = {"blind": [0, 0], "library": [0, 0]}
        for prof, mine, other, src in pairs:
            c = cents[gc.families_for(prof)[0]]
            sm = float((_unit(pair_emb[b][mine]) @ c).mean())
            so = float((_unit(pair_emb[b][other]) @ c).mean())
            agree[src][0] += sm > so
            agree[src][1] += 1
        tot = agree["blind"][0] + agree["library"][0], agree["blind"][1] + agree["library"][1]
        lines.append(f"| {b} | {acc:.2f} | {acc_i:.2f} | {np.median(noise):.3f} | "
                     f"{agree['blind'][0]}/{agree['blind'][1]} | {agree['library'][0]}/{agree['library'][1]} | "
                     f"{tot[0]}/{tot[1]} |")
        print(lines[-1], flush=True)
    path = os.path.join(CACHE, "bakeoff.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
