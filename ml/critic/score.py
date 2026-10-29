"""The critic's verdict on riddim's renders, and the keep-or-revert rule.

A song's score per part is the critic's log-odds (ml/critic/train.py critic):
- the mix, through a record's codec as in training (dataset.lossy)
- its drums, bass and other, split by Demucs as the records' were
  (mine_cells.split)

Higher is more like a record.

compare() sets two renders of the same seeds side by side (A as it is, B with
a change) and returns, per part, the mean paired difference B - A with a 95%
bootstrap interval over seeds. Pairing by seed takes the song's own content
out of the comparison, which the judge's FAD on two separate sets never
could.

The rule: keep B if the mix's interval sits above 0 and no stem's interval
sits below 0; revert if the mix's sits below 0; otherwise the critic cannot
tell.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.dirname(HERE)
sys.path[:0] = [os.path.dirname(ML), ML, HERE]

import train  # noqa: E402

STEMS = ("drums", "bass", "other")


def song_scores(paths: list[str], cache_dir: str, stems: bool = True) -> dict[str, dict[str, float]]:
    """{path: {"mix": z, "drums": z, ...}} for audio files (any format)."""
    import dataset
    import mine_cells
    os.makedirs(cache_dir, exist_ok=True)
    out = {}
    tmp = tempfile.mkdtemp(prefix="critic_score_")
    try:
        for p in paths:
            enc = dataset.lossy(p, os.path.abspath(p)) if p.endswith(".wav") else p
            parts = {"mix": enc}
            if stems:
                folder = os.path.join(tmp, "stems")
                shutil.rmtree(folder, ignore_errors=True)
                mine_cells.split(enc, folder)
                parts.update({s: os.path.join(folder, s + ".wav") for s in STEMS})
            z = train.score_files(parts, os.path.join(tmp, "_c"))
            shutil.rmtree(os.path.join(tmp, "_c"), ignore_errors=True)
            if enc != p:
                os.remove(enc)
            out[p] = z
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def interval(values: np.ndarray, draws: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """The mean and its 95% bootstrap interval."""
    rng = np.random.default_rng(seed)
    v = np.asarray(values, float)
    means = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(draws)]
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compare(a: dict[str, dict[str, float]], b: dict[str, dict[str, float]], pairs: list[tuple[str, str]]) -> dict:
    """Per part, the mean of B - A over the seed pairs ((a_path, b_path)),
    with its interval, and the rule's verdict."""
    res = {}
    for part in ("mix",) + STEMS:
        d = [b[pb][part] - a[pa][part] for pa, pb in pairs if part in a.get(pa, {}) and part in b.get(pb, {})]
        if len(d) >= 3:
            res[part] = interval(np.array(d))
    mix = res.get("mix")
    if mix and mix[1] > 0 and not any(res[s][2] < 0 for s in STEMS if s in res):
        verdict = "keep"
    elif mix and mix[2] < 0:
        verdict = "revert"
    else:
        verdict = "can't tell"
    res["verdict"] = verdict
    return res


# --------------------------------------------------------------------------
# The keep-or-revert score: closeness to the family's records on AST
# --------------------------------------------------------------------------
# The trained critic separates records from riddim perfectly but its order
# among riddim's songs agreed with Ed on 11 of his 19 picks; a song's
# closeness to its family's centroid on the same loudness-blind AST
# embeddings agreed on 14 (ml/critic/ed_agree.py, references/critic). So the
# closeness decides, paired by seed; the critics' stem scores ride along as
# diagnostics.

_CENTROIDS: dict = {}


def centroid(family: str) -> np.ndarray:
    if family not in _CENTROIDS:
        d = train._load(os.path.join(train.CRIT, "ast", family + ".npz"))
        rows = [train._unit(v) for v in d.values() if len(v)]
        _CENTROIDS[family] = train._unit(np.concatenate(rows).mean(0))
    return _CENTROIDS[family]


def closeness(paths: list[str], profile: str, cache_dir: str) -> dict[str, float]:
    """Each song's mean window cosine to its profile's main family centroid
    (AST, loudness-blind)."""
    import embed
    import goal_check as gc
    c = centroid(gc.families_for(profile)[0])
    emb = embed.embed_files(paths, cache_dir, "close", ("ast",))["ast"]
    return {p: float((train._unit(e) @ c).mean()) for p, e in emb.items() if len(e)}


def ab(profile: str, a_changes: list, b_changes: list, seeds, out_dir: str, stems: bool = False) -> dict:
    """Render `profile` on `seeds` with A's changes and with B's (as
    blind_ab.patched takes them), and judge B against A: the mean paired
    closeness difference with its 95% interval, the verdict (keep B when the
    interval sits above 0, revert when below, else can't tell), and, with
    `stems`, the critics' per-part differences."""
    from blind_ab import patched
    from riddim.compose import compose
    from riddim.render import audio
    paths = {}
    for side, changes in (("a", a_changes), ("b", b_changes)):
        os.makedirs(os.path.join(out_dir, side), exist_ok=True)
        paths[side] = []
        for s in seeds:
            f = os.path.join(out_dir, side, f"{profile}_{s}.wav")
            if not os.path.exists(f):
                with patched(changes):
                    audio.write(compose(profile, seed=s), f)
            paths[side].append(f)
    cl = {side: closeness(paths[side], profile, os.path.join(out_dir, "_emb")) for side in paths}
    d = np.array([cl["b"][pb] - cl["a"][pa] for pa, pb in zip(paths["a"], paths["b"])])
    m, lo, hi = interval(d)
    out = {"closeness": (m, lo, hi), "a": float(np.mean(list(cl["a"].values()))),
           "b": float(np.mean(list(cl["b"].values()))),
           "verdict": "keep" if lo > 0 else ("revert" if hi < 0 else "can't tell")}
    if stems:
        sa = song_scores(paths["a"], os.path.join(out_dir, "_critic"))
        sb = song_scores(paths["b"], os.path.join(out_dir, "_critic"))
        out["critic"] = compare(sa, sb, list(zip(paths["a"], paths["b"])))
    return out
