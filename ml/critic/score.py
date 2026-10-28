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
