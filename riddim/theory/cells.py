"""Bass rhythms fitted to records (ml/mine_cells.py): how often the bass
players in each family strike each step of the bar, and how many notes they
play a bar, read from separated stems of 40 records a family.

Exact bar rhythms repeat across records too rarely to draw from (the most
shared roots bar is played by 2 records of 80), so a bar is drawn from the
fit instead: a note count from the family's spread, the steps by how often
the records strike them. The data is grid positions, not audio. A profile
with no fit (or no cells.json, or RIDDIM_CELLS=off) keeps the generator.
"""

from __future__ import annotations

import json
import os
import random
from functools import lru_cache

from .rhythm import Hit, Pattern, metric_weight
from ..units import STEPS_PER_BAR

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cells.json")


def _data(path: str = PATH) -> dict:
    if os.environ.get("RIDDIM_CELLS", "").lower() == "off":
        return {}
    return _read(path)


@lru_cache(maxsize=None)
def _read(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def pool(profile: str, kind: str = "bass") -> list[tuple[int, int]]:
    """[(mask, records), ...] for a profile, most played first."""
    return [(int(m), int(n)) for m, n in _data().get(profile, {}).get(kind, [])]


def stats(profile: str) -> dict:
    return dict(_data().get(profile, {}).get("stats", {}))


def pattern(mask: int, velocity: tuple[int, int] = (88, 112), note_length: float = 1.0) -> Pattern:
    """A mask as one bar, accented by metric weight like groove.generate."""
    lo, hi = velocity
    hits = []
    for step in range(STEPS_PER_BAR):
        if mask >> step & 1:
            w = metric_weight(step, STEPS_PER_BAR) / 4.0
            hits.append(Hit(float(step), int(round(lo + (hi - lo) * (0.35 + 0.65 * w))), note_length))
    return Pattern(STEPS_PER_BAR, hits, STEPS_PER_BAR)


def _draw(weights: dict[int, float], n: int, rng: random.Random) -> set[int]:
    """`n` steps without replacement, each by its weight."""
    out: set[int] = set()
    pool = {k: w for k, w in weights.items() if w > 0}
    while pool and len(out) < n:
        k = rng.choices(list(pool), list(pool.values()))[0]
        out.add(k)
        del pool[k]
    return out


def _mask(steps) -> int:
    m = 0
    for k in steps:
        m |= 1 << k
    return m


def bass_pair(profile: str, rng: random.Random, most: int = 8) -> list[Pattern] | None:
    """A verse bar drawn from the profile's fit, and a peak bar that keeps
    it and adds a note or two the records also strike - the same bassist
    digging in, not a different tune."""
    s = stats(profile)
    shares, counts = s.get("steps"), s.get("notes")
    if not shares or not counts:
        return None
    sizes = list(range(1, min(most, len(counts) - 1) + 1))
    n = rng.choices(sizes, [counts[k] for k in sizes])[0] if sum(counts[k] for k in sizes) else 4
    weights = {k: float(w) for k, w in enumerate(shares)}
    verse = _draw(weights, n, rng)
    extra = _draw({k: w for k, w in weights.items() if k not in verse},
                  min(rng.choice((1, 2)), most - len(verse)), rng)
    return [pattern(_mask(verse)), pattern(_mask(verse | extra))]
