"""Rhythm cells mined from records (ml/mine_cells.py): the bar rhythms real
bass players in each family play, by how many records play them.

The pools are grid positions counted from separated stems, not audio. A
profile with no pool (or no cells.json at all) keeps the generator alone.
"""

from __future__ import annotations

import json
import os
import random
from functools import lru_cache

from .rhythm import Hit, Pattern, metric_weight
from ..units import STEPS_PER_BAR

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cells.json")


@lru_cache(maxsize=None)
def _data(path: str = PATH) -> dict:
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


def _hits(mask: int) -> int:
    return bin(mask).count("1")


def bass_pair(profile: str, rng: random.Random) -> list[Pattern] | None:
    """A verse cell drawn by how many records play it, and a peak cell that
    plays at least as much and keeps the verse's downbeat - the same bassist
    digging in, not a different tune."""
    cells = pool(profile)
    if not cells:
        return None
    masks, weights = zip(*cells)
    verse = rng.choices(masks, weights)[0]
    busier = [(m, w) for m, w in cells if m != verse and _hits(m) >= _hits(verse)
              and (m & 1) == (verse & 1) and _hits(m) <= _hits(verse) + 2]
    peak = rng.choices(*zip(*busier))[0] if busier else verse
    return [pattern(verse), pattern(peak)]
