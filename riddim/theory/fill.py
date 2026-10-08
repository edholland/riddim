"""Drum fills: the last bar of a phrase says what is about to happen.

A fill has one job - hand the barline over with momentum. Its shape is
specified by the energy of the section ending, the same way the rest of the
engine specifies things: a hole ends on a couple of soft rims, a build ends
on a snare run of climbing eighths, and a peak ends on sixteenth notes with
a velocity arc that hits the downbeat hardest. Nothing here is a table of
named rolls; the arc, the step size and the voice all follow the energy
number, which is what a fill actually *means*.
"""

from __future__ import annotations

import random

from .rhythm import Hit, Pattern
from ..units import STEPS_PER_BAR


def make(
    energy: float,
    rng: random.Random | None = None,
    steps: int = STEPS_PER_BAR,
) -> dict[str, Pattern]:
    """One bar of fill by energy, as {voice: Pattern}.

    All voices here land in the second half of the bar: a fill is a run *into*
    the next downbeat, and hitting the first half would pre-empt the phrase it
    is supposed to point at.
    """
    rng = rng or random.Random()
    half = steps // 2

    if energy < 0.35:
        # A hole is coming. No flourish - a couple of soft rims, the last one
        # landing the hardest, and then the drop.
        hits = [
            Hit(float(s), int(55 + rng.randint(0, 25)), 1.0, "rim")
            for s in (half + 2, half + 4, half + 6)
        ]
        hits[-1] = Hit(float(steps - 2), 85, 1.0, "rim")
        return {"rim": Pattern(steps, hits, steps, "fill")}

    if energy < 0.7:
        # Eighths with a climbing velocity arc - a roll that gains weight.
        hits = _run(rng, half, steps, 2, 78, 115)
        return {"snare": Pattern(steps, hits, steps, "fill")}

    # A peak is coming: sixteenths all the way, the last hit loudest, with
    # the rim doubling the penultimate beat for the big thwack.
    hits = _run(rng, half, steps, 1, 85, 120)
    rim = [
        Hit(float(s), int(90 + rng.randint(0, 20)), 1.0, "rim")
        for s in (steps - 4, steps - 2)
    ]
    return {
        "snare": Pattern(steps, hits, steps, "fill"),
        "rim": Pattern(steps, rim, steps, "fill"),
    }


def _run(
    rng: random.Random,
    start: int,
    end: int,
    step_size: int,
    vel_lo: int,
    vel_hi: int,
) -> list[Hit]:
    """A run of hits from `start` to `end`, velocity climbing the arc."""
    hits = []
    s = start
    while s < end:
        pos = (s - start) / max(1, end - start - 1)
        vel = int(round(vel_lo + (vel_hi - vel_lo) * pos)) + rng.randint(-5, 5)
        hits.append(Hit(float(s), max(1, min(127, vel)), 1.0))
        s += step_size
    return hits
