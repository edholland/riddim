"""Tension and release: the moves that make a drop land.

The energy curve says *how much*; this module says *how it gets there*. A
build that only gets denser is a ramp. A build that rolls - snare in
quarters, then eighths, then sixteenths - with a riser sweeping underneath,
then **stops dead** for a beat before the drop, is an event. Those are the
devices, and each is a knob:

- **roll** - an accelerating snare roll through the last bars of a build.
  Subdivision doubles as the drop approaches; velocity climbs with it.
- **riser** - one long note on the perc kit's riser row (load a noise sweep
  or reverse cymbal there), ending exactly where the gap starts.
- **drop_gap** - beats of silence before the drop. Silence is the loudest
  thing in the arrangement; the downbeat after it is the drop.
- **impact** - a crash on the drop's downbeat, and lighter ones on every
  eighth bar inside a peak, so the long sections keep announcing phrases.
- **switch_up** - mid-peak breaks: at the end of each sixteen inside a long
  peak the band stops for half a bar and lets the lead (or nothing) through.
- **fills** - how often an eight-bar phrase inside a section ends with a
  drum fill rather than the plain pattern.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from ..units import TICKS_PER_BAR, TICKS_PER_BEAT
from .structure import Section, kind_of


@dataclass(frozen=True)
class DynamicsParams:
    roll: float = 0.7
    riser: float = 0.6
    drop_gap: float = 1.0       # in beats: 0, 0.5, 1, 2
    impact: float = 0.8
    switch_up: float = 0.4
    fills: float = 0.6
    # How often the last bars before a drop sit on a tension chord - the
    # dominant, or the phrygian bII - that the drop's downbeat resolves.
    cadence: float = 0.8
    # How much a long drop unfolds: layers held back and brought in by
    # eight-bar block instead of one arrangement for thirty-two bars.
    layering: float = 0.6
    # How likely the last drop is lifted into a new key - up a semitone or
    # a tone, the approach bars turning into the new key's dominant so the
    # lift lands as a modulation, not a jump. Everything from there to the
    # end stays up.
    lift: float = 0.0


def params_for(family: str, rng: random.Random) -> DynamicsParams:
    """Draw dynamics for a genre family. Dubstep commits harder to the drop."""
    if family == "dubstep":
        # The 140 drop is earned by what is held back (the low end), not by
        # an EDM snare roll and white-noise riser: the records have short
        # rolls if any, and the riser is rare.
        return DynamicsParams(
            roll=round(rng.uniform(0.1, 0.45), 2),
            riser=round(rng.uniform(0.0, 0.35), 2),
            drop_gap=rng.choice((0.5, 1.0, 1.0, 2.0)),
            impact=round(rng.uniform(0.7, 1.0), 2),
            switch_up=round(rng.uniform(0.1, 0.4), 2),
            fills=round(rng.uniform(0.2, 0.5), 2),
            cadence=round(rng.uniform(0.6, 1.0), 2),
            layering=round(rng.uniform(0.4, 0.9), 2),
            lift=0.15,
        )
    if family == "garage":
        # Garage builds less and stops more: shorter rolls, the bass and
        # band cutting out mid-drop (the switch-up is the garage trick),
        # and the gap before the drop as often a half-beat as a full one.
        return DynamicsParams(
            roll=round(rng.uniform(0.3, 0.7), 2),
            riser=round(rng.uniform(0.3, 0.7), 2),
            drop_gap=rng.choice((0.5, 0.5, 1.0, 2.0)),
            impact=round(rng.uniform(0.5, 0.85), 2),
            switch_up=round(rng.uniform(0.4, 0.8), 2),
            fills=round(rng.uniform(0.3, 0.6), 2),
            cadence=round(rng.uniform(0.5, 0.9), 2),
            layering=round(rng.uniform(0.4, 0.8), 2),
            lift=0.35,
        )
    return DynamicsParams(
        roll=round(rng.uniform(0.2, 0.6), 2),
        riser=round(rng.uniform(0.0, 0.4), 2),
        drop_gap=rng.choice((0.0, 0.5, 1.0, 1.0)),
        impact=round(rng.uniform(0.3, 0.7), 2),
        switch_up=round(rng.uniform(0.05, 0.3), 2),
        fills=round(rng.uniform(0.2, 0.5), 2),
        cadence=round(rng.uniform(0.4, 0.9), 2),
        layering=round(rng.uniform(0.3, 0.8), 2),
        lift=0.3,
    )


def drop_entry(prev: Section | None, section: Section) -> bool:
    """True when `section` is a drop: a peak arrived at by a real climb."""
    if prev is None:
        return False
    return kind_of(section) == "peak" and section.energy - prev.energy >= 0.25


def gap_ticks(params: DynamicsParams) -> int:
    beats = params.drop_gap
    if beats <= 0:
        return 0
    return int(round(min(4.0, beats) * TICKS_PER_BEAT))


def roll_bars(params: DynamicsParams, section: Section) -> int:
    """How many bars at the end of a build the roll covers."""
    if params.roll <= 0.05:
        return 0
    return max(1, min(section.bars, int(round(1 + params.roll * 3))))


def roll(bars_left: int, total: int, params: DynamicsParams,
         rng: random.Random, gap_steps: int = 0) -> list[tuple[float, int]]:
    """The snare roll for one bar, as (step, velocity).

    `bars_left` counts down to the drop (1 = the last bar). Subdivision
    doubles as it approaches - quarters, eighths, sixteenths, and the final
    beat in thirty-seconds at full roll - and velocity climbs across the
    whole roll, not per bar, so it is one gesture.
    """
    if bars_left > total or total <= 0:
        return []
    if bars_left >= 3:
        every = 4.0
    elif bars_left == 2:
        every = 2.0
    else:
        every = 1.0
    end = 16 - gap_steps
    hits = []
    s = 0.0
    while s < end:
        progress = 1.0 - (bars_left - 1 + (1 - s / 16)) / total
        vel = int(52 + 70 * max(0.0, min(1.0, progress)) * (0.6 + 0.4 * params.roll))
        hits.append((s, max(1, min(127, vel + rng.randint(-4, 4)))))
        step = every
        if bars_left == 1 and s >= 14 and params.roll >= 0.85:
            step = 0.5
        s += step
    return hits


def riser_span(section: Section, params: DynamicsParams, gap: int) -> tuple[int, int] | None:
    """(start_tick, length_ticks) of the riser in a build, or None."""
    if params.riser <= 0.05:
        return None
    bars = max(1, min(section.bars, int(round(2 + params.riser * 6))))
    end = section.start_tick + section.length_ticks - gap
    start = end - bars * TICKS_PER_BAR
    start = max(section.start_tick, start)
    if end - start < TICKS_PER_BAR // 2:
        return None
    return start, end - start


def switch_windows(section: Section, params: DynamicsParams,
                   rng: random.Random) -> list[tuple[int, int]]:
    """Half-bar band stops inside a long peak: (start_tick, end_tick).

    At eight-bar boundaries (sixteens twice as likely), never at the end of
    the section - that belongs to the fill and the next section's entry.
    """
    if kind_of(section) != "peak" or section.bars < 16 or params.switch_up <= 0:
        return []
    out = []
    for block_end in range(8, section.bars, 8):
        chance = params.switch_up * (1.0 if block_end % 16 == 0 else 0.5)
        if rng.random() < chance:
            end = section.start_tick + block_end * TICKS_PER_BAR
            out.append((end - TICKS_PER_BAR // 2, end))
    return out
