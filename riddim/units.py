"""Timing units.

The Deluge internal grid is 24 ticks per 16th note, which gives 384 ticks per
4/4 bar. Every duration in this package is expressed in those ticks so that
nothing has to be converted on the way out to the hardware.
"""

TICKS_PER_16TH = 24
TICKS_PER_BEAT = TICKS_PER_16TH * 4  # 96
TICKS_PER_BAR = TICKS_PER_BEAT * 4  # 384

STEPS_PER_BAR = 16


def bars(n: float) -> int:
    """Bars to ticks."""
    return int(round(n * TICKS_PER_BAR))


def beats(n: float) -> int:
    """Beats to ticks."""
    return int(round(n * TICKS_PER_BEAT))


def steps(n: float) -> int:
    """16th-note steps to ticks."""
    return int(round(n * TICKS_PER_16TH))


def step_of(tick: int) -> int:
    """Which 16th-note step a tick falls on, counted from the start."""
    return tick // TICKS_PER_16TH


def beat_of(tick: int) -> int:
    return tick // TICKS_PER_BEAT


def bar_of(tick: int) -> int:
    return tick // TICKS_PER_BAR


def position_in_bar(tick: int) -> int:
    return tick % TICKS_PER_BAR
