"""Hand percussion: the congas, bongos, toms and wood that fill a sound
system record's middle.

A drum machine's kit is the frame; the hand percussion is what makes it
breathe. Listen to a Mala intro and before any kick or sub there is a whole
rhythm section of hands - a low conga answering the bar's end, a high one
chattering in ghost notes, a bongo running eighths, a woodblock playing a
clave, a run of toms falling into the next phrase. It is dense, it is never
loud, and it is the reason the record sounds played rather than programmed.
Measured on the DMZ catalogue: their intros carry the 200 Hz - 10 kHz band
within a few dB of the body while the kick and sub are still out, and most of
that is percussion.

The voices, each with its own job:

    conga_lo   the tumba - two or three open tones leaning into the next
               downbeat, never on the backbeat (that is the snare's)
    conga_hi   the slap on its spot, heel-and-toe ghosts around it
    bongo      the martillo: running eighths, the fourth of each group hit
    block      a clave: son or rumba 3-2, the tresillo, or its mirror
    tamb       the offbeat eighths, light
    tom_lo/hi  a run at the end of a phrase, falling into the next

A pattern is two bars, the same each time round - hands repeat a part, the
way the kit does - with the last bar of every eight given to a call: the toms'
run, or the high conga picking up into the one. Pure: degrees of nothing,
steps and velocities only.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

STEPS = 32           # a two-bar pattern, in sixteenths

VOICES = ("conga_lo", "conga_hi", "bongo", "block", "tamb", "tom_lo", "tom_hi")

# Claves over two bars of sixteenths: the sound system's oldest rhythms.
CLAVES = {
    "son_3_2": (0, 6, 12, 20, 24),
    "rumba_3_2": (0, 6, 14, 20, 24),
    "son_2_3": (4, 8, 16, 22, 28),
    "tresillo": (0, 6, 12, 16, 22, 28),
    "dotted": (0, 6, 12, 16, 22, 26),
}


@dataclass(frozen=True)
class PercParams:
    """How the hands play. `voices` is how many of the hand voices the song
    uses (the ensemble), `density` how busy each is, `ghosts` how many of
    the soft in-between strokes a hand drummer plays, `runs` how often a
    phrase ends in a tom run, `clave` the woodblock's rhythm."""

    voices: tuple[str, ...] = ("conga_lo", "conga_hi", "bongo")
    density: float = 0.5
    ghosts: float = 0.5
    runs: float = 0.5
    clave: str = "son_3_2"


def params_for(family: str, rng: random.Random) -> PercParams:
    """Ranges, not presets. Dubstep and dub lean on congas and wood; garage
    on the tambourine and bongo over its shaker."""
    if family == "garage":
        pool = ("bongo", "tamb", "conga_hi", "block", "conga_lo")
    else:
        pool = ("conga_lo", "conga_hi", "bongo", "block", "tamb")
    k = rng.choice((2, 3, 3, 4))
    chosen = list(pool[:2]) + rng.sample(list(pool[2:]), max(0, k - 2))
    return PercParams(
        voices=tuple(v for v in VOICES if v in chosen) + ("tom_lo", "tom_hi"),
        density=round(rng.uniform(0.35, 0.8), 2),
        ghosts=round(rng.uniform(0.3, 0.8), 2),
        runs=round(rng.uniform(0.3, 0.8), 2),
        clave=rng.choice(tuple(CLAVES)),
    )


def _backbeat(beat_steps: int) -> set[int]:
    """Where the snare speaks in two bars: 2 and 4 straight, 3 in half-time."""
    if beat_steps >= 8:
        return {8, 24}
    return {4, 12, 20, 28}


def pattern(p: PercParams, beat_steps: int, energy: float, rng: random.Random) -> dict:
    """{voice: [(step, velocity), ...]} over two bars, at a section's energy.
    The toms are not here - they play the calls (`call`)."""
    back = _backbeat(beat_steps)
    busy = max(0.0, min(1.0, p.density * (0.45 + 0.75 * energy)))
    out: dict[str, list[tuple[float, int]]] = {}

    if "conga_lo" in p.voices:
        # Open tones leaning into the next downbeat: the bar's last beat,
        # and the and-of-four - one bar plain, the next answered.
        spots = [14, 30] + ([15, 31] if busy > 0.45 else []) + ([11] if busy > 0.7 else [])
        out["conga_lo"] = [(float(s), 92 if s in (14, 30) else 78) for s in sorted(spots)
                           if s not in back]

    if "conga_hi" in p.voices:
        # The slap on an offbeat next to the backbeat; heel-and-toe ghosts
        # filling the eighths around it.
        slap = {(b + 2) % STEPS for b in back}
        hits = [(float(s), 100) for s in sorted(slap)]
        for s in range(0, STEPS, 2):
            if s in slap or s in back:
                continue
            if rng.random() < p.ghosts * busy:
                hits.append((float(s), rng.randint(30, 46)))
        out["conga_hi"] = sorted(hits)

    if "bongo" in p.voices:
        # Martillo: eighths, the fourth of every group of four struck open,
        # the rest light; at low energy only the open ones.
        hits = []
        for s in range(0, STEPS, 2):
            accent = s % 8 == 6
            if accent:
                hits.append((float(s), 88))
            elif busy > 0.35 and rng.random() < 0.55 + 0.4 * busy:
                hits.append((float(s), rng.randint(40, 60)))
            if busy > 0.75 and accent and rng.random() < p.ghosts:
                hits.append((float(s + 1), 44))       # the double
        out["bongo"] = hits

    if "block" in p.voices:
        out["block"] = [(float(s), 84 if i == 0 else 72)
                        for i, s in enumerate(CLAVES.get(p.clave, CLAVES["son_3_2"]))]

    if "tamb" in p.voices:
        hits = [(float(s), 62) for s in range(2, STEPS, 4)]
        if busy > 0.6:
            hits += [(float(s), 34) for s in range(3, STEPS, 4)]
        out["tamb"] = sorted(hits)
    return out


def call(p: PercParams, rng: random.Random) -> dict:
    """The last bar of a phrase (16 steps): a tom run falling into the next
    downbeat - high, high, low, low, the last one held open - or, when the
    phrase does not run, the high conga picking up into the one - on the
    sixteenth grid, where the Deluge's swing can move it with the kit."""
    if "tom_hi" in p.voices and rng.random() < p.runs:
        start = rng.choice((8, 10, 12))
        steps = list(range(start, 16))
        n = len(steps)
        hits_hi, hits_lo = [], []
        for i, s in enumerate(steps):
            v = int(70 + 30 * i / max(1, n - 1))
            (hits_hi if i < n / 2 else hits_lo).append((float(s), v))
        return {"tom_hi": hits_hi, "tom_lo": hits_lo}
    if "conga_hi" in p.voices:
        return {"conga_hi": [(13.0, 66), (14.0, 82), (15.0, 98)]}
    return {}
