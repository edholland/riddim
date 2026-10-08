"""Modal progressions for dub and dubstep.

This music is not functional-harmony music. There is no ii-V-I; there is a
vamp, and the interest comes from rhythm, space and timbre. So the progression
library here is deliberately small and modal: one- and two-chord riddims,
the bVII/bVI moves that define minor-key dub, and the Andalusian descent that
does most of the heavy lifting in dark dubstep.

Degrees are 0-indexed scale degrees, so 0 is the tonic and 6 is the seventh
degree (bVII in a minor scale).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from . import chord as chord_mod
from .chord import Chord, ChordEvent
from .scale import Scale
from ..units import TICKS_PER_BAR


@dataclass(frozen=True)
class Progression:
    """A named cycle of scale degrees."""

    name: str
    degrees: tuple[int, ...]
    modes: tuple[str, ...] = ("minor", "dorian")
    tension: float = 0.5  # 0 = restful vamp, 1 = maximum unease
    note: str = ""
    # Degrees whose third is raised a semitone out of the scale, borrowing
    # from harmonic minor. This is how a minor key gets a major V: the raised
    # leading tone is a chromatic alteration, not a change of key, which is
    # why it is a property of the progression and not of the scale.
    major_degrees: tuple[int, ...] = ()

    def __len__(self) -> int:
        return len(self.degrees)


# The library. `tension` is what the arranger uses to match a progression to a
# section's energy, so these ratings matter more than they look.
PROGRESSIONS: tuple[Progression, ...] = (
    Progression(
        "i_V", (0, 4), ("harmonic_minor",), 0.7,
        "Am-E. In harmonic minor the raised 7th makes the V a major chord, so "
        "this is a real dominant pulling back to the tonic - the only place "
        "in this library where that happens. Heavy and classical-dark.",
    ),
    Progression(
        "i_bVI_V", (0, 5, 4), ("harmonic_minor",), 0.85,
        "Am-F-E. The flamenco cadence. The gap between the bVI and the major "
        "V is an augmented second, which is the sound people mean by "
        "'Middle Eastern' - and by 'menacing dubstep'.",
    ),
    Progression(
        "tonic_drone", (0,), ("minor", "dorian", "phrygian", "harmonic_minor"), 0.05,
        "One chord for the whole tune. The most authentic roots option - all "
        "movement comes from the riddim and the desk.",
    ),
    Progression(
        "i_bVII", (0, 6), ("minor", "dorian"), 0.2,
        "Dm-C. The default minor-key rock/dub rocking motion.",
    ),
    Progression(
        "i_iv", (0, 3), ("minor", "dorian", "harmonic_minor"), 0.25,
        "Dm-Gm. Plagal, heavy, sits still. Good under a busy riddim.",
    ),
    Progression(
        "i_bVI", (0, 5), ("minor", "harmonic_minor"), 0.4,
        "Dm-Bb. Lifts without resolving; the bVI is the brightest note in "
        "natural minor.",
    ),
    Progression(
        "i_bII", (0, 1), ("phrygian", "phrygian_dominant"), 0.7,
        "Dm-Eb. The Phrygian half-step. Instantly menacing - this is the "
        "dubstep move.",
    ),
    Progression(
        "phrygian_i_bvii", (0, 6), ("phrygian",), 0.55,
        "Gm-Fm. The bvii in phrygian is *minor* - a whole step down to "
        "another minor chord, which is darker and heavier than the major "
        "bVII of natural minor.",
    ),
    Progression(
        "phrygian_i_bIII_bII", (0, 2, 1), ("phrygian",), 0.75,
        "Gm-Bb-Ab. Up to the relative major, then the half-step slide back "
        "down onto the bII - the tension comes from landing a semitone above "
        "home and refusing to resolve.",
    ),
    Progression(
        "phrygian_i_iv_bII", (0, 3, 1), ("phrygian",), 0.65,
        "Gm-Cm-Ab. Plagal weight, then the Phrygian half-step. Two minor "
        "chords and one major - the major one is the surprise.",
    ),
    Progression(
        "phrygian_i_bVI_bvii", (0, 5, 6), ("phrygian",), 0.5,
        "Gm-Eb-Fm. Rises through the bVI to a minor bvii and falls back. "
        "The phrygian answer to the i-bVI-bVII dub turnaround.",
    ),
    Progression(
        "i_bVI_bVII", (0, 5, 6), ("minor",), 0.45,
        "Dm-Bb-C. Classic three-chord dub turnaround.",
    ),
    Progression(
        "i_bVII_bVI_bVII", (0, 6, 5, 6), ("minor",), 0.4,
        "Rocking back and forth without ever leaving home.",
    ),
    Progression(
        "andalusian", (0, 6, 5, 4), ("minor",), 0.75,
        "i-bVII-bVI-v. The descending tetrachord, all diatonic. Relentless, "
        "and it never resolves - it just falls forever.",
    ),
    Progression(
        "andalusian_cadence", (0, 6, 5, 4), ("minor",), 0.85,
        "i-bVII-bVI-V with a major V. Raising the third of the last chord "
        "borrows the leading tone from harmonic minor and turns the descent "
        "into a real cadence. This is the flamenco/dark-dubstep sound.",
        major_degrees=(4,),
    ),
    Progression(
        "i_bIII_bVII_iv", (0, 2, 6, 3), ("minor", "dorian"), 0.55,
        "Dm-F-C-Gm. More movement; use where a section needs to travel.",
    ),
    Progression(
        # Not phrygian: the v triad there is diminished, which is no place
        # for a vamp to sit.
        "i_v", (0, 4), ("minor", "dorian"), 0.3,
        "Dm-Am. Modal and hollow, especially with quartal voicings.",
    ),
    Progression(
        "dorian_i_IV", (0, 3), ("dorian",), 0.35,
        "Dm-G. The major IV is what makes Dorian sound like Dorian - the "
        "natural 6th. Bright-dark, very Massive Attack.",
    ),
)

BY_NAME = {p.name: p for p in PROGRESSIONS}


def is_safe_in(progression: Progression, mode: str) -> bool:
    """True if every chord of the progression is stable in `mode`.

    "Stable" means no diminished or augmented triad: those cannot hold down a
    bar of a vamp, they are passing chords. This is computed rather than
    trusted from `modes`, because the declared list is a musical preference
    and this is a correctness floor - and because a progression picked through
    a fallback path has no declaration protecting it at all.
    """
    scale = Scale(45, mode)
    for degree in progression.degrees:
        built = chord_mod.stacked(scale, degree, size=3)
        if degree in progression.major_degrees:
            built = chord_mod.with_major_third(built)
        if built.quality in ("dim", "aug", "?"):
            return False
    return True


def for_mode(mode: str) -> list[Progression]:
    """Progressions that make sense in a given mode, and are safe in it."""
    return [p for p in PROGRESSIONS if mode in p.modes and is_safe_in(p, mode)]


def choose(
    mode: str,
    tension: float = 0.5,
    rng: random.Random | None = None,
    tolerance: float = 0.25,
) -> Progression:
    """Pick a progression for a mode at roughly the requested tension.

    Falls back widthways, never downwards: preferred progressions first, then
    any progression that is merely *safe* in the mode. It must never return
    something that puts a diminished chord under a vamp, which is what an
    unguarded "fall back to everything" did for harmonic minor.
    """
    rng = rng or random.Random()
    pool = for_mode(mode)
    if not pool:
        pool = [p for p in PROGRESSIONS if is_safe_in(p, mode)]
    if not pool:
        raise ValueError(
            f"no progression in the library is harmonically safe in {mode!r}"
        )
    near = [p for p in pool if abs(p.tension - tension) <= tolerance]
    return rng.choice(near or pool)


@dataclass
class HarmonyPlan:
    """A progression realised as timed chords."""

    scale: Scale
    progression: Progression
    events: list[ChordEvent] = field(default_factory=list)

    def at(self, tick: int) -> ChordEvent | None:
        """The chord sounding at a given tick."""
        for ev in self.events:
            if ev.start <= tick < ev.end:
                return ev
        return None

    def symbols(self) -> list[str]:
        return [ev.chord.symbol() for ev in self.events]

    def offset(self, ticks: int) -> "HarmonyPlan":
        """A copy of this plan moved `ticks` along the timeline.

        Used to lay one plan per section: each section's plan is realised at
        the origin, then shifted to the section's absolute start bar.
        """
        return HarmonyPlan(
            self.scale, self.progression,
            [ChordEvent(ev.chord, ev.start + ticks, ev.length,
                         ev.label, dict(ev.meta)) for ev in self.events],
        )


def merge_section_plans(
    plans: list[HarmonyPlan],
    core: Progression | None = None,
) -> HarmonyPlan:
    """Flatten one plan per section onto a single timeline.

    Events are already at absolute positions, so the merge is a plain
    concatenation. The merged plan reports `core` - the tune's base
    progression - rather than whatever happens to sit in the first section,
    because first sections are usually holes (a drone), not the tune.
    """
    if not plans:
        raise ValueError("no section plans to merge")
    return HarmonyPlan(
        plans[0].scale,
        core or plans[0].progression,
        [ev for plan in plans for ev in plan.events],
    )


def realise(
    scale: Scale,
    progression: Progression,
    bars: int,
    bars_per_chord: float = 1.0,
    size: int = 3,
    skip: int = 2,
) -> HarmonyPlan:
    """Lay a progression out over `bars`, cycling it as needed.

    `bars_per_chord` is the harmonic rhythm. Dub wants this slow - one or two
    bars per chord - because fast chord changes fight the riddim.
    """
    if bars <= 0:
        raise ValueError(f"bars must be positive, got {bars}")
    if bars_per_chord <= 0:
        raise ValueError(f"bars_per_chord must be positive, got {bars_per_chord}")

    ticks_per_chord = int(round(bars_per_chord * TICKS_PER_BAR))
    total_ticks = bars * TICKS_PER_BAR

    events: list[ChordEvent] = []
    tick = 0
    i = 0
    while tick < total_ticks:
        degree = progression.degrees[i % len(progression.degrees)]
        built = chord_mod.stacked(scale, degree, size=size, skip=skip)
        if degree in progression.major_degrees:
            built = chord_mod.with_major_third(built)
        length = min(ticks_per_chord, total_ticks - tick)
        events.append(ChordEvent(built, tick, length, label=built.symbol()))
        tick += length
        i += 1

    return HarmonyPlan(scale, progression, events)


def voiced(
    plan: HarmonyPlan,
    low: int,
    high: int,
    rootless: bool = True,
    centre: int | None = None,
    first: Chord | None = None,
) -> HarmonyPlan:
    """Apply voice leading to a plan, optionally dropping roots.

    Rootless is the default because the bass part owns the root. Doubling it
    in the chords is the single most common way these arrangements turn to
    mud.

    `first` is the voicing the *previous* section ended on, so that a section
    change - including a change of progression - continues by the smallest
    move instead of every voice jumping back to home.
    """
    from .voicing import lead

    chords = [ev.chord for ev in plan.events]
    if rootless:
        # Only worth doing to a seventh chord or bigger. Stripping the root
        # from a triad leaves two notes, which is too thin to state the
        # harmony - and the mud it was meant to prevent is avoided anyway by
        # keeping the chord register well above the sub.
        chords = [chord_mod.rootless(c) if len(c.pitches) >= 4 else c for c in chords]

    led = lead(chords, low, high, centre=centre, first=first)
    events = [
        ChordEvent(new, ev.start, ev.length, label=ev.label, meta=dict(ev.meta))
        for ev, new in zip(plan.events, led)
    ]
    return HarmonyPlan(plan.scale, plan.progression, events)


def bass_root(scale: Scale, degree: int, register_low: int, register_high: int) -> int:
    """The root note for a degree, placed in the bass register."""
    from . import pitch

    return pitch.transpose_into(scale.note(degree), register_low, register_high)


# --------------------------------------------------------------------------
# Generative: a root-motion grammar
# --------------------------------------------------------------------------

# How much each scale degree pulls away from home, for the minor-family
# modes this music lives in. The bII is the most tense place a vamp can sit;
# the iv barely leaves.
_DEGREE_TENSION = {0: 0.0, 1: 0.9, 2: 0.35, 3: 0.3, 4: 0.6, 5: 0.45, 6: 0.4}

# Root motion by scale steps, and how idiomatic each is. Step motion (the
# i-bVII rock) and fourth/fifth motion (plagal, dominant) are the strong
# ones; thirds are mediant slides; sixths are the i-bVI lift.
_MOTION_WEIGHT = {1: 1.0, 6: 1.0, 3: 0.9, 4: 0.9, 2: 0.5, 5: 0.5}

_ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII"]
_MAJOR_INTERVALS = (0, 2, 4, 5, 7, 9, 11)


def roman(scale: Scale, degree: int, chord: Chord | None = None) -> str:
    """Roman-numeral name of a degree in a scale: 'i', 'bVII', 'V', 'iv'.

    Lower case for minor, upper for major, a flat sign when the degree is
    lowered relative to the major scale. This is real analysis, not a
    lookup, so it names chords the library never listed.
    """
    index = degree % 7
    numeral = _ROMAN[index]
    semis = scale.intervals[index]
    flat = "b" if semis < _MAJOR_INTERVALS[index] else ""
    if chord is None:
        chord = chord_mod.triad(scale, degree)
    q = chord.quality
    if q in ("m", "m7", "dim", "m7b5"):
        numeral = numeral.lower()
    suffix = "°" if q.startswith("dim") else ""
    return f"{flat}{numeral}{suffix}"


def _safe_degrees(scale: Scale) -> list[int]:
    out = []
    for d in range(scale.size):
        if chord_mod.triad(scale, d).quality not in ("dim", "aug", "?"):
            out.append(d)
    return out


def _tension_of(degrees: tuple[int, ...], major_degrees: tuple[int, ...]) -> float:
    """Tension of a cycle: mostly how far its non-tonic chords sit from
    home, plus a little for how much it moves, plus a lift for a borrowed
    dominant. A drone is 0.05; i-bII is about 0.7; nothing exceeds ~0.85."""
    if len(degrees) <= 1:
        return 0.05
    away = [_DEGREE_TENSION.get(d % 7, 0.5) for d in degrees if d % 7 != 0]
    base = sum(away) / max(1, len(away))
    motion = (len(degrees) - 1) / 3.0
    lift = 0.1 if major_degrees else 0.0
    return min(1.0, 0.7 * base + 0.25 * motion + lift)


def generate_progression(
    scale: Scale,
    tension: float = 0.5,
    rng: random.Random | None = None,
    length: int | None = None,
    tolerance: float = 0.18,
    tries: int = 200,
) -> Progression:
    """Generate a modal vamp by walking root motions from the tonic.

    Starts on i, moves by weighted root intervals through degrees whose
    triads are stable in the mode, and closes the cycle so the last chord
    has strong motion back to i. Candidates are scored for tension and the
    first within `tolerance` of the target is returned; the closest is
    returned if none is.

    In minor, a cycle ending on v may borrow the leading tone and become V -
    the harmonic-minor cadence - which is a chromatic alteration of one
    chord, not a change of mode.
    """
    rng = rng or random.Random()
    safe = _safe_degrees(scale)
    if 0 not in safe:
        raise ValueError(f"tonic triad is unstable in {scale}; not a usable key")

    best: tuple[float, Progression] | None = None
    for _ in range(tries):
        if length is not None:
            n = length
        elif tension < 0.2:
            n = rng.choices([1, 2, 3], [4, 3, 1])[0]     # low tension: short vamps
        else:
            n = rng.choices([2, 3, 4], [3, 3, 2])[0]
        degrees = [0]
        ok = True
        for i in range(1, n):
            current = degrees[-1]
            options = []
            for d in safe:
                if d == current:
                    continue
                if i == n - 1 and d == 0:
                    continue  # cycle already returns to i; don't end on it
                interval = (d - current) % 7
                w = _MOTION_WEIGHT.get(interval, 0.3)
                # Prefer not to revisit within a cycle, and strongly avoid
                # touching the tonic mid-cycle - that reads as two vamps
                # glued together rather than one progression.
                if d == 0:
                    w *= 0.05
                elif d in degrees:
                    w *= 0.15
                options.append((d, w))
            if not options:
                ok = False
                break
            ds, ws = zip(*options)
            degrees.append(rng.choices(ds, ws)[0])
        if not ok:
            continue
        # Close the cycle: the last chord should move well back to i.
        if n > 1:
            back = (0 - degrees[-1]) % 7
            if _MOTION_WEIGHT.get(back, 0.3) < 0.5 and rng.random() < 0.7:
                continue

        major_degrees: tuple[int, ...] = ()
        if (scale.name in ("minor", "aeolian") and n > 1 and degrees[-1] == 4
                and rng.random() < 0.5):
            major_degrees = (4,)

        prog = Progression(
            name="-".join(
                roman(scale, d,
                      chord_mod.with_major_third(chord_mod.triad(scale, d))
                      if d in major_degrees else None)
                for d in degrees
            ),
            degrees=tuple(degrees),
            modes=(scale.name,),
            tension=_tension_of(tuple(degrees), major_degrees),
            major_degrees=major_degrees,
            note="generated",
        )
        if not is_safe_in(prog, scale.name):
            continue
        err = abs(prog.tension - tension)
        if err <= tolerance:
            return prog
        if best is None or err < best[0]:
            best = (err, prog)
    if best is None:
        return BY_NAME["tonic_drone"]
    return best[1]


def enumerate_progressions(scale: Scale, max_length: int = 4) -> list[Progression]:
    """Every idiomatic cycle the grammar can produce in a scale, for a menu.

    All cycles from i of one to three chords over the stable degrees, plus
    the four-chord cycles whose every root motion is a strong one (step,
    fourth or fifth). Each is named by analysis and carries its tension, and
    the list is sorted by tension so a menu reads from restful to tense.
    A cycle ending on v in natural minor is listed twice: as v and as the
    borrowed major V.
    """
    safe = [d for d in _safe_degrees(scale) if d != 0]
    out: list[Progression] = []
    seen: set[tuple] = set()

    def add(degrees, majors=()):
        key = (degrees, majors)
        if key in seen:
            return
        prog = Progression(
            name="-".join(
                roman(scale, d, chord_mod.with_major_third(chord_mod.triad(scale, d))
                      if d in majors else None)
                for d in degrees
            ),
            degrees=degrees, modes=(scale.name,),
            tension=_tension_of(degrees, majors), major_degrees=majors,
            note="grammar",
        )
        if is_safe_in(prog, scale.name):
            seen.add(key)
            out.append(prog)

    def walk(prefix, length):
        if len(prefix) == length:
            add(tuple(prefix))
            if (scale.name in ("minor", "aeolian") and prefix[-1] == 4 and length > 1):
                add(tuple(prefix), (4,))
            return
        for d in safe:
            if d == prefix[-1]:
                continue
            if length == 4:
                motion = _MOTION_WEIGHT.get((d - prefix[-1]) % 7, 0.3)
                if motion < 0.5 or d in prefix:
                    continue
            walk(prefix + [d], length)

    add((0,))
    for length in range(2, max_length + 1):
        walk([0], length)
    out.sort(key=lambda p: (p.tension, len(p.degrees), p.name))
    return out
