"""Melody as motif plus development.

The reason writing melody feels hard is usually that it is being attempted in
one pass: invent eight bars of good tune. That is not how it is done. You
invent one or two bars - a *motif* - and then you develop it, and the
development is mechanical. Transpose it, invert it, cut it in half, displace
it by a beat, stretch it. Those operations are the entire content of most
instrumental music.

So this module has two halves: `generate` makes one short cell, and everything
below `# Development` transforms a cell into a related cell. A part that
evolves over a song is just a motif plus a schedule of transformations.

Everything works in *diatonic* space - notes are scale degrees, not
semitones - so a transposed or inverted motif is automatically still in key.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace

from .chord import Chord
from .rhythm import Pattern, metric_weight
from .scale import Scale
from ..units import TICKS_PER_16TH


@dataclass(frozen=True)
class Tone:
    """A note in diatonic space: a scale degree at a point in time."""

    degree: int
    step: float
    length: float = 1.0
    velocity: int = 90

    @property
    def end(self) -> float:
        return self.step + self.length


@dataclass
class Motif:
    """A short melodic cell, measured in steps and scale degrees."""

    tones: list[Tone] = field(default_factory=list)
    length: int = 16  # steps
    steps_per_bar: int = 16
    name: str = ""

    def sorted_tones(self) -> list[Tone]:
        return sorted(self.tones, key=lambda t: t.step)

    def degrees(self) -> list[int]:
        return [t.degree for t in self.sorted_tones()]

    def range(self) -> int:
        """Span in scale degrees. Small is good; a motif is not a solo."""
        ds = self.degrees()
        return max(ds) - min(ds) if ds else 0

    def copy(self) -> "Motif":
        return Motif(list(self.tones), self.length, self.steps_per_bar, self.name)

    def notes(self, scale: Scale) -> list[tuple[int, float, float, int]]:
        """Realise to (midi, step, length, velocity)."""
        return [
            (scale.note(t.degree), t.step, t.length, t.velocity)
            for t in self.sorted_tones()
        ]

    def to_ticks(
        self, scale: Scale, ticks_per_step: int = TICKS_PER_16TH, offset: int = 0
    ) -> list[tuple[int, int, int, int]]:
        """Realise to (midi, start_tick, length_ticks, velocity)."""
        out = []
        for midi, step, length, vel in self.notes(scale):
            start = offset + int(round(step * ticks_per_step))
            ticks = max(1, int(round(length * ticks_per_step)))
            out.append((midi, start, ticks, vel))
        return out

    def __len__(self) -> int:
        return len(self.tones)


# --------------------------------------------------------------------------
# Contour
# --------------------------------------------------------------------------

# A contour is a normalised shape sampled across the motif; the generator
# follows it loosely rather than exactly, so the result has a direction
# without sounding like a ramp.
CONTOURS: dict[str, list[float]] = {
    "arch": [0.0, 0.5, 1.0, 0.6, 0.1],
    "valley": [1.0, 0.4, 0.0, 0.4, 0.9],
    "ascending": [0.0, 0.3, 0.55, 0.8, 1.0],
    "descending": [1.0, 0.75, 0.5, 0.25, 0.0],
    "static": [0.4, 0.5, 0.45, 0.5, 0.4],
    "zigzag": [0.2, 0.9, 0.3, 0.8, 0.25],
    "terraced": [0.1, 0.1, 0.6, 0.6, 1.0],
    "falling_tail": [0.6, 0.8, 0.7, 0.3, 0.0],
}


def sample_contour(name: str, position: float) -> float:
    """Value of a named contour at `position` in 0..1, linearly interpolated."""
    shape = CONTOURS.get(name)
    if shape is None:
        raise ValueError(
            f"unknown contour {name!r}; known: {', '.join(sorted(CONTOURS))}"
        )
    if len(shape) == 1:
        return shape[0]
    x = max(0.0, min(1.0, position)) * (len(shape) - 1)
    i = int(x)
    if i >= len(shape) - 1:
        return shape[-1]
    frac = x - i
    return shape[i] * (1 - frac) + shape[i + 1] * frac


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------

def generate(
    rhythm: Pattern,
    contour: str = "arch",
    span: int = 5,
    centre: int = 0,
    chord: Chord | None = None,
    scale: Scale | None = None,
    rng: random.Random | None = None,
    step_bias: float = 0.75,
    name: str = "",
) -> Motif:
    """Put degrees onto a rhythm, following a contour.

    Three constraints keep it musical:

    - **Step bias.** Most motion is by one scale step. Melodies that leap
      constantly sound like arpeggiator output.
    - **Leap recovery.** After a leap of 3+ degrees, the next move steps back
      the other way. This is a real rule from species counterpoint and it is
      astonishingly effective.
    - **Chord tones on strong beats.** If a chord is supplied, notes on
      metrically strong steps are pulled onto chord tones, so the melody
      agrees with the harmony where it counts and is free elsewhere.
    """
    rng = rng or random.Random()
    hits = rhythm.sorted_hits()
    if not hits:
        return Motif([], rhythm.length, rhythm.steps_per_bar, name)

    low, high = centre - span // 2, centre + span - span // 2
    tones: list[Tone] = []
    previous: int | None = None
    last_leap = 0

    for i, hit in enumerate(hits):
        position = i / max(1, len(hits) - 1)
        target = low + sample_contour(contour, position) * (high - low)

        if previous is None:
            degree = int(round(target))
        elif last_leap >= 3:
            # Recover from the leap by stepping back against it.
            degree = previous - (1 if last_leap > 0 else -1)
        elif rng.random() < step_bias:
            # Step toward the contour, or hold if we are already there.
            if abs(target - previous) < 0.5:
                degree = previous
            else:
                degree = previous + (1 if target > previous else -1)
        else:
            degree = int(round(target)) + rng.choice([-1, 0, 1])

        degree = max(low - 2, min(high + 2, degree))

        # Snap strong beats onto chord tones.
        if chord is not None and scale is not None:
            weight = metric_weight(int(hit.step), rhythm.steps_per_bar)
            if weight >= 2.0:
                degree = _nearest_chord_degree(degree, chord, scale)

        last_leap = 0 if previous is None else degree - previous
        if abs(last_leap) < 3:
            last_leap = 0

        tones.append(Tone(degree, hit.step, hit.length, hit.velocity))
        previous = degree

    return Motif(tones, rhythm.length, rhythm.steps_per_bar, name)


def _nearest_chord_degree(degree: int, chord: Chord, scale: Scale) -> int:
    """Nudge a scale degree to the closest degree that is a chord tone."""
    targets = chord.pitch_classes()
    for offset in (0, -1, 1, -2, 2, -3, 3):
        candidate = degree + offset
        if scale.note(candidate) % 12 in targets:
            return candidate
    return degree


# --------------------------------------------------------------------------
# Development
# --------------------------------------------------------------------------

def transpose(motif: Motif, degrees: int) -> Motif:
    """Diatonic transposition - stays in key by construction."""
    out = motif.copy()
    out.tones = [replace(t, degree=t.degree + degrees) for t in motif.tones]
    return out


def invert(motif: Motif, axis: int | None = None) -> Motif:
    """Mirror the melody around an axis degree. Rising becomes falling.

    Defaults to the first note, which keeps the motif's opening anchored and
    makes the relationship audible.
    """
    if not motif.tones:
        return motif.copy()
    if axis is None:
        axis = motif.sorted_tones()[0].degree
    out = motif.copy()
    out.tones = [replace(t, degree=2 * axis - t.degree) for t in motif.tones]
    return out


def retrograde(motif: Motif) -> Motif:
    """Play the pitches backwards over the same rhythm.

    Pitch-only retrograde rather than full time-reversal: reversing the rhythm
    as well usually destroys the groove, which we care about more than we care
    about strict serial technique.
    """
    tones = motif.sorted_tones()
    degrees = [t.degree for t in tones][::-1]
    out = motif.copy()
    out.tones = [replace(t, degree=d) for t, d in zip(tones, degrees)]
    return out


def augment(motif: Motif, factor: float = 2.0) -> Motif:
    """Stretch in time. factor 2 = half speed, 0.5 = double speed."""
    if factor <= 0:
        raise ValueError(f"augment factor must be positive, got {factor}")
    out = motif.copy()
    out.tones = [
        replace(t, step=t.step * factor, length=t.length * factor)
        for t in motif.tones
    ]
    out.length = max(1, int(round(motif.length * factor)))
    return out


def diminish(motif: Motif, factor: float = 2.0) -> Motif:
    return augment(motif, 1.0 / factor)


def displace(motif: Motif, steps: float) -> Motif:
    """Shift in time, wrapping within the motif length.

    Rhythmic displacement is the most underused development trick there is:
    the same notes starting a 16th later read as a completely new idea.
    """
    if motif.length <= 0:
        return motif.copy()
    out = motif.copy()
    out.tones = [
        replace(t, step=(t.step + steps) % motif.length) for t in motif.tones
    ]
    return out


def fragment(motif: Motif, keep: float = 0.5, from_start: bool = True) -> Motif:
    """Keep only part of the motif. Fragmentation is how tension builds."""
    tones = motif.sorted_tones()
    if not tones:
        return motif.copy()
    cutoff = motif.length * max(0.05, min(1.0, keep))
    if from_start:
        kept = [t for t in tones if t.step < cutoff]
    else:
        kept = [
            replace(t, step=t.step - (motif.length - cutoff))
            for t in tones
            if t.step >= motif.length - cutoff
        ]
    out = motif.copy()
    out.tones = kept
    return out


def sequence(motif: Motif, offsets: list[int]) -> Motif:
    """Repeat the motif at successive diatonic transpositions.

    `sequence(m, [0, -1, -2])` is a descending sequence - three statements,
    each a scale step lower. This is the single most reliable way to turn one
    bar into four that sound composed.
    """
    tones: list[Tone] = []
    for i, offset in enumerate(offsets):
        shift = i * motif.length
        for t in motif.sorted_tones():
            tones.append(replace(t, degree=t.degree + offset, step=t.step + shift))
    out = motif.copy()
    out.tones = tones
    out.length = motif.length * max(1, len(offsets))
    return out


def ornament(
    motif: Motif, amount: float = 0.3, rng: random.Random | None = None
) -> Motif:
    """Add passing notes between tones that are a third apart.

    Fills gaps rather than adding notes anywhere, so ornamentation always
    connects something instead of cluttering.
    """
    rng = rng or random.Random()
    tones = motif.sorted_tones()
    out_tones = list(tones)
    for a, b in zip(tones, tones[1:]):
        gap = b.degree - a.degree
        room = b.step - a.end
        if abs(gap) == 2 and room >= 0 and a.length >= 0.5 and rng.random() < amount:
            mid_step = a.step + a.length / 2
            if mid_step < b.step:
                out_tones.append(
                    Tone(
                        a.degree + (1 if gap > 0 else -1),
                        mid_step,
                        max(0.25, a.length / 2),
                        max(1, a.velocity - 15),
                    )
                )
    out = motif.copy()
    out.tones = sorted(out_tones, key=lambda t: t.step)
    return out


def thin_tones(
    motif: Motif, keep: float = 0.6, rng: random.Random | None = None
) -> Motif:
    """Drop notes, weakest metric position first. The dub version of a solo."""
    if keep >= 1.0 or not motif.tones:
        return motif.copy()
    rng = rng or random.Random()
    target = max(1, int(round(len(motif.tones) * keep)))
    ranked = sorted(
        motif.tones,
        key=lambda t: (
            -metric_weight(int(t.step), motif.steps_per_bar),
            -t.velocity,
            rng.random(),
        ),
    )
    out = motif.copy()
    out.tones = sorted(ranked[:target], key=lambda t: t.step)
    return out


def octave_drop(motif: Motif, scale: Scale, octaves: int = -1) -> Motif:
    """Move the whole motif by octaves, in degrees."""
    return transpose(motif, octaves * scale.size)


# --------------------------------------------------------------------------
# Scheduled development
# --------------------------------------------------------------------------

# A menu of (name, function) development moves, ordered roughly by how far
# they take you from the original. The arranger walks this list as a song
# progresses so that later sections are further from the source material.
DEVELOPMENTS = (
    ("statement", lambda m, rng: m.copy()),
    ("thinned", lambda m, rng: thin_tones(m, 0.7, rng)),
    ("displaced", lambda m, rng: displace(m, 2)),
    ("transposed", lambda m, rng: transpose(m, rng.choice([-2, 2, 3]))),
    ("ornamented", lambda m, rng: ornament(m, 0.5, rng)),
    ("fragmented", lambda m, rng: fragment(m, 0.5)),
    ("inverted", lambda m, rng: invert(m)),
    ("augmented", lambda m, rng: augment(m, 2.0)),
    ("retrograde", lambda m, rng: retrograde(m)),
)


def develop(
    motif: Motif, stage: int, rng: random.Random | None = None
) -> tuple[Motif, str]:
    """Return the motif transformed for a given development stage.

    Stage 0 is the plain statement; higher stages wander further. Returns the
    name of the transformation too, so the arrangement can be explained.
    """
    rng = rng or random.Random()
    name, fn = DEVELOPMENTS[stage % len(DEVELOPMENTS)]
    result = fn(motif, rng)
    result.name = f"{motif.name}:{name}" if motif.name else name
    return result, name
