"""Chords built from scale degrees, and voicings that sit in a register.

Chords here are always derived from a Scale rather than spelled by name. That
guarantees every chord is in key, and it means a progression is just a list of
degrees - which is what makes modal interchange and transposition trivial.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import pitch
from .scale import Scale

# Quality names for display only; the actual notes come from the scale.
_QUALITY_BY_INTERVALS: dict[tuple[int, ...], str] = {
    (0, 3, 7): "m",
    (0, 4, 7): "",
    (0, 3, 6): "dim",
    (0, 4, 8): "aug",
    (0, 2, 7): "sus2",
    (0, 5, 7): "sus4",
    (0, 3, 7, 10): "m7",
    (0, 4, 7, 11): "maj7",
    (0, 4, 7, 10): "7",
    (0, 3, 6, 10): "m7b5",
    (0, 3, 6, 9): "dim7",
    (0, 3, 7, 11): "mMaj7",
    (0, 4, 8, 11): "maj7#5",
    (0, 5, 7, 10): "7sus4",
    (0, 2, 3, 7, 10): "m9",
    (0, 2, 4, 7, 10): "9",
    (0, 2, 4, 7, 11): "maj9",
    (0, 2, 4, 7): "add9",
    (0, 2, 3, 7): "madd9",
}


@dataclass(frozen=True)
class Chord:
    """A set of MIDI pitches with the scale degree it was built from."""

    pitches: tuple[int, ...]
    degree: int = 0
    scale: Scale | None = None

    @property
    def root(self) -> int:
        return min(self.pitches)

    @property
    def bass(self) -> int:
        return min(self.pitches)

    @property
    def root_pc(self) -> int:
        """Pitch class of the harmonic root.

        Taken from the scale degree the chord was built on, not from the
        lowest sounding note - an inverted or rootless voicing still belongs
        to the same chord, and naming it after whatever ended up on the
        bottom produces nonsense like calling a rootless Gm a "D".
        """
        if self.scale is not None:
            return self.scale.note(self.degree) % 12
        return self.pitches[0] % 12

    def intervals(self) -> tuple[int, ...]:
        """Semitone intervals above the root, deduplicated and octave-reduced."""
        root_pc = self.root_pc
        seen = []
        for p in sorted(self.pitches):
            iv = (p - root_pc) % 12
            if iv not in seen:
                seen.append(iv)
        return tuple(sorted(seen))

    @property
    def quality(self) -> str:
        found = _QUALITY_BY_INTERVALS.get(self.intervals())
        if found is not None:
            return found
        # A rootless or otherwise incomplete voicing: name it from the full
        # chord the degree implies, and mark that the root is not sounding.
        # First from the notes actually played plus the missing root - so a
        # rootless m9 is an m9 - then from the stacked thirds.
        with_root = _QUALITY_BY_INTERVALS.get(tuple(sorted(set(self.intervals()) | {0})))
        if with_root is not None and 0 not in self.intervals():
            return with_root + "(no root)"
        if self.scale is not None:
            full = tuple(sorted({(self.scale.note(self.degree + 2 * i) - self.root_pc) % 12
                                 for i in range(len(self.pitches) + 1)}))
            implied = _QUALITY_BY_INTERVALS.get(full)
            if implied is not None:
                return implied + ("(no root)" if 0 not in self.intervals() else "")
        return "?"

    def symbol(self) -> str:
        flats = self.scale is not None and self.scale.prefers_flats
        names = pitch.FLAT_NAMES if flats else pitch.SHARP_NAMES
        return f"{names[self.root_pc]}{self.quality}"

    def pitch_classes(self) -> set[int]:
        return {p % 12 for p in self.pitches}

    def contains_pc(self, midi: int) -> bool:
        return midi % 12 in self.pitch_classes()

    def transposed(self, semitones: int) -> "Chord":
        return Chord(
            tuple(p + semitones for p in self.pitches), self.degree, self.scale
        )

    def with_pitches(self, pitches) -> "Chord":
        return Chord(tuple(sorted(pitches)), self.degree, self.scale)

    def __str__(self) -> str:
        return self.symbol()


def stacked(scale: Scale, degree: int, size: int = 3, skip: int = 2) -> Chord:
    """Build a chord by stacking scale steps from `degree`.

    `skip=2` gives thirds (the normal triad/seventh stack); `skip=3` gives
    quartal voicings, which is a good trick for dub chord stabs because they
    sound modal and rootless rather than major/minor.
    """
    if size < 1:
        raise ValueError(f"chord size must be >= 1, got {size}")
    degrees = [degree + skip * i for i in range(size)]
    return Chord(tuple(scale.note(d) for d in degrees), degree, scale)


def triad(scale: Scale, degree: int) -> Chord:
    return stacked(scale, degree, size=3)


def seventh(scale: Scale, degree: int) -> Chord:
    return stacked(scale, degree, size=4)


def quartal(scale: Scale, degree: int, size: int = 3) -> Chord:
    return stacked(scale, degree, size=size, skip=3)


def sus(scale: Scale, degree: int, which: int = 4) -> Chord:
    """Suspended chord: replace the third with the 2nd or 4th scale step."""
    if which not in (2, 4):
        raise ValueError(f"sus must be 2 or 4, got {which}")
    offset = 1 if which == 2 else 3
    degrees = [degree, degree + offset, degree + 4]
    return Chord(tuple(scale.note(d) for d in degrees), degree, scale)


def with_major_third(chord: Chord) -> Chord:
    """Raise a minor third to a major one, leaving everything else alone.

    Used to borrow a leading tone from harmonic minor - the note this adds is
    deliberately outside the scale, which is the entire point of a major V in
    a minor key.
    """
    ps = sorted(chord.pitches)
    root = chord.root_pc
    raised = []
    for p in ps:
        raised.append(p + 1 if (p - root) % 12 == 3 else p)
    return chord.with_pitches(raised)


def inversions(chord: Chord) -> list[Chord]:
    """All inversions of a chord, each within one octave of the original."""
    out = []
    pitches = list(chord.pitches)
    for _ in range(len(pitches)):
        out.append(chord.with_pitches(pitches))
        pitches = pitches[1:] + [pitches[0] + 12]
    return out


def voicings(chord: Chord, low: int, high: int, max_span: int = 24) -> list[Chord]:
    """Every inversion of `chord`, at every octave that fits in [low, high].

    Used as the candidate set for voice leading. `max_span` rejects voicings
    that are spread too wide to read as one chord.
    """
    out: list[Chord] = []
    seen: set[tuple[int, ...]] = set()
    for inv in inversions(chord):
        base = list(inv.pitches)
        span = max(base) - min(base)
        if span > max_span:
            continue
        # Slide the whole voicing octave by octave across the window.
        shift = -((min(base) - low) // 12) * 12
        for octaves in range(-3, 4):
            moved = tuple(p + shift + 12 * octaves for p in base)
            if min(moved) < low or max(moved) > high:
                continue
            if moved in seen:
                continue
            seen.add(moved)
            out.append(inv.with_pitches(moved))
    return out


def drop2(chord: Chord) -> Chord:
    """Drop the second-from-top voice an octave. Opens up a close voicing."""
    ps = sorted(chord.pitches)
    if len(ps) < 3:
        return chord
    ps[-2] -= 12
    return chord.with_pitches(ps)


def shell(chord: Chord) -> Chord:
    """Root, third and seventh only - no fifth.

    The fifth is the first thing to go when a chord has to leave room for a
    sub bass, which in this music it always does.
    """
    ps = sorted(chord.pitches)
    if len(ps) < 4:
        return chord
    return chord.with_pitches([ps[0], ps[1], ps[3]])


def rootless(chord: Chord) -> Chord:
    """Drop the root. The bass is already playing it - this stops the mud."""
    ps = sorted(chord.pitches)
    if len(ps) <= 2:
        return chord
    return chord.with_pitches(ps[1:])


@dataclass
class ChordEvent:
    """A chord placed on the timeline."""

    chord: Chord
    start: int  # ticks
    length: int  # ticks
    label: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def end(self) -> int:
        return self.start + self.length
