"""Scales and modes, with degree arithmetic that wraps across octaves.

Degree indexing is the thing that makes the rest of the library simple: a
melody is a list of scale degrees, and `Scale.note(degree)` turns any integer
degree - negative, or past the top of the scale - into a MIDI note. That means
transposition and inversion can happen in *diatonic* space, so a transformed
motif is always still in key.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import pitch

# Interval sets, in semitones from the root.
SCALES: dict[str, tuple[int, ...]] = {
    "major": (0, 2, 4, 5, 7, 9, 11),
    "ionian": (0, 2, 4, 5, 7, 9, 11),
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "phrygian": (0, 1, 3, 5, 7, 8, 10),
    "lydian": (0, 2, 4, 6, 7, 9, 11),
    "mixolydian": (0, 2, 4, 5, 7, 9, 10),
    "minor": (0, 2, 3, 5, 7, 8, 10),
    "aeolian": (0, 2, 3, 5, 7, 8, 10),
    "locrian": (0, 1, 3, 5, 6, 8, 10),
    "harmonic_minor": (0, 2, 3, 5, 7, 8, 11),
    "melodic_minor": (0, 2, 3, 5, 7, 9, 11),
    "phrygian_dominant": (0, 1, 4, 5, 7, 8, 10),
    "minor_pentatonic": (0, 3, 5, 7, 10),
    "major_pentatonic": (0, 2, 4, 7, 9),
    "blues": (0, 3, 5, 6, 7, 10),
}

# The modes that carry the idiom. Dub and dubstep live in minor-ish modal
# space; phrygian and phrygian dominant are where the menace comes from.
DARK_MODES = ("minor", "dorian", "phrygian", "harmonic_minor", "phrygian_dominant")

# How far each mode sits round the circle of fifths from aeolian. Used only to
# work out whether a key is spelled with sharps or flats, so that a G minor
# tune prints "Eb" and not "D#".
_MODE_FIFTHS = {
    "lydian": 4, "major": 3, "ionian": 3, "major_pentatonic": 3,
    "mixolydian": 2, "dorian": 1,
    "minor": 0, "aeolian": 0, "harmonic_minor": 0, "melodic_minor": 0,
    "minor_pentatonic": 0, "blues": 0,
    "phrygian": -1, "phrygian_dominant": -1, "locrian": -2,
}


@dataclass(frozen=True)
class Scale:
    """A root pitch class plus an interval set."""

    root: int  # MIDI note of the tonic, register included
    name: str = "minor"

    def __post_init__(self):
        if self.name not in SCALES:
            raise ValueError(
                f"unknown scale {self.name!r}; known: {', '.join(sorted(SCALES))}"
            )

    @property
    def intervals(self) -> tuple[int, ...]:
        return SCALES[self.name]

    @property
    def size(self) -> int:
        return len(self.intervals)

    @property
    def root_pc(self) -> int:
        return self.root % 12

    @property
    def pitch_classes(self) -> tuple[int, ...]:
        return tuple(sorted((self.root_pc + i) % 12 for i in self.intervals))

    def note(self, degree: int, octave_shift: int = 0) -> int:
        """MIDI note for a scale degree. Degree 0 is the tonic.

        Degrees outside 0..size-1 wrap into the octave above or below, so
        degree 7 in a 7-note scale is the tonic an octave up and degree -1 is
        the leading tone below.
        """
        index = degree % self.size
        octaves = degree // self.size + octave_shift
        return self.root + self.intervals[index] + 12 * octaves

    def contains(self, midi: int) -> bool:
        return midi % 12 in self.pitch_classes

    def degree_of(self, midi: int) -> int | None:
        """Absolute scale degree of a note, or None if it is not in the scale.

        The degree is relative to the scale's own root register, so
        `scale.note(scale.degree_of(n)) == n` for any in-scale n.
        """
        offset = midi - self.root
        octaves, within = divmod(offset, 12)
        try:
            index = self.intervals.index(within)
        except ValueError:
            return None
        return index + octaves * self.size

    def snap(self, midi: int, prefer: str = "nearest") -> int:
        """Move an out-of-scale note onto the nearest scale tone.

        `prefer` breaks ties and sets direction: "nearest", "down", "up".
        """
        if self.contains(midi):
            return midi
        if prefer == "down":
            candidates = range(midi, midi - 12, -1)
        elif prefer == "up":
            candidates = range(midi, midi + 12)
        else:
            candidates = sorted(
                range(midi - 6, midi + 7), key=lambda n: (abs(n - midi), n > midi)
            )
        for note in candidates:
            if 0 <= note <= 127 and self.contains(note):
                return note
        return midi

    def notes_between(self, low: int, high: int) -> list[int]:
        """Every scale tone in [low, high], ascending."""
        return [n for n in range(low, high + 1) if self.contains(n)]

    @property
    def sharps(self) -> int:
        """Key signature as a signed count: positive sharps, negative flats.

        Derived from the tonic's distance round the circle of fifths from A
        (the zero-accidental minor key), adjusted for the mode.
        """
        fifths = ((self.root_pc - 9) * 7) % 12
        if fifths > 6:
            fifths -= 12
        return fifths + _MODE_FIFTHS.get(self.name, 0)

    @property
    def prefers_flats(self) -> bool:
        return self.sharps < 0

    def spell(self, midi: int) -> str:
        """Note name using this key's preferred accidentals."""
        return pitch.note_name(midi, flats=self.prefers_flats)

    def transposed(self, semitones: int) -> "Scale":
        return Scale(self.root + semitones, self.name)

    def relabel(self, name: str) -> "Scale":
        """Same root, different mode - for modal interchange."""
        return Scale(self.root, name)

    def __str__(self) -> str:
        names = pitch.FLAT_NAMES if self.prefers_flats else pitch.SHARP_NAMES
        return f"{names[self.root_pc]} {self.name}"


def parse_key(text: str) -> Scale:
    """Parse 'G minor', 'F#1 phrygian', 'C dorian' into a Scale.

    A bare note letter with no octave defaults to octave 1 (MIDI 36-47), which
    is where dub basslines live.
    """
    parts = text.replace("-", "_").split()
    if not parts:
        raise ValueError("empty key")
    root_text = parts[0]
    mode = "_".join(parts[1:]).lower() if len(parts) > 1 else "minor"

    try:
        root = pitch.parse_note(root_text)
    except ValueError:
        # No octave given: default to octave 1, the sub-bass register.
        root = pitch.parse_note(f"{root_text}1")
    return Scale(root, mode)
