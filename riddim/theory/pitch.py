"""Note names and pitch-class arithmetic.

MIDI note numbers throughout: 60 is middle C (C3 in Deluge's naming, which
labels C60 as C3). We use the Deluge convention so that note names printed by
this package match what the screen shows.
"""

from __future__ import annotations

SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

# Deluge displays MIDI 60 as C3, i.e. octave number = midi // 12 - 2.
OCTAVE_OFFSET = 2

_NAME_TO_PC = {}
for _pc, _n in enumerate(SHARP_NAMES):
    _NAME_TO_PC[_n.upper()] = _pc
for _pc, _n in enumerate(FLAT_NAMES):
    _NAME_TO_PC[_n.upper()] = _pc


def pitch_class(midi: int) -> int:
    return midi % 12


def octave(midi: int) -> int:
    return midi // 12 - OCTAVE_OFFSET


def note_name(midi: int, flats: bool = False) -> str:
    names = FLAT_NAMES if flats else SHARP_NAMES
    return f"{names[midi % 12]}{octave(midi)}"


def parse_note(name: str) -> int:
    """Parse 'C3', 'Eb1', 'F#-1' into a MIDI note number.

    Raises ValueError on anything it cannot read, rather than guessing.
    """
    text = name.strip()
    if not text:
        raise ValueError("empty note name")

    i = 1
    if len(text) > 1 and text[1] in "#b":
        i = 2
    letter = text[:i].upper()
    if letter not in _NAME_TO_PC:
        raise ValueError(f"unknown note letter {text[:i]!r} in {name!r}")

    rest = text[i:]
    if not rest:
        raise ValueError(f"note name {name!r} is missing an octave")
    try:
        oct_num = int(rest)
    except ValueError as exc:
        raise ValueError(f"bad octave {rest!r} in {name!r}") from exc

    midi = _NAME_TO_PC[letter] + (oct_num + OCTAVE_OFFSET) * 12
    if not 0 <= midi <= 127:
        raise ValueError(f"note {name!r} is outside MIDI range (got {midi})")
    return midi


def to_midi(value) -> int:
    """Accept either a MIDI number or a note name and return a MIDI number."""
    if isinstance(value, str):
        return parse_note(value)
    return int(value)


def transpose_into(midi: int, low: int, high: int) -> int:
    """Move `midi` by whole octaves until it sits inside [low, high].

    If the window is narrower than an octave the note may not fit; we return
    the closest octave transposition rather than failing, because callers use
    this for register placement where "as close as possible" is the intent.
    """
    if low > high:
        raise ValueError(f"empty register window: [{low}, {high}]")
    note = midi
    while note < low:
        note += 12
    while note > high:
        note -= 12
    if note < low:
        # Window narrower than an octave: pick whichever end is nearer.
        return low if (low - note) <= (note + 12 - high) else note + 12
    return note


def interval_name(semitones: int) -> str:
    names = {
        0: "unison", 1: "b2", 2: "2", 3: "b3", 4: "3", 5: "4",
        6: "b5", 7: "5", 8: "b6", 9: "6", 10: "b7", 11: "7",
    }
    return names.get(semitones % 12, str(semitones))
