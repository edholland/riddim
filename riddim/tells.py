"""Tells: where a listener who knows the music would hear a machine.

`analyze.py` asks whether a song does its job by the engine's own lights,
and `critique.py` whether anything is broken. This asks a harder question:
**where would a producer who plays this music clock that nobody wrote it?**

The answer is rarely a wrong note. It is too many notes - a hook that walks
the whole scale, a sub that visits every degree, a line that never stops and
never repeats itself. People write with a handful of pitches and say the
same thing again; generators reach for everything they are allowed. So each
tell here is a plain statistic of the first drop - the part a listener judges
a record by - against the range the family's records sit in.

The ranges are genre knowledge written down to be argued with, not measured
from a corpus: when reference MIDI is to hand, calibrate them against it.

    from riddim.tells import tells
    for t in tells(song):
        print(t)          # lead_vocab 6 (want <= 4)  TELL
"""

from __future__ import annotations

from dataclasses import dataclass

from .arrange import structure
from .song import Song
from .units import TICKS_PER_BAR

# (low, high) per family; None is an open end. A drop is judged against its
# family: a dub melodica line is a song and may use more notes than a 140
# lead, a dubstep sub is one note and its fifth, a garage bass moves.
TARGETS: dict[str, dict[str, tuple[float | None, float | None]]] = {
    "dub": {
        # Distinct pitch classes in the drop's lead. A pentatonic tune and
        # its mode's colour note is how people write; reaching for the
        # sixth and seventh scale tones as well is the scale being walked.
        # A melodica line is a song and gets one more.
        "lead_vocab": (None, 6),
        "lead_range": (None, 12),      # semitones, ignoring octave flicks
        "lead_density": (None, 5.0),   # notes per bar
        "lead_restated": (0.45, None), # bars that say a bar of the phrase again
        # Distinct sub pitch classes that are neither a root nor a fifth of
        # the drop's chords: what the line adds beyond following the
        # changes. A roots bassline is a melody and adds plenty.
        "sub_extra": (None, 3),
        "sub_on_root": (0.45, None),   # share of the sub's sounding time on the chord root
        "drop_chords": (None, 4),      # distinct chords under the first drop
    },
    "dubstep": {
        "lead_vocab": (None, 5),
        "lead_range": (None, 12),
        "lead_density": (None, 4.0),
        "lead_restated": (0.5, None),
        "sub_extra": (None, 2),        # the b2 or a riff's third, and one approach note
        "sub_on_root": (0.75, None),   # the 140 sub is the root, and the root is the event
        "drop_chords": (None, 2),      # a 140 drop is a drone or two chords, not a pop loop
    },
    "garage": {
        "lead_vocab": (None, 5),
        "lead_range": (None, 12),
        "lead_density": (None, 6.0),
        "lead_restated": (0.45, None),
        "sub_extra": (None, 2),
        "sub_on_root": (0.55, None),
        "drop_chords": (None, 4),
    },
}


@dataclass(frozen=True)
class Tell:
    name: str
    value: float
    low: float | None
    high: float | None

    @property
    def ok(self) -> bool:
        return ((self.low is None or self.value >= self.low)
                and (self.high is None or self.value <= self.high))

    def __str__(self) -> str:
        want = (f"<= {self.high:g}" if self.low is None else
                f">= {self.low:g}" if self.high is None else
                f"{self.low:g}..{self.high:g}")
        return f"{self.name:14s} {self.value:6.2f}  (want {want}){'' if self.ok else '  TELL'}"


def first_drop(song: Song):
    """The section a listener judges the record by: the first peak."""
    if song.arrangement is None:
        return None
    peaks = [s for s in song.arrangement.sections if structure.kind_of(s) == "peak"]
    return peaks[0] if peaks else None


def _in(part, sec):
    if part is None or sec is None:
        return []
    a = sec.start_bar * TICKS_PER_BAR
    b = a + sec.bars * TICKS_PER_BAR
    return sorted((n for n in part.notes if a <= n.start < b), key=lambda n: (n.start, n.midi))


def _core_range(notes) -> int:
    """The line's range with octave flicks folded back: a note an octave from
    one of its neighbours is the 140 lead's gesture, not more range."""
    if not notes:
        return 0
    pitches = [n.midi for n in notes]
    keep = []
    for i, p in enumerate(pitches):
        near = pitches[max(0, i - 1):i] + pitches[i + 1:i + 2]
        if any(abs(p - q) == 12 for q in near) and p == max(p, *near):
            continue
        keep.append(p)
    keep = keep or pitches
    return max(keep) - min(keep)


def _restated(notes, sec) -> float:
    """The share of the drop's sung bars that repeat a bar one, two, four or
    eight bars back - rhythm and pitch, note for note. An eight-bar hook
    said again is repetition as much as a one-bar riff is."""
    if sec is None or sec.bars < 2:
        return 1.0
    a = sec.start_bar * TICKS_PER_BAR
    bars: list[tuple] = []
    for b in range(sec.bars):
        lo = a + b * TICKS_PER_BAR
        bars.append(tuple((n.start - lo, n.midi) for n in notes
                          if lo <= n.start < lo + TICKS_PER_BAR))
    sung = [i for i in range(1, len(bars)) if bars[i]]
    if not sung:
        return 1.0
    hits = sum(1 for i in sung if any(i - k >= 0 and bars[i - k] == bars[i] for k in (1, 2, 4, 8)))
    return hits / len(sung)


def _chords_in(song: Song, sec) -> list:
    if sec is None or song.harmony is None:
        return []
    a = sec.start_bar * TICKS_PER_BAR
    b = a + sec.bars * TICKS_PER_BAR
    return [e.chord for e in song.harmony.events if e.start < b and e.start + e.length > a]


def _on_root(song: Song, notes) -> float:
    """The share of the line's sounding time spent on the root of the chord
    under it - by time, as it is heard: a held root is the bass, a passing
    note is a passing note."""
    total = on = 0
    for n in notes:
        ev = song.harmony.at(n.start) if song.harmony else None
        total += n.length
        if ev is None or n.midi % 12 == ev.chord.root_pc:
            on += n.length
    return on / total if total else 1.0


def tells(song: Song) -> list[Tell]:
    """Every tell for the song's first drop, against its family's ranges."""
    from . import profile as profile_mod
    family = profile_mod.get(song.params.profile).family if song.params else "dub"
    want = TARGETS.get(family, TARGETS["dub"])
    sec = first_drop(song)
    lead = _in(song.part("lead"), sec)
    sub = _in(song.part("sub"), sec)
    values: dict[str, float] = {}
    if lead:
        values["lead_vocab"] = len({n.midi % 12 for n in lead})
        values["lead_range"] = _core_range(lead)
        values["lead_density"] = len(lead) / sec.bars
        values["lead_restated"] = _restated(lead, sec)
    chords = _chords_in(song, sec)
    if chords:
        values["drop_chords"] = len({(c.root_pc, frozenset(c.pitch_classes())) for c in chords})
    if sub:
        frame = {c.root_pc for c in chords} | {(c.root_pc + 7) % 12 for c in chords}
        values["sub_extra"] = len({n.midi % 12 for n in sub} - frame)
        values["sub_on_root"] = _on_root(song, sub)
    return [Tell(k, round(float(v), 3), *want[k]) for k, v in values.items()]


def count(song: Song) -> int:
    """How many tells the song gives away."""
    return sum(1 for t in tells(song) if not t.ok)


def survey(songs) -> dict[str, dict[str, float]]:
    """Mean value and pass rate of each tell across a batch."""
    rows: dict[str, list[Tell]] = {}
    for s in songs:
        for t in tells(s):
            rows.setdefault(t.name, []).append(t)
    return {k: {"mean": sum(t.value for t in v) / len(v),
                "pass": sum(t.ok for t in v) / len(v), "n": len(v)}
            for k, v in rows.items()}
