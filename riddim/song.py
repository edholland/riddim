"""The song model: notes, parts, and the rig they get sent to.

Everything upstream of here works in scale degrees and steps. Everything
downstream works in MIDI notes and ticks. This module is the boundary, and it
is deliberately dumb - no theory, no generation, just timed notes grouped into
parts with a destination.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .theory import pitch
from .units import TICKS_PER_BAR


@dataclass(frozen=True)
class Note:
    """A MIDI note on the absolute song timeline, in ticks."""

    midi: int
    start: int
    length: int
    velocity: int = 100

    @property
    def end(self) -> int:
        return self.start + self.length

    def moved(self, delta: int) -> "Note":
        return Note(self.midi, self.start + delta, self.length, self.velocity)

    def __str__(self) -> str:
        return (
            f"{pitch.note_name(self.midi)}@{self.start / TICKS_PER_BAR:.2f}b"
            f" len={self.length} v={self.velocity}"
        )


@dataclass(frozen=True)
class Automation:
    """A parameter ramp or hold on the timeline.

    `param` is one of the Deluge AI SysEx aliases: lpf, resonance, hpf, reverb.
    Values are 0..1 here and get scaled to the wire format at send time.
    """

    param: str
    start: int
    length: int
    start_value: float
    end_value: float | None = None

    @property
    def is_ramp(self) -> bool:
        return self.end_value is not None and self.end_value != self.start_value


@dataclass
class Part:
    """One instrument's worth of notes."""

    role: str
    name: str = ""
    notes: list[Note] = field(default_factory=list)
    automation: list[Automation] = field(default_factory=list)
    # Where it goes. `output` is "midi" for external gear or "synth"/"kit" for
    # a Deluge internal instrument.
    output: str = "synth"
    channel: int = 1          # MIDI channel, 1-16, for output == "midi"
    preset: str = ""          # Deluge preset name, for output == "synth"
    monophonic: bool = False
    comment: str = ""

    def add(self, note: Note) -> None:
        self.notes.append(note)

    def extend(self, notes) -> None:
        self.notes.extend(notes)

    def sorted_notes(self) -> list[Note]:
        return sorted(self.notes, key=lambda n: (n.start, n.midi))

    def range(self) -> tuple[int, int]:
        if not self.notes:
            return (0, 0)
        return (min(n.midi for n in self.notes), max(n.midi for n in self.notes))

    def in_bars(self, start_bar: int, end_bar: int) -> list[Note]:
        lo, hi = start_bar * TICKS_PER_BAR, end_bar * TICKS_PER_BAR
        return [n for n in self.sorted_notes() if lo <= n.start < hi]

    def __len__(self) -> int:
        return len(self.notes)


@dataclass
class Song:
    """A complete arrangement, ready to render."""

    name: str
    bpm: int
    parts: list[Part] = field(default_factory=list)
    arrangement: object | None = None   # arrange.structure.Arrangement
    harmony: object | None = None       # theory.harmony.HarmonyPlan
    scale: object | None = None         # theory.scale.Scale
    seed: int = 0
    concept: str = ""
    params: object | None = None        # params.SongParams, fully resolved
    notes_for_player: list[str] = field(default_factory=list)
    # The final hit after the last bar, as (role, Note). Kept off the parts:
    # the parts are what the Deluge loops, and nothing may run past the
    # arrangement there. A MIDI file or a WAV ends on it.
    coda: list = field(default_factory=list)

    def notes_with_coda(self, part: "Part") -> list:
        return part.sorted_notes() + [n for r, n in self.coda if r == part.role]

    def part(self, role: str) -> Part | None:
        for p in self.parts:
            if p.role == role:
                return p
        return None

    @property
    def length_ticks(self) -> int:
        ends = [n.end for p in self.parts for n in p.notes]
        return max(ends) if ends else 0

    @property
    def bars(self) -> int:
        if self.arrangement is not None:
            return self.arrangement.bars
        return -(-self.length_ticks // TICKS_PER_BAR)

    def note_count(self) -> int:
        return sum(len(p.notes) for p in self.parts)

    def summary(self) -> str:
        lines = [
            f"{self.name}  -  {self.bpm}bpm, {self.bars} bars, "
            f"{self.note_count()} notes  (seed {self.seed})"
        ]
        if self.scale is not None:
            lines.append(f"key: {self.scale}")
        if self.concept:
            lines.append(f"concept: {self.concept}")
        if self.harmony is not None:
            prog = self.harmony.progression
            lines.append(f"progression: {prog.name}  {' '.join(self.harmony.symbols()[:8])}")
            if prog.note:
                lines.append(f"  {prog.note}")
        for p in self.parts:
            lo, hi = p.range()
            dest = (
                f"MIDI ch{p.channel}"
                if p.output == "midi"
                else f"Deluge {p.output}" + (f" '{p.preset}'" if p.preset else "")
            )
            span = f"{pitch.note_name(lo)}-{pitch.note_name(hi)}" if p.notes else "-"
            lines.append(
                f"  {p.role:<8} {len(p.notes):>4} notes  {span:<12} -> {dest}"
            )
            if p.comment:
                lines.append(f"      {p.comment}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Rig
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Destination:
    output: str          # "midi" | "synth" | "kit"
    channel: int = 1
    preset: str = ""
    monophonic: bool = False
    comment: str = ""


@dataclass
class Rig:
    """Where each role goes in the physical setup.

    The default assumes: every pitched part goes out over MIDI to the rack -
    chords, sub, lead and pad - and only the drums and percussion stay on
    Deluge kits.

    One consequence worth knowing: the firmware only accepts param automation
    on SYNTH clips, so with no pitched part on a Deluge synth, no filter
    automation is written at all. That is correct - the filters being moved
    belong to the Coral - but it means every bit of filter movement is played,
    not sequenced. `ALL_DELUGE` keeps the parts internal and does get
    automation, which is useful for sketching without the rack patched.
    """

    destinations: dict[str, Destination] = field(
        default_factory=lambda: {
            "chords": Destination(
                "midi", 1,
                comment="Coral - chord voice. Short offbeat stabs; feed these "
                        "to the Nostalgia send.",
            ),
            "sub": Destination(
                "midi", 2, monophonic=True,
                comment="Coral - single oscillator bass voice. Mono, so notes "
                        "are truncated to never overlap.",
            ),
            "lead": Destination("midi", 3, comment="Coral - lead voice."),
            "pad": Destination(
                "midi", 4,
                comment="Coral - pad voice. Long held chords under the stabs.",
            ),
            "drums": Destination("kit", 10, comment="Deluge kit."),
            "perc": Destination("kit", 11, comment="Deluge kit, upper rows."),
        }
    )

    def for_role(self, role: str) -> Destination:
        return self.destinations.get(role, Destination("synth"))


ALL_DELUGE = Rig(
    destinations={
        "sub": Destination("synth", 1, monophonic=True, comment="Deluge synth bass."),
        "chords": Destination("synth", 2, comment="Deluge synth chords."),
        "pad": Destination("synth", 3, comment="Deluge synth pad."),
        "lead": Destination("synth", 4, comment="Deluge synth lead."),
        "drums": Destination("kit", 10, comment="Deluge kit."),
        "perc": Destination("kit", 10, comment="Deluge kit, upper rows."),
    }
)
