"""Song structure: sections, energy, and which parts are playing.

The insight that makes arranging tractable is that a dub arrangement is
*subtractive*. You do not write eight different sections. You write one
riddim, and then you make sections by muting things. The engineer at the desk
pulling the bass out for four bars and throwing the snare into the delay is
doing arrangement, and it is the whole genre.

So a Section here is mostly a mute map plus an energy number. `energy` then
drives density, register and filter automation everywhere else, which is what
makes a song build and release instead of just looping.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..units import TICKS_PER_BAR

# The parts we know about. A part is a role, not a specific instrument, so the
# same arrangement can be voiced on Deluge synths or sent out to the modular.
ROLES = ("drums", "perc", "sub", "chords", "pad", "lead")


@dataclass
class Section:
    """A span of bars with a fixed energy level and mute map."""

    name: str
    bars: int
    energy: float = 0.5
    active: tuple[str, ...] = ROLES
    # How far the motifs have been developed by this point in the song.
    development: int = 0
    # Free-text direction, surfaced in the printed arrangement so the plan is
    # readable rather than just executable.
    direction: str = ""
    start_bar: int = 0
    dub_throw: bool = False
    # This section's harmonic personality, as a progression name. None means
    # "inherit the tune's core progression". A song changes its harmonic
    # identity where the energy curve asks for it: the drop earns a cadence,
    # the hole earns a drone.
    progression: str | None = None
    # How much the section climbs through its own span, 0..1. A build with a
    # ramp thickens bar by bar toward its energy level instead of landing on
    # a plateau - the approach to the drop becomes audible.
    ramp: float = 0.0

    @property
    def end_bar(self) -> int:
        return self.start_bar + self.bars

    @property
    def start_tick(self) -> int:
        return self.start_bar * TICKS_PER_BAR

    @property
    def length_ticks(self) -> int:
        return self.bars * TICKS_PER_BAR

    def plays(self, role: str) -> bool:
        return role in self.active

    def __str__(self) -> str:
        parts = "+".join(self.active) if self.active else "silence"
        return f"{self.name} [{self.bars}b @ {self.energy:.2f}] {parts}"


# --------------------------------------------------------------------------
# Templates
# --------------------------------------------------------------------------

# (name, bars, energy, active roles, development stage, direction, throw)
_DUB_TEMPLATE = [
    ("intro", 8, 0.20, ("drums", "sub"), 0,
     "Drums and bass only. Let the riddim establish before anything else "
     "arrives.", False),
    ("riddim_a", 16, 0.55, ("drums", "perc", "sub", "chords"), 0,
     "Skank enters. This is the tune's home state.", False),
    ("version", 16, 0.70, ("drums", "perc", "sub", "chords", "lead"), 1,
     "Melody over the riddim. First development of the motif.", False),
    ("dub_out", 8, 0.30, ("drums", "sub"), 2,
     "Everything drops but drums and bass. Throw the chord stabs into the "
     "delay and let them wash. This is the moment the Nostalgia earns its "
     "rack space.", True),
    ("riddim_b", 16, 0.65, ("drums", "perc", "sub", "chords", "pad", "lead"), 3,
     "Back in, with the pad added for weight and the lead answering the "
     "skank.", False),
    ("breakdown", 8, 0.15, ("pad", "chords"), 4,
     "Drums and bass out. Just harmony and space - the hole before the last "
     "section.", True),
    ("riddim_c", 16, 0.85, ("drums", "perc", "sub", "chords", "pad", "lead"), 5,
     "Full arrangement, highest energy. Everything that has been introduced, "
     "playing at once.", False),
    ("outro", 8, 0.25, ("drums", "sub"), 6,
     "Strip back to the foundation and fade. Ride the delay feedback up as "
     "it goes.", True),
]

_DUBSTEP_TEMPLATE = [
    ("intro", 8, 0.20, ("drums", "pad"), 0,
     "Atmosphere and a skeleton beat. No sub yet - save it.", False),
    ("build", 8, 0.45, ("drums", "pad", "chords"), 1,
     "Harmony enters, filter opens across the section, drums thicken toward "
     "the drop.", False),
    ("drop", 16, 0.95, ("drums", "sub", "chords"), 2,
     "Sub arrives. Everything else gets out of its way - this is why the "
     "chord register sits above C3.", False),
    ("breakdown", 8, 0.25, ("pad", "lead"), 3,
     "Sub and drums cut dead. Pad and a fragment of the melody, drenched.", True),
    ("build_2", 8, 0.55, ("drums", "pad", "chords", "lead"), 4,
     "Second build. Reuse the first build's shape so it reads as a return.",
     False),
    ("drop_2", 16, 1.00, ("drums", "perc", "sub", "chords", "lead"), 5,
     "Second drop, denser than the first, with the lead riding it. Vary the "
     "sub figure, keep the rhythm recognisable.", False),
    ("outro", 8, 0.20, ("drums", "sub"), 6,
     "Halve the elements every 4 bars and let the delay carry the tail.",
     True),
]

# One long drop instead of two. The build earns it; the drop is the record.
_DUBSTEP_SINGLE_DROP = [
    ("intro", 8, 0.20, ("drums", "pad"), 0,
     "Atmosphere and a skeleton beat.", False),
    ("build", 16, 0.50, ("drums", "pad", "chords", "lead"), 1,
     "A long build - twice the usual. Everything enters over sixteen bars "
     "so the drop lands on a room that is already moving.", False),
    ("drop", 32, 0.95, ("drums", "perc", "sub", "chords", "lead"), 2,
     "Thirty-two bars of drop. The sub figure swaps at the halfway point; "
     "the lead comes and goes over it.", False),
    ("breakdown", 8, 0.20, ("pad", "lead"), 3,
     "Cut to nothing. Delay tail.", True),
    ("outro", 8, 0.30, ("drums", "sub"), 4,
     "Drums and sub, thinning out.", True),
]

# Starts on the drop. No intro, no build - the first bar is the loudest.
_DUBSTEP_COLD_OPEN = [
    ("drop", 16, 0.90, ("drums", "sub", "chords"), 0,
     "Cold open. First bar of the track is the drop.", False),
    ("breakdown", 8, 0.25, ("pad", "chords", "lead"), 1,
     "The breath after. Pad, stabs and a fragment of lead, drenched.", True),
    ("build", 8, 0.50, ("drums", "pad", "chords", "lead"), 2,
     "Now the build - it reads as a return because the drop came first.",
     False),
    ("drop_2", 16, 1.00, ("drums", "perc", "sub", "chords", "lead"), 3,
     "Full drop with the lead riding it.", False),
    ("dub_out", 8, 0.35, ("drums", "sub"), 4,
     "Drums and sub only. Throw everything else into the delay.", True),
    ("outro", 8, 0.20, ("sub", "pad"), 5,
     "Sub and pad fade. No drums.", True),
]

# Slow and patient. Long sections, a proper mid-track breakdown.
_DUBSTEP_PATIENT = [
    ("intro", 16, 0.15, ("pad", "perc"), 0,
     "Sixteen bars of pad and shaker. No kick yet.", False),
    ("build", 16, 0.45, ("drums", "pad", "chords"), 1,
     "The beat comes in. Sixteen bars to get there.", False),
    ("drop", 16, 0.90, ("drums", "sub", "chords", "lead"), 2,
     "First drop, lead over it.", False),
    ("breakdown", 16, 0.20, ("pad", "lead", "perc"), 3,
     "A long breakdown - half the record's tension lives here.", True),
    ("drop_2", 16, 1.00, ("drums", "perc", "sub", "chords", "lead"), 4,
     "Everything. The only time all six parts play together.", False),
    ("outro", 8, 0.25, ("drums", "sub"), 5,
     "Strip and fade.", True),
]

# Roots dub with a long riddim and a desk-driven second half.
_DUB_VERSION = [
    ("intro", 4, 0.20, ("drums",), 0,
     "Four bars of drums. Just the one-drop.", False),
    ("riddim", 32, 0.60, ("drums", "perc", "sub", "chords"), 0,
     "The riddim, for a long time. Let it settle - this is the tune.",
     False),
    ("version", 16, 0.75, ("drums", "perc", "sub", "chords", "lead"), 1,
     "The melody, over the same riddim.", False),
    ("dub_out", 16, 0.30, ("drums", "sub"), 2,
     "Sixteen bars of drums and bass. Delay throws on every fourth bar.",
     True),
    ("riddim_b", 16, 0.70, ("drums", "perc", "sub", "chords", "pad", "lead"), 3,
     "Back in, with pad for weight and the lead returning.", False),
    ("outro", 8, 0.25, ("drums", "sub"), 4,
     "Fade on the foundation.", True),
]

# Short and immediate. No intro, straight into the skank.
_DUB_DIRECT = [
    ("riddim_a", 16, 0.55, ("drums", "perc", "sub", "chords"), 0,
     "Starts on the riddim - skank from bar one.", False),
    ("version", 16, 0.70, ("drums", "perc", "sub", "chords", "lead"), 1,
     "Melody.", False),
    ("dub_out", 8, 0.30, ("drums", "sub"), 2,
     "Drop to drums and bass.", True),
    ("breakdown", 8, 0.15, ("pad",), 3,
     "Pad alone.", True),
    ("riddim_c", 16, 0.85, ("drums", "perc", "sub", "chords", "pad", "lead"), 4,
     "Everything.", False),
    ("outro", 8, 0.25, ("drums", "sub"), 5,
     "Fade.", True),
]

TEMPLATES = {
    "dub": _DUB_TEMPLATE,
    "dub_version": _DUB_VERSION,
    "dub_direct": _DUB_DIRECT,
    "dubstep": _DUBSTEP_TEMPLATE,
    "dubstep_single_drop": _DUBSTEP_SINGLE_DROP,
    "dubstep_cold_open": _DUBSTEP_COLD_OPEN,
    "dubstep_patient": _DUBSTEP_PATIENT,
}

# Which templates belong to which idiom. A profile names a family and the
# seed picks a shape from it.
FAMILIES = {
    "dub": ("dub", "dub_version", "dub_direct"),
    "dubstep": ("dubstep", "dubstep_single_drop", "dubstep_cold_open",
                "dubstep_patient"),
    "garage": ("dubstep", "dubstep_single_drop", "dubstep_patient"),
}


@dataclass
class Arrangement:
    """An ordered list of sections with absolute bar positions."""

    sections: list[Section] = field(default_factory=list)
    name: str = ""

    @property
    def bars(self) -> int:
        return sum(s.bars for s in self.sections)

    @property
    def length_ticks(self) -> int:
        return self.bars * TICKS_PER_BAR

    def at_bar(self, bar: int) -> Section | None:
        for s in self.sections:
            if s.start_bar <= bar < s.end_bar:
                return s
        return None

    def at_tick(self, tick: int) -> Section | None:
        return self.at_bar(tick // TICKS_PER_BAR)

    def roles_used(self) -> list[str]:
        """Every role that plays at least once, in canonical order."""
        used = {r for s in self.sections for r in s.active}
        return [r for r in ROLES if r in used]

    def duration_seconds(self, bpm: int) -> float:
        return self.bars * 4 * 60.0 / bpm

    def describe(self, bpm: int | None = None) -> str:
        lines = []
        for s in self.sections:
            bar_range = f"{s.start_bar + 1:>3}-{s.end_bar:<3}"
            meter = "#" * int(round(s.energy * 20))
            throw = "  <<throw>>" if s.dub_throw else ""
            lines.append(f"  {bar_range} {s.name:<12} {meter:<20}{throw}")
            if s.direction:
                lines.append(f"      {s.direction}")
        header = f"{self.name} - {self.bars} bars"
        if bpm:
            total = self.duration_seconds(bpm)
            header += f", {int(total // 60)}:{int(total % 60):02d} at {bpm}bpm"
        return header + "\n" + "\n".join(lines)


def build(template: str = "dub", bars: int | None = None, name: str = "") -> Arrangement:
    """Build an arrangement from a template, optionally scaled to `bars`.

    Scaling keeps section proportions and snaps every section to a multiple of
    4 bars, because sections that are not multiples of 4 fight the phrasing.
    """
    if template not in TEMPLATES:
        raise ValueError(
            f"unknown template {template!r}; known: {', '.join(sorted(TEMPLATES))}"
        )
    rows = TEMPLATES[template]
    natural = sum(r[1] for r in rows)
    scale = 1.0 if bars is None else bars / natural

    sections: list[Section] = []
    cursor = 0
    for row_name, row_bars, energy, active, development, direction, throw in rows:
        scaled = max(4, int(round(row_bars * scale / 4.0)) * 4)
        section = Section(
            name=row_name,
            bars=scaled,
            energy=energy,
            active=tuple(active),
            development=development,
            direction=direction,
            start_bar=cursor,
            dub_throw=throw,
            # A build climbs; a plateau is a build that never arrives.
            ramp=0.5 if row_name.startswith("build") else 0.0,
        )
        sections.append(section)
        cursor += scaled

    return Arrangement(sections, name or template)


def sample(
    family: str,
    rng,
    bars: int | None = None,
    drop_roles: tuple[str, ...] = (),
) -> Arrangement:
    """Draw one arrangement from a family, then perturb it.

    This is the difference between every song having the same skeleton and
    each one having its own: the template gives a musically vetted shape,
    and the mutations - a section stretched or halved, an energy nudged, a
    role left out entirely - make two songs from the same template still
    read as two songs.
    """
    if family not in FAMILIES:
        raise ValueError(
            f"unknown family {family!r}; known: {', '.join(sorted(FAMILIES))}"
        )
    template = rng.choice(FAMILIES[family])
    arrangement = build(template, bars=bars, name=template)
    return mutate(arrangement, rng, drop_roles=drop_roles)


def mutate(
    arrangement: Arrangement,
    rng,
    drop_roles: tuple[str, ...] = (),
    length_ops: int | None = None,
) -> Arrangement:
    """Perturb an arrangement in place and return it.

    - Every section's energy is nudged by up to +/-0.08, so the same
      template does not always produce the same density curve.
    - One or two non-peak sections are doubled or halved in length.
    - `drop_roles` are removed from every section's active set. Leaving the
      lead out of a track, or the pad, is one of the strongest ways two
      records with the same bones stop sounding like each other.

    Section lengths stay multiples of 4, and the drop is never shortened -
    a four-bar drop is not a drop.
    """
    sections = arrangement.sections
    if not sections:
        return arrangement

    for s in sections:
        s.energy = max(0.05, min(1.0, s.energy + rng.uniform(-0.08, 0.08)))

    ops = rng.randint(0, 2) if length_ops is None else length_ops
    candidates = [s for s in sections if s.energy < 0.85 and s.bars >= 4]
    for _ in range(min(ops, len(candidates))):
        s = rng.choice(candidates)
        candidates.remove(s)
        if s.bars >= 16 or (s.bars >= 8 and rng.random() < 0.5):
            s.bars //= 2
        else:
            s.bars *= 2
        s.bars = max(4, (s.bars // 4) * 4)

    if drop_roles:
        dropped = set(drop_roles)
        for s in sections:
            s.active = tuple(r for r in s.active if r not in dropped)
        # A section with nothing left in it is not a section.
        arrangement.sections = [s for s in sections if s.active]
        sections = arrangement.sections

    cursor = 0
    for s in sections:
        s.start_bar = cursor
        cursor += s.bars
    return arrangement


# --------------------------------------------------------------------------
# Energy -> parameters
# --------------------------------------------------------------------------

# What a section *does* in the song. Everything that shapes tension - the
# hook's treatment, the build roll, the gap before a drop, the impact on its
# downbeat - asks this rather than the name, so a section the UI renamed
# still behaves. Names win where they are explicit; energy decides the rest.
KINDS = ("intro", "build", "peak", "release", "body", "outro")
_KIND_PREFIXES = (
    ("intro", "intro"), ("outro", "outro"), ("build", "build"),
    ("peak", "peak"), ("drop", "peak"), ("dub_out", "release"),
    ("breakdown", "release"), ("break", "release"), ("version", "body"),
    ("groove", "body"),
    ("riddim", "body"),
)


def kind_of(section: Section) -> str:
    """The section's job: intro, build, peak, release, body or outro."""
    name = section.name.lower()
    for prefix, kind in _KIND_PREFIXES:
        if name.startswith(prefix):
            if kind == "body" and section.energy >= 0.8:
                return "peak"
            return kind
    if section.energy >= 0.8:
        return "peak"
    if section.energy <= 0.35:
        return "release"
    return "body"


def density_for(energy: float, role: str) -> float:
    """How full a part should be at a given energy, 0..1.

    Roles respond differently: drums thicken a lot with energy, the sub barely
    changes because the foundation should stay recognisable, and the lead is
    all-or-nothing.
    """
    slope = {
        "drums": (0.35, 1.0),
        "perc": (0.20, 1.0),
        "sub": (0.65, 1.0),
        "chords": (0.40, 1.0),
        "pad": (0.60, 1.0),
        "lead": (0.30, 1.0),
    }.get(role, (0.4, 1.0))
    low, high = slope
    return low + (high - low) * max(0.0, min(1.0, energy))


def brightness_for(energy: float) -> float:
    """Filter openness 0..1. Low energy sections sit darker.

    Not linear: the top of the range opens fast, so a drop actually sounds
    like the lid came off.
    """
    e = max(0.0, min(1.0, energy))
    return 0.25 + 0.75 * (e ** 1.5)


def velocity_for(energy: float, base: int = 100) -> int:
    """Scale a base velocity by section energy."""
    return max(1, min(127, int(round(base * (0.72 + 0.35 * energy)))))


def bar_energy(section: Section, bar: int) -> float:
    """The energy at one bar into the section.

    A flat section sits on its energy everywhere. A ramped one climbs from
    `energy - ramp` up to `energy` across its span, so a build thickens bar
    by bar instead of jumping to a plateau: the drop still gets the full
    level, but the approach is audible in the density and velocity curve.
    """
    if section.ramp <= 0 or section.bars <= 1:
        return section.energy
    pos = min(1.0, bar / (section.bars - 1))
    return max(0.05, section.energy - section.ramp * (1.0 - pos))


# --------------------------------------------------------------------------
# Generative: an energy grammar
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class StructureParams:
    """The knobs of the structure generator.

    A song is a sequence of *phases* at energy levels - low, mid, high - and
    the grammar is small: an optional intro, then a body of build-and-peak
    cycles separated by releases, then an outro. What makes it a dub record
    rather than a pop record is the *release*: a hole where the mix drops to
    drums and bass (a dub-out) or to pad alone (a breakdown), which is the
    desk doing the arranging.

    Roles enter *subtractively*: each has an energy threshold, and a section
    plays every role whose threshold it clears, minus whatever the hole
    removes. `sub_threshold` is the genre switch - a dub bassline is the
    foundation and enters at once; a dubstep sub is saved for the drop.
    """

    total_bars: int = 72
    peaks: int = 2                 # build-and-peak cycles
    intro: bool = True
    cold_open: bool = False        # start on the peak
    release: str = "dub_out"       # "dub_out" | "breakdown" | "mixed"
    long_peak: bool = False        # 32-bar peaks
    sub_threshold: float = 0.3
    lead_threshold: float = 0.5
    perc_threshold: float = 0.6
    pad_threshold: float = 0.0     # pads in the quiet bits by default
    chords_threshold: float = 0.35
    drums_threshold: float = 0.12
    energy_jitter: float = 0.06
    # Chance of a groove section before a build - the riddim settling in
    # before it climbs. Changes the shape of a record more than any length.
    grooves: float = 0.0
    # Roles left out of the whole song. A record with no lead, or no pad, is
    # a different record - this is the strongest single variation there is.
    omit: tuple[str, ...] = ()
    # A live track: a palette of short loops for jam mode - straight in at
    # energy, the low point only where the performer launches it, no long
    # intro or outro. See `_live`.
    live: bool = False


# Energy bands per phase kind, and the bar lengths each may take.
_PHASE = {
    "intro":     ((0.12, 0.28), (4, 8, 8, 16)),
    "build":     ((0.42, 0.62), (8, 8, 16)),
    "peak":      ((0.86, 1.00), (16, 16, 32)),
    "dub_out":   ((0.22, 0.38), (8, 8, 16)),
    "breakdown": ((0.10, 0.25), (8, 8, 16)),
    "outro":     ((0.15, 0.30), (4, 8, 8)),
    # A groove: the riddim playing with the tune over it, at a level - not
    # building, not peaking. Records breathe in these.
    "groove":    ((0.50, 0.68), (8, 16, 16)),
}


def _phases(p: StructureParams, rng) -> list[str]:
    seq: list[str] = []
    if p.cold_open:
        seq.append("peak")
        for _ in range(p.peaks - 1):
            seq += [_release(p, rng), "build", "peak"]
        seq.append(_release(p, rng))
    else:
        if p.intro:
            seq.append("intro")
        if rng.random() < p.grooves:
            seq.append("groove")
        seq += ["build", "peak"]
        for _ in range(p.peaks - 1):
            seq.append(_release(p, rng))
            if rng.random() < p.grooves:
                seq.append("groove")
            seq += ["build", "peak"]
    seq.append("outro")
    return seq


def _release(p: StructureParams, rng) -> str:
    if p.release == "mixed":
        return rng.choice(("dub_out", "breakdown"))
    return p.release


def _roles_for(kind: str, energy: float, p: StructureParams) -> tuple[str, ...]:
    thresholds = {
        "drums": p.drums_threshold, "sub": p.sub_threshold,
        "chords": p.chords_threshold, "pad": p.pad_threshold,
        "lead": p.lead_threshold, "perc": p.perc_threshold,
    }
    active = {r for r, t in thresholds.items() if energy >= t}
    active -= set(p.omit)
    # The holes. These are the arrangement.
    if kind == "dub_out":
        active = {"drums", "sub"}
    elif kind == "breakdown":
        active -= {"drums", "sub", "perc"}
        active |= {"pad"}
        if energy >= p.lead_threshold * 0.5:
            active.add("lead")   # a fragment of melody over the hole
    elif kind == "intro":
        # An intro establishes one thing. Drums for dub, atmosphere for
        # dubstep - decided by whether the sub is a foundation instrument.
        active = {"drums", "sub"} if p.sub_threshold <= 0.3 else {"drums", "pad"}
    elif kind == "outro":
        active &= {"drums", "sub", "pad"}
        if not active:
            active = {"sub"}
    elif kind == "peak":
        if energy < 0.95:
            active.discard("pad")
    if kind in ("intro", "build", "outro") and "lead" not in p.omit:
        # The hook is teased before it is given: once, quietly, at the end
        # of the intro, and as a climbing fragment through the build - and
        # remembered as the song leaves: the outro's cell, then its head,
        # then nothing (the coda). The theme plan keeps all three sparse -
        # this only lets the lead be there.
        active.add("lead")
    active -= set(p.omit)
    if not active:
        active = {"drums"} - set(p.omit) or {"sub"}
    return tuple(r for r in ROLES if r in active)


_DIRECTION = {
    "intro": "Establish one thing and hold it.",
    "build": "Parts enter one at a time. Density rises toward the peak.",
    "peak": "Everything the section can carry. The sub is the event.",
    "dub_out": "Drums and bass only. Throw the last stab into the delay.",
    "breakdown": "Rhythm section out. Harmony and space.",
    "outro": "Strip to the foundation and fade.",
    "groove": "The riddim at a level, the tune over it. Settle before the climb.",
}


def generate(
    p: StructureParams,
    rng,
    name: str = "generated",
) -> Arrangement:
    """Generate an arrangement from the energy grammar.

    Phase lengths are drawn from each phase's allowed set, then the whole
    thing is scaled to `total_bars` and snapped to multiples of 4 - except
    peaks, which are never shortened below 16 bars. Energies are drawn from
    the phase band and jittered so two songs with the same phase sequence
    still have different curves.
    """
    if p.live:
        return _live(p, rng, name)
    kinds = _phases(p, rng)
    lengths = []
    for k in kinds:
        options = _PHASE[k][1]
        if k == "peak" and p.long_peak:
            options = (32,)
        lengths.append(rng.choice(options))

    natural = sum(lengths)
    scale = p.total_bars / natural
    counts: dict[str, int] = {}
    sections: list[Section] = []
    cursor = 0
    dev = 0
    # A short sketch (under 48 bars) is allowed 8-bar peaks and 4-bar
    # sections; a full track snaps to 16/32 and eights.
    short = p.total_bars < 48
    for k, raw in zip(kinds, lengths):
        scaled = raw * scale
        if k == "peak":
            # Peaks are 16 or 32. A 36-bar drop is a 32-bar drop that lost
            # count; a 12-bar one is not a drop.
            if short:
                bars = 16 if scaled >= 12 else 8
            else:
                bars = 32 if scaled >= 24 else 16
        elif short and scaled < 6:
            bars = 4
        elif short:
            bars = max(4, int(round(scaled / 4.0)) * 4)
        else:
            # Everything else in eights: the drum fills land every 8 and
            # the bass phrases every 4, and a 12-bar section fights both.
            bars = max(8, int(round(scaled / 8.0)) * 8)
        lo, hi = _PHASE[k][0]
        # Rounded here, not on write-back: the params must reproduce the
        # song to the bit, and a velocity computed from 0.93389 is not the
        # velocity computed from 0.934.
        energy = round(max(0.05, min(1.0, rng.uniform(lo, hi)
                                     + rng.uniform(-p.energy_jitter, p.energy_jitter))), 3)
        # Only builds climb. Everything else sits on its level - a peak that
        # rambled up from below would stop reading as a drop.
        ramp = round(rng.uniform(0.4, 0.85), 2) if k == "build" else 0.0
        counts[k] = counts.get(k, 0) + 1
        label = k if counts[k] == 1 else f"{k}_{counts[k]}"
        sections.append(Section(
            name=label, bars=bars, energy=energy,
            active=_roles_for(k, energy, p), development=dev,
            direction=_DIRECTION[k], start_bar=cursor,
            dub_throw=k in ("dub_out", "breakdown", "outro"),
            ramp=ramp,
        ))
        cursor += bars
        dev += 1
    # Drops escalate: each peak clears the one before, so the last drop is
    # the biggest thing in the record. Same energies, re-dealt in order.
    peaks = [s for s in sections if kind_of(s) == "peak"]
    for s, e in zip(peaks, sorted(s.energy for s in peaks)):
        s.energy = e
    # And the last drop is the one place the whole band plays: the pad
    # joins it (unless the song goes without one) - the final chorus earns
    # a layer nothing before it had.
    if len(peaks) >= 2 and "pad" not in p.omit:
        last = peaks[-1]
        last.active = tuple(r for r in ROLES if r in set(last.active) | {"pad"})
    if not short:
        _fit_length(sections, p.total_bars)
    return Arrangement(sections, name)


# A live track, section by section: (name, phase kind for its roles, bars,
# energy band, ramp). Every section is a loop the performer can hold (jam
# mode), so none needs to be long; the energy stays up everywhere except the
# break, which plays only when it is launched.
_LIVE = (
    ("groove", "groove", 8, (0.62, 0.70), 0.0),     # the entry: the riddim at full groove
    ("build", "build", 8, (0.50, 0.62), 0.5),
    ("drop", "peak", 16, (0.88, 0.94), 0.0),
    ("break", None, 8, (0.26, 0.36), 0.0),          # the opt-in low point
    ("build_2", "build", 8, (0.52, 0.64), 0.6),
    ("drop_2", "peak", 16, (0.92, 1.00), 0.0),
    ("exit", "groove", 8, (0.58, 0.66), 0.0),       # a groove to leave on, not a fade
)


def _live(p: StructureParams, rng, name: str) -> Arrangement:
    """A track for the stage: short loopable sections, straight in at energy.

    The entry and exit are the riddim with drums and bass from the first
    bar - something to launch into and out of, not an atmosphere and a fade.
    The break (a dub-out or a breakdown, by the song's `release`) is the only
    low point, and in jam mode it sounds only when the performer launches it.
    The second drop is the bigger one and brings the whole band."""
    sections: list[Section] = []
    cursor = 0
    for dev, (label, kind, bars, (lo, hi), ramp) in enumerate(_LIVE):
        if kind is None:
            kind = p.release if p.release in ("dub_out", "breakdown") else rng.choice(("dub_out", "breakdown"))
        energy = round(rng.uniform(lo, hi), 3)
        active = set(_roles_for(kind, energy, p))
        if label in ("groove", "exit"):
            active |= {"drums", "sub"}                 # in at energy: the riddim from bar one
        if label == "drop_2" and "pad" not in p.omit:
            active.add("pad")                          # the whole band, once
        sections.append(Section(
            name=label, bars=bars, energy=energy,
            active=tuple(r for r in ROLES if r in active), development=dev,
            direction=_DIRECTION.get(kind, ""), start_bar=cursor,
            dub_throw=kind in ("dub_out", "breakdown"), ramp=ramp,
        ))
        cursor += bars
    return Arrangement(sections, name)


# Which sections grow first when snapping left a song short, and how far.
# Peaks are the record, so they take the first sixteen; then the intro and
# outro grow to sixteen (room to mix in and out); builds stop at sixteen
# because a roll covers eight and a longer climb stops climbing.
_GROW = (("peak", 32, 16), ("intro", 16, 8), ("outro", 16, 8),
         ("body", 32, 8), ("release", 16, 8), ("build", 16, 8))


def _fit_length(sections: list[Section], target: int) -> None:
    """Grow (or trim) sections until the song is within a phrase of
    `target` bars, then re-lay the start bars."""
    caps = {k: cap for k, cap, _ in _GROW}
    for s in sections:
        cap = caps.get(kind_of(s))
        if cap is not None and s.bars > cap:
            s.bars = cap
    total = sum(s.bars for s in sections)
    guard = 0
    while total < target - 4 and guard < 32:
        guard += 1
        grown = False
        for kind, cap, by in _GROW:
            cands = [s for s in sections if kind_of(s) == kind and s.bars + by <= cap]
            if cands:
                pick = min(cands, key=lambda s: s.bars)
                pick.bars += by
                total += by
                grown = True
                break
        if not grown:
            break
    # And trim one that came out long: a slow tempo makes the same bars a
    # longer record. Peaks keep sixteen, everything else keeps eight.
    while total > target + 8:
        cands = [s for s in sections
                 if s.bars > (16 if kind_of(s) == "peak" else 8)]
        if not cands:
            break
        pick = max(cands, key=lambda s: (s.bars, kind_of(s) != "peak"))
        pick.bars -= 8
        total -= 8
    cursor = 0
    for s in sections:
        s.start_bar = cursor
        cursor += s.bars


def _bars_for_minutes(rng, bpm: int | None, lo: float, hi: float, default) -> int:
    if not bpm:
        return rng.choice(default)
    minutes = rng.uniform(lo, hi)
    return max(48, int(round(minutes * bpm / 4 / 8)) * 8)


def params_for_family(family: str, rng, bpm: int | None = None) -> StructureParams:
    """Draw structure parameters for a genre family. Ranges, not templates.

    Given the tempo, the length is drawn in *minutes* - three and a half to
    five and a bit - and turned into bars, so a 72bpm dub and an 88bpm
    half-time track are both records you can play out.
    """
    if family == "dubstep":
        # 128-176 bars at 140 is 3:40-5:00 - a record you can play out, with
        # sixteen-bar intros and outros a DJ (or the next track) can mix over.
        # Anything shorter snaps sections to four bars, which at this tempo
        # is seven seconds: a sketch, not a song. Two drops is the form; a
        # long track earns a third, a cold open always gets a second.
        total = _bars_for_minutes(rng, bpm, 3.7, 5.0, (128, 144, 144, 160))
        cold = rng.random() < 0.2
        peaks = 3 if total >= 168 and rng.random() < 0.5 else 2
        if not cold and total <= 136 and rng.random() < 0.15:
            peaks = 1
        return StructureParams(
            total_bars=total,
            peaks=peaks,
            intro=not cold and rng.random() < 0.85,
            cold_open=cold,
            release=rng.choice(("breakdown", "breakdown", "dub_out", "mixed")),
            long_peak=rng.random() < 0.3,
            sub_threshold=0.75, lead_threshold=rng.uniform(0.45, 0.6),
            perc_threshold=rng.uniform(0.55, 0.8), chords_threshold=0.4,
            omit=_draw_omit(rng),
            grooves=rng.choice((0.0, 0.25, 0.5)),
        )
    if family == "garage":
        # A garage record is built for the mix: a drum intro to blend over,
        # two drops of the bassline, a breakdown for the hook to carry, and
        # the bass dropping out and slamming back as the main trick. The
        # chords play almost throughout - the Rhodes is the record's warmth.
        total = _bars_for_minutes(rng, bpm, 3.8, 5.2, (128, 144, 160))
        return StructureParams(
            total_bars=total,
            peaks=3 if total >= 160 and rng.random() < 0.5 else 2,
            intro=rng.random() < 0.9,
            cold_open=False,
            release=rng.choice(("breakdown", "breakdown", "mixed")),
            long_peak=rng.random() < 0.25,
            sub_threshold=rng.uniform(0.45, 0.6), lead_threshold=rng.uniform(0.4, 0.55),
            perc_threshold=rng.uniform(0.4, 0.6), chords_threshold=0.25,
            omit=_draw_omit(rng),
            grooves=rng.choice((0.25, 0.5, 0.5)),
        )
    total = _bars_for_minutes(rng, bpm, 3.6, 5.4, (72, 80, 96, 104))
    cold = rng.random() < 0.12
    return StructureParams(
        total_bars=total,
        peaks=max(2 if cold else 1, rng.choices((1, 2, 3), (2, 4, 1))[0]),
        intro=not cold and rng.random() < 0.8,
        cold_open=cold,
        release=rng.choice(("dub_out", "dub_out", "mixed")),
        long_peak=rng.random() < 0.2,
        sub_threshold=0.15, lead_threshold=rng.uniform(0.5, 0.7),
        perc_threshold=rng.uniform(0.45, 0.65), chords_threshold=0.35,
        pad_threshold=rng.choice((0.0, 0.6)),
        omit=_draw_omit(rng),
        grooves=rng.choice((0.25, 0.5, 0.75)),
    )


def _draw_omit(rng) -> tuple[str, ...]:
    """Which roles this song goes without. The lead only rarely."""
    out = []
    # Rare: the hook is the song, and a show set wants it sung. When the
    # lead is left out the bass carries the hook through the peaks.
    if rng.random() < 0.04:
        out.append("lead")
    if rng.random() < 0.3:
        out.append("pad")
    if rng.random() < 0.3:
        out.append("perc")
    return tuple(out)
