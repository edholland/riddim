"""Genre profiles: everything that differs between roots dub and 140 dubstep.

The theory modules are genre-neutral. This is where the idiom lives, so that
switching feel is one flag rather than a rewrite. If a constant is
genre-specific it belongs here and nowhere else.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .theory import pitch


@dataclass(frozen=True)
class Register:
    """A pitch window for a part."""

    low: int
    high: int

    @property
    def centre(self) -> int:
        return (self.low + self.high) // 2

    def __str__(self) -> str:
        return f"{pitch.note_name(self.low)}..{pitch.note_name(self.high)}"


@dataclass(frozen=True)
class Profile:
    """A complete genre setting."""

    name: str
    bpm: int
    modes: tuple[str, ...]
    # Pools. A profile is an idiom - a distribution of choices - and a seed
    # draws one song from it. Every field that is a tuple here is something
    # that used to be a single fixed value, which is why every song from a
    # profile had the same skeleton.
    drum_styles: tuple[str, ...]
    bass_figures: tuple[str, ...]
    stab_patterns: tuple[str, ...] = ("skank",)
    bars_per_chord_choices: tuple[float, ...] = (2.0,)
    # Roles a song may leave out entirely. Leaving the lead off a record is
    # one of the strongest ways two tracks with the same bones stop sounding
    # like each other.
    optional_roles: tuple[str, ...] = ("lead", "pad", "perc")
    optional_role_chance: float = 0.35
    # The lead is treated separately: leaving it out entirely is a legitimate
    # choice for a dub record but it should be the exception. At the general
    # 35% chance, combined with the lead only playing in a few sections, it
    # was effectively never heard.
    lead_omit_chance: float = 0.12
    # How the lead phrases. See compose._build_lead.
    lead_styles: tuple[str, ...] = ("call_response", "long_tones", "riff")
    # Structure family - see arrange.structure.FAMILIES.
    family: str = "dub"
    swing: float = 0.0
    bars_per_chord: float = 2.0
    chord_size: int = 3
    progression_tension: float = 0.35
    # Registers. The sub window is deliberately narrow - one oscillator, and
    # anything above about C2 stops reading as sub bass.
    sub: Register = Register(28, 48)       # E0 .. C2
    chords: Register = Register(55, 79)    # G2 .. G4
    pad: Register = Register(48, 72)       # C2 .. C4
    lead: Register = Register(67, 91)      # G3 .. G5
    # Arrangement
    section_template: str = "dub"
    bass_note_length: float = 1.0   # multiplier on figure note lengths
    skank_length: float = 0.45      # in steps; short is the point
    # Length of the *felt* beat in 16th steps. Four in a straight feel; eight
    # in half-time, where the backbeat is on 3 and the music is felt at half
    # the written tempo. Drives where the skank lands - see rhythm.is_offbeat.
    beat_steps: int = 4
    delay_throw_on: tuple[str, ...] = ("dub_out", "breakdown", "outro")
    note: str = ""

    @property
    def drum_style(self) -> str:
        """The idiom's default drum style - the first in the pool."""
        return self.drum_styles[0]

    @property
    def feel(self) -> str:
        """'half_time' or 'straight' - selects the stab-pattern table."""
        return "half_time" if self.beat_steps >= 8 else "straight"


ROOTS_DUB = Profile(
    name="roots_dub",
    bpm=72,
    modes=("minor", "dorian", "phrygian"),
    drum_styles=("one_drop", "one_drop_flyer", "one_drop_pickup", "rockers"),
    bass_figures=("one_drop_walk", "anticipation", "octave_drop", "rolling_dub",
                  "dub_walk", "riddim_bounce", "fifth_pedal", "question_answer",
                  "minor_third_bounce", "descending_walk"),
    stab_patterns=("skank", "double_chop", "half_skank", "three_four",
                   "late_skank"),
    bars_per_chord_choices=(1.0, 2.0, 2.0, 4.0),
    family="dub",
    swing=0.16,
    bars_per_chord=2.0,
    progression_tension=0.25,
    section_template="dub",
    skank_length=0.4,
    note="72bpm one-drop. Slow harmonic rhythm, offbeat skank, bass enters "
         "late in the bar. The mix is the arrangement.",
)

STEPPERS_DUB = Profile(
    name="steppers_dub",
    bpm=78,
    modes=("minor", "dorian"),
    drum_styles=("steppers", "steppers_skip", "rockers"),
    bass_figures=("steppers_pulse", "tresillo_sub", "rolling_dub", "dub_walk",
                  "fifth_pedal", "triplet_halves", "syncopated_stabs",
                  "drop_and_climb"),
    stab_patterns=("skank", "double_chop", "half_skank"),
    bars_per_chord_choices=(2.0, 2.0, 4.0),
    family="dub",
    swing=0.1,
    bars_per_chord=2.0,
    progression_tension=0.3,
    section_template="dub",
    note="Four-to-the-floor militant dub. More forward motion than one-drop, "
         "bass pulses with the kick.",
)

DUBSTEP_140 = Profile(
    name="dubstep_140",
    bpm=140,
    modes=("minor", "phrygian", "harmonic_minor"),
    drum_styles=("half_time", "half_time_push", "half_time_double",
                 "half_time_tresillo", "half_time_roll"),
    bass_figures=("half_time_roll", "dotted_push", "octave_bounce",
                  "phrygian_creep", "neighbour_slide", "reese_answer",
                  "descending_walk", "syncopated_stabs", "triplet_halves",
                  "question_answer", "long_then_busy", "drop_and_climb",
                  "tresillo_sub", "half_time_sub"),
    stab_patterns=("offbeats", "push", "single", "answer", "tresillo",
                   "late_pair"),
    bars_per_chord_choices=(2.0, 4.0, 4.0, 8.0),
    family="dubstep",
    swing=0.0,
    bars_per_chord=4.0,      # very slow harmony; the sub is the event
    progression_tension=0.6,
    chord_size=3,
    sub=Register(24, 45),    # C0 .. A1 - deeper than roots dub
    chords=Register(55, 76),
    section_template="dubstep",
    skank_length=0.35,
    beat_steps=8,
    note="Written at 140, felt at 70. Kick on 1, snare on 3, and the bar is "
         "mostly empty so the sub can be the whole event.",
)

DUBBY_HALFTIME = Profile(
    name="dubby_halftime",
    bpm=88,
    modes=("minor", "dorian", "phrygian"),
    drum_styles=("half_time", "half_time_push", "half_time_sparse"),
    bass_figures=("half_time_roll", "anticipation", "dotted_push",
                  "reese_answer", "minor_third_bounce", "long_then_busy",
                  "half_time_sub", "sparse_stab"),
    stab_patterns=("offbeats", "push", "answer", "late_pair"),
    bars_per_chord_choices=(2.0, 4.0),
    family="dub",
    swing=0.12,
    bars_per_chord=2.0,
    progression_tension=0.45,
    section_template="dub",
    beat_steps=8,
    note="Between the two. Heavy, slow, trip-hop adjacent.",
)

PROFILES: dict[str, Profile] = {
    p.name: p
    for p in (ROOTS_DUB, STEPPERS_DUB, DUBSTEP_140, DUBBY_HALFTIME)
}

# Short aliases for the command line.
ALIASES = {
    "dub": "roots_dub",
    "roots": "roots_dub",
    "steppers": "steppers_dub",
    "dubstep": "dubstep_140",
    "140": "dubstep_140",
    "halftime": "dubby_halftime",
}


def get(name: str) -> Profile:
    key = ALIASES.get(name.lower(), name.lower())
    if key not in PROFILES:
        known = ", ".join(sorted(set(PROFILES) | set(ALIASES)))
        raise ValueError(f"unknown profile {name!r}; known: {known}")
    return PROFILES[key]


def register_for(profile: Profile, role: str) -> Register:
    """Register window for a part role."""
    mapping = {
        "sub": profile.sub,
        "bass": profile.sub,
        "chords": profile.chords,
        "skank": profile.chords,
        "pad": profile.pad,
        "lead": profile.lead,
        "melody": profile.lead,
    }
    if role not in mapping:
        raise ValueError(
            f"no register for role {role!r}; known: {', '.join(sorted(mapping))}"
        )
    return mapping[role]
