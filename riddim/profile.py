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
    # How the lead delivers the hook. See arrange.interplay.styled: "hook"
    # sings it as written, "riff" loops the cell with a turnaround,
    # "call_response" rests every other unit for the band to answer, and
    # "long_tones" is the half-speed melodica version. Weighted by repetition.
    lead_styles: tuple[str, ...] = ("hook", "hook", "call_response", "riff", "long_tones")
    # Structure family - see arrange.structure.FAMILIES.
    family: str = "dub"
    # Swing is set on the Deluge, never written into the notes.
    swing: float = 0.0
    # The Deluge's swing for this feel (50 = straight, as the Deluge shows
    # it): every off-beat sixteenth moved later together. Not in the notes -
    # set it on the Deluge; the audio preview applies it the same way. Set
    # from the records (riddim.listen's swing on ~300 a family, and the
    # bass's offbeat lateness in ml/mine_cells.py, which agree): roots 54,
    # steppers and dubstep 52-53, garage near straight - speed garage 53, UK
    # garage 50, dark garage and grime 47-51.
    deluge_swing: int = 50
    # The tempo is drawn per song from this range - the same feel at 68 and
    # at 76 is two different records.
    bpm_range: tuple[int, int] = (0, 0)
    bars_per_chord: float = 2.0
    # The most chords a drop's loop may carry. A dub version or a garage
    # record runs a three- or four-chord loop happily; a 140 drop is a
    # drone or a two-chord rock (i-bII, i-bVI, i-V) and the sub is the
    # event - a four-chord anthem loop under it is a trance record.
    drop_chords: int = 4
    # What the pad track plays: "held" chords with a moving top voice, or
    # the reggae organ's "bubble" - the shuffle under the skank that is the
    # other half of reggae keys (a held string pad is the "new age" in a
    # dub record; measured, the dub records' mid band is a busy keys part).
    pad_style: str = "held"
    chord_size: int = 3
    progression_tension: float = 0.35
    # Registers. The sub window is deliberately narrow - one oscillator, and
    # anything above about C2 stops reading as sub bass.
    sub: Register = Register(28, 48)       # E0 .. C2
    chords: Register = Register(55, 79)    # G2 .. G4
    pad: Register = Register(48, 72)       # C2 .. C4
    lead: Register = Register(64, 97)      # E3 .. C#6 - room for a wide hook to keep its shape
    # Arrangement
    section_template: str = "dub"
    bass_note_length: float = 1.0   # multiplier on figure note lengths
    skank_length: float = 0.45      # in steps; short is the point
    # Length of the *felt* beat in 16th steps. Four in a straight feel; eight
    # in half-time, where the backbeat is on 3 and the music is felt at half
    # the written tempo. Drives where the skank lands - see rhythm.is_offbeat.
    beat_steps: int = 4
    delay_throw_on: tuple[str, ...] = ("dub_out", "breakdown", "outro")
    # How often the siren fires where one may (a dub-out, a later drop's
    # downbeat): the sound-system dub's signature, rarer the further from it.
    sirens: float = 0.0
    # The desk's bass pulls per 32 bars, where the profile parts from its
    # family (arrange.dynamics.params_for); None is the family's range.
    bass_drops: tuple[float, float] | None = None
    # How often a walking bassline takes its rhythm from the cells mined
    # from this profile's records (theory/cells.py) instead of the generator.
    mined_bass: float = 0.5
    # Whether the sub may sit on the in-between sixteenths (compose._on_eighths).
    sub_sixteenths: bool = False
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
    bpm_range=(66, 78),
    modes=("minor", "dorian", "phrygian"),
    drum_styles=("one_drop", "one_drop_flyer", "one_drop_pickup", "rockers"),
    bass_figures=("one_drop_walk", "anticipation", "octave_drop", "rolling_dub",
                  "dub_walk", "riddim_bounce", "fifth_pedal", "question_answer",
                  "minor_third_bounce", "descending_walk"),
    stab_patterns=("skank", "double_chop", "half_skank", "three_four",
                   "late_skank"),
    bars_per_chord_choices=(1.0, 2.0, 2.0, 4.0),
    # 300 roots dub records, felt tempo: the sub pulled ~0.6 times per 32 bars.
    bass_drops=(0.3, 0.9),
    # 80 roots records (ml/mine_cells.py, the one from beat_this) strike the a
    # of 1 and of 3 (steps 3, 11) about as often as the off-eighths - the
    # "x..x" figure; the transcriber puts no such notes into riddim's own
    # eighth-note subs, so they are the records'.
    # Every walking line from the records' fit: the generator's walks skip the
    # one in nearly every bar or in none (0.06-0.17 in steppers 201-203,
    # against the records' 0.54), and the fit won the judge each time.
    mined_bass=1.0,
    sub_sixteenths=True,
    deluge_swing=54,
    sirens=0.6,
    family="dub",
    bars_per_chord=2.0,
    progression_tension=0.25,
    section_template="dub",
    skank_length=0.4,
    pad_style="bubble",
    note="72bpm one-drop. Slow harmonic rhythm, offbeat skank, bass enters "
         "late in the bar. The mix is the arrangement.",
)

STEPPERS_DUB = Profile(
    name="steppers_dub",
    bpm=78,
    bpm_range=(74, 84),
    modes=("minor", "dorian"),
    drum_styles=("steppers", "steppers_skip", "rockers"),
    bass_figures=("steppers_pulse", "tresillo_sub", "rolling_dub", "dub_walk",
                  "fifth_pedal", "triplet_halves", "syncopated_stabs",
                  "drop_and_climb"),
    stab_patterns=("skank", "double_chop", "half_skank"),
    bars_per_chord_choices=(2.0, 2.0, 4.0),
    # 96 steppers records: steps 3 and 11 struck in 0.41 and 0.38 of bars.
    # Every walking line from the records' fit: the generator's walks skip the
    # one in nearly every bar or in none (0.06-0.17 in steppers 201-203,
    # against the records' 0.54), and the fit won the judge each time.
    mined_bass=1.0,
    sub_sixteenths=True,
    deluge_swing=52,
    sirens=0.6,
    family="dub",
    bars_per_chord=2.0,
    progression_tension=0.3,
    section_template="dub",
    pad_style="bubble",
    note="Four-to-the-floor militant dub. More forward motion than one-drop, "
         "bass pulses with the kick.",
)

DUBSTEP_140 = Profile(
    name="dubstep_140",
    bpm=140,
    bpm_range=(138, 142),
    modes=("minor", "phrygian", "harmonic_minor"),
    # The last two are dubstep's garage inheritance: the 2-step's late kick
    # and skipping hats under a half-time snare.
    drum_styles=("half_time", "half_time_push", "half_time_double",
                 "half_time_tresillo", "half_time_roll", "half_time_skip",
                 "half_time_garage"),
    bass_figures=("half_time_roll", "dotted_push", "octave_bounce",
                  "phrygian_creep", "neighbour_slide", "reese_answer",
                  "descending_walk", "syncopated_stabs", "triplet_halves",
                  "question_answer", "long_then_busy", "drop_and_climb",
                  "tresillo_sub", "half_time_sub"),
    stab_patterns=("offbeats", "push", "single", "answer", "tresillo",
                   "late_pair", "organ", "dotted"),
    bars_per_chord_choices=(2.0, 4.0, 4.0, 8.0),
    sirens=0.3,
    # The records' bass reads near-flat across the bar even with a tracked
    # downbeat (wobble and grime subs defeat the note reader), so the mined
    # fit has nothing to say here yet.
    mined_bass=0.0,
    family="dubstep",
    drop_chords=2,
    lead_styles=("hook", "hook", "hook", "riff", "call_response"),
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
    deluge_swing=53,
)

DUBBY_HALFTIME = Profile(
    name="dubby_halftime",
    bpm=88,
    bpm_range=(82, 94),
    modes=("minor", "dorian", "phrygian"),
    drum_styles=("half_time", "half_time_push", "half_time_sparse", "half_time_skip"),
    bass_figures=("half_time_roll", "anticipation", "dotted_push",
                  "reese_answer", "minor_third_bounce", "long_then_busy",
                  "half_time_sub", "sparse_stab"),
    stab_patterns=("offbeats", "push", "answer", "late_pair"),
    bars_per_chord_choices=(2.0, 4.0),
    sirens=0.4,
    family="dub",
    bars_per_chord=2.0,
    progression_tension=0.45,
    section_template="dub",
    beat_steps=8,
    note="Between the two. Heavy, slow, trip-hop adjacent.",
    deluge_swing=52,
)

UK_GARAGE = Profile(
    name="uk_garage",
    bpm=132,
    bpm_range=(130, 136),
    # Dorian is the garage minor: the raised sixth is what makes the m9
    # Rhodes chord sound warm instead of dark.
    modes=("dorian", "dorian", "minor"),
    drum_styles=("two_step", "two_step", "two_step_skip", "two_step_roll",
                 "garage_4x4"),
    bass_figures=("octave_bounce", "dotted_push", "syncopated_stabs",
                  "question_answer", "tresillo_sub"),
    stab_patterns=("organ", "two_step", "stutter", "dotted", "answer_kick"),
    bars_per_chord_choices=(1.0, 2.0, 2.0),
    lead_styles=("hook", "hook", "riff", "riff", "call_response"),
    sirens=0.1,
    family="garage",
    bars_per_chord=2.0,
    progression_tension=0.4,
    chord_size=5,
    sub=Register(28, 52),    # E0 .. E2 - the garage bass moves, so it gets room
    chords=Register(57, 81),
    section_template="dubstep",
    skank_length=1.0,        # an organ stab, not a guitar chop
    delay_throw_on=("breakdown", "outro"),
    note="2-step at 132: kick on 1 and the & of 3, hats skipping into the "
         "backbeat, minor-ninth stabs dancing round the kick, a bass that "
         "bounces with it and a chopped hook over the top.",
    deluge_swing=53,
)

DARK_GARAGE = Profile(
    name="dark_garage",
    bpm=138,
    bpm_range=(134, 140),
    # Where dubstep came from: 2-step drums, the sound system's sub and
    # minor-key menace. Horsepower, El-B, the first Skream records.
    modes=("minor", "phrygian", "minor"),
    drum_styles=("two_step", "two_step_skip", "two_step_roll"),
    bass_figures=("dotted_push", "octave_bounce", "phrygian_creep",
                  "neighbour_slide", "question_answer"),
    stab_patterns=("organ", "stutter", "answer_kick", "half_skank", "dotted"),
    bars_per_chord_choices=(2.0, 4.0, 4.0),
    lead_styles=("hook", "riff", "riff", "call_response"),
    sirens=0.2,
    # The records' bass reads near-flat across the bar even with a tracked
    # downbeat (wobble and grime subs defeat the note reader), so the mined
    # fit has nothing to say here yet.
    mined_bass=0.0,
    family="dubstep",
    drop_chords=2,
    bars_per_chord=4.0,
    progression_tension=0.55,
    sub=Register(24, 45),
    chords=Register(55, 76),
    section_template="dubstep",
    skank_length=0.6,
    note="Proto-dubstep at 138: the 2-step kit, a deep sub and dark "
         "harmony - the garage record that turned into dubstep.",
    deluge_swing=50,
)

# The Deluge kit preset each family's drums play (deluge/kit.py makes them):
# the same fifteen rows in riddim's order, the family's own sound.
KIT_PRESETS = {"dub": "RIDDIM DUB", "dubstep": "RIDDIM 140", "garage": "RIDDIM GARAGE"}


PROFILES: dict[str, Profile] = {
    p.name: p
    for p in (ROOTS_DUB, STEPPERS_DUB, DUBSTEP_140, DUBBY_HALFTIME,
              UK_GARAGE, DARK_GARAGE)
}

# Short aliases for the command line.
ALIASES = {
    "dub": "roots_dub",
    "roots": "roots_dub",
    "steppers": "steppers_dub",
    "dubstep": "dubstep_140",
    "140": "dubstep_140",
    "halftime": "dubby_halftime",
    "garage": "uk_garage",
    "ukg": "uk_garage",
    "2step": "uk_garage",
    "2-step": "uk_garage",
    "dark": "dark_garage",
    "darkgarage": "dark_garage",
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
