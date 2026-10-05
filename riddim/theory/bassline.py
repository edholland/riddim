"""Basslines - the part the Coral plays, and the part that carries the tune.

In dub the bass is not accompaniment, it is the lead instrument. So it gets
its own generator rather than being treated as a melody in a low register,
because the constraints are genuinely different:

- **It is mostly rests.** A dub bass figure is a gesture with silence around
  it, not a continuous line.
- **It is root-centric.** Root, fifth, flat seventh and octave carry it; the
  third is used sparingly because it fights the chord part.
- **It anticipates.** Landing on the "&" of 4 and holding through the barline
  is the signature push that makes a riddim lurch forward.
- **It is monophonic and in one register.** One oscillator, roughly E0-E2.
  Notes that stray above that stop being bass.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from . import pitch
from .melody import Motif, Tone
from .rhythm import Pattern, Hit, metric_weight
from .scale import Scale
from ..units import STEPS_PER_BAR

# Scale degrees a bassline may use, in order of how safe they are. Degree
# numbers are scale steps, so in minor these are root, 5th, b7, 4th, b3, b6.
BASS_DEGREES = (0, 4, 6, 3, 2, 5)
SAFE_DEGREES = (0, 4, 6, 3)


@dataclass(frozen=True)
class BassFigure:
    """A named one-bar bass template: (step, degree, length) triples."""

    name: str
    notes: tuple[tuple[float, int, float], ...]
    bars: int = 1
    note: str = ""


# Idiomatic figures. These are shapes, not transcriptions - the degrees get
# re-pointed at whatever chord is current.
FIGURES: tuple[BassFigure, ...] = (
    BassFigure(
        "root_hold",
        ((0.0, 0, 6.0), (8.0, 0, 6.0)),
        note="Two long roots. The foundation - use it when everything else "
             "is busy.",
    ),
    BassFigure(
        "one_drop_walk",
        ((6.0, 0, 2.0), (8.0, 0, 3.0), (11.0, 6, 1.0), (12.0, 4, 3.0)),
        note="Enters on the & of 2, lands with the one-drop kick on 3, walks "
             "down through the b7 to the 5th.",
    ),
    BassFigure(
        "anticipation",
        ((14.0, 0, 4.0), (4.0, 4, 2.0), (8.0, 6, 2.0)),
        note="Pushes across the barline from the & of 4. The lurch.",
    ),
    BassFigure(
        "steppers_pulse",
        ((0.0, 0, 3.0), (4.0, 0, 2.0), (8.0, 4, 3.0), (12.0, 6, 3.0)),
        note="One note per beat, militant, moves through root-5-b7.",
    ),
    BassFigure(
        "tresillo_sub",
        ((0.0, 0, 3.0), (3.0, 0, 3.0), (6.0, 0, 2.0), (12.0, 4, 3.0)),
        note="3-3-2 grouping against the 4/4. Latin cell, works everywhere.",
    ),
    BassFigure(
        "half_time_sub",
        ((0.0, 0, 7.0), (8.0, 0, 3.0), (12.0, 6, 3.0)),
        note="140bpm dubstep: one long sub through the first half, movement "
             "in the second.",
    ),
    BassFigure(
        "wobble_cell",
        ((0.0, 0, 4.0), (6.0, 0, 2.0), (8.0, 0, 4.0), (14.0, 5, 2.0)),
        note="Repeated root with rhythmic gaps - built to be modulated. Only "
             "worth using if something is actually moving the filter; with a "
             "static patch it just sits there.",
    ),
    BassFigure(
        "half_time_roll",
        ((0.0, 0, 3.0), (4.0, 0, 1.5), (6.0, 7, 1.5), (8.0, 0, 3.0),
         (12.0, 6, 1.5), (14.0, 4, 2.0)),
        note="Roots anchoring 1 and 3, an octave flick on the & of 2, then a "
             "walk down through the b7 to the 5th. Movement without ever "
             "leaving the sub register.",
    ),
    BassFigure(
        "dotted_push",
        ((0.0, 0, 3.0), (3.0, 0, 3.0), (6.0, 6, 2.0), (10.0, 0, 2.0),
         (13.0, 4, 3.0)),
        note="Dotted-eighth grouping - 3+3+4+3+3 across the bar. Pulls "
             "against the 4/4 so the bass never lands where the ear expects.",
    ),
    BassFigure(
        "octave_bounce",
        ((0.0, 0, 2.0), (2.0, 7, 1.0), (4.0, 0, 2.0), (7.0, 6, 1.0),
         (8.0, 0, 3.0), (12.0, 7, 1.5), (14.0, 4, 2.0)),
        note="Root and octave alternating with the b7 filling the gaps. Busy "
             "for a sub - wants a short envelope or it turns to mud.",
    ),
    BassFigure(
        "phrygian_creep",
        ((0.0, 0, 4.0), (6.0, 1, 2.0), (8.0, 0, 2.0), (11.0, 6, 1.0),
         (12.0, 4, 3.0)),
        note="Leans on the second degree - in phrygian that is the b2, the "
             "half-step above the root, which is the whole reason to be in "
             "the mode. Menacing.",
    ),
    BassFigure(
        "fifth_pedal",
        ((0.0, 0, 2.0), (2.0, 4, 2.0), (4.0, 0, 2.0), (6.0, 4, 2.0),
         (8.0, 0, 3.0), (12.0, 6, 2.0), (14.0, 4, 2.0)),
        note="Root and fifth alternating in eighths, then a b7 turn. The "
             "open fifth is the most stable interval there is, so this can "
             "run under anything.",
    ),
    BassFigure(
        "descending_walk",
        ((0.0, 7, 2.0), (2.0, 6, 2.0), (4.0, 4, 2.0), (6.0, 2, 2.0),
         (8.0, 0, 4.0), (12.0, 0, 2.0), (14.0, 6, 2.0)),
        note="Starts on the octave and walks down through b7, 5, b3 to land "
             "on the root at beat 3 - with the snare. A bassline with a "
             "destination.",
    ),
    BassFigure(
        "triplet_halves",
        ((0.0, 0, 3.0), (3.0, 0, 3.0), (6.0, 6, 2.0), (8.0, 4, 3.0),
         (11.0, 4, 3.0), (14.0, 2, 2.0)),
        note="3+3+2 in each half of the bar. The same tresillo cell twice, "
             "on different notes - root then fifth.",
    ),
    BassFigure(
        "minor_third_bounce",
        ((0.0, 0, 2.0), (2.0, 2, 2.0), (4.0, 0, 2.0), (7.0, 2, 1.0),
         (8.0, 0, 4.0), (12.0, 4, 2.0), (14.0, 2, 2.0)),
        note="Leans on the b3 - the note that makes it minor. Bouncy, "
             "almost a ska line slowed down.",
    ),
    BassFigure(
        "neighbour_slide",
        ((0.0, 0, 3.0), (3.0, 1, 1.0), (4.0, 0, 3.0), (8.0, 0, 2.0),
         (10.0, 6, 1.0), (11.0, 0, 3.0), (14.0, 4, 2.0)),
        note="Root with its upper neighbour as a grace note - in phrygian "
             "that is a semitone and it snarls; in minor it is a whole step "
             "and it rocks.",
    ),
    BassFigure(
        "syncopated_stabs",
        ((0.0, 0, 1.0), (3.0, 0, 1.0), (6.0, 0, 1.0), (8.0, 0, 1.0),
         (11.0, 4, 1.0), (13.0, 6, 1.0), (14.0, 0, 2.0)),
        note="All short notes, off the grid. Wants a plucky envelope. The "
             "gaps are the point - this is a bassline made of rests.",
    ),
    BassFigure(
        "reese_answer",
        ((0.0, 0, 6.0), (7.0, 6, 1.0), (8.0, 0, 4.0), (13.0, 2, 1.0),
         (14.0, 4, 2.0)),
        note="A long root on 1, then short answers. Half the bar is one "
             "note - let the oscillator detune do the work there.",
    ),
    BassFigure(
        "riddim_bounce",
        ((2.0, 0, 2.0), (6.0, 0, 2.0), (8.0, 0, 2.0), (10.0, 4, 2.0),
         (12.0, 0, 2.0), (14.0, 6, 2.0)),
        note="Starts on the & of 1, not the 1. Every note is on an upbeat "
             "until beat 3 - it sits inside the skank instead of under it.",
    ),
    # Two-bar figures. A question in the first bar and an answer in the
    # second is how bass players actually think - one bar is a cell, two
    # bars is a phrase.
    BassFigure(
        "question_answer",
        ((0.0, 0, 3.0), (4.0, 0, 2.0), (6.0, 6, 2.0), (8.0, 0, 4.0),
         (12.0, 4, 4.0),
         (16.0, 0, 3.0), (20.0, 7, 2.0), (22.0, 6, 2.0), (24.0, 4, 4.0),
         (28.0, 2, 2.0), (30.0, 0, 2.0)),
        bars=2,
        note="Bar one asks, bar two answers with the octave and walks back "
             "down. Two bars is a phrase; one bar is a loop.",
    ),
    BassFigure(
        "long_then_busy",
        ((0.0, 0, 8.0), (8.0, 0, 8.0),
         (16.0, 0, 2.0), (18.0, 7, 1.0), (19.0, 6, 1.0), (20.0, 4, 2.0),
         (22.0, 2, 2.0), (24.0, 0, 3.0), (27.0, 6, 1.0), (28.0, 4, 4.0)),
        bars=2,
        note="A whole bar of held root, then a bar of movement. The contrast "
             "is the hook - the held bar makes the busy one land.",
    ),
    BassFigure(
        "drop_and_climb",
        ((0.0, 7, 2.0), (2.0, 0, 6.0), (8.0, 0, 4.0), (12.0, 0, 4.0),
         (16.0, 0, 2.0), (18.0, 2, 2.0), (20.0, 4, 2.0), (22.0, 6, 2.0),
         (24.0, 7, 4.0), (28.0, 6, 2.0), (30.0, 4, 2.0)),
        bars=2,
        note="Octave drop onto a long root, then a scale climb back up to "
             "the octave over the second bar. Big arc.",
    ),
    BassFigure(
        "dub_walk",
        ((6.0, 0, 2.0), (8.0, 0, 2.0), (10.0, 2, 1.0), (11.0, 3, 1.0),
         (12.0, 4, 2.0), (15.0, 6, 1.0)),
        note="Enters late, walks up through the b3 and 4th to the 5th, then "
             "a b7 pickup over the barline. The busiest roots-dub option.",
    ),
    BassFigure(
        "octave_drop",
        ((0.0, 0, 2.0), (2.0, 7, 2.0), (8.0, 0, 4.0), (14.0, 4, 2.0)),
        note="Root then octave up then back down. Classic reggae bounce.",
    ),
    BassFigure(
        "sparse_stab",
        ((0.0, 0, 2.0), (10.0, 0, 4.0)),
        note="Almost nothing. For drops, and for when the delay is doing the "
             "work.",
    ),
    BassFigure(
        "rolling_dub",
        ((0.0, 0, 2.0), (3.0, 2, 1.0), (4.0, 0, 2.0), (8.0, 4, 2.0),
         (11.0, 6, 1.0), (12.0, 0, 3.0)),
        note="Busier walking figure for verses that need motion.",
    ),
)

BY_NAME = {f.name: f for f in FIGURES}


def distinct_degrees(fig: BassFigure) -> int:
    """How many different scale degrees a figure uses.

    A figure with one or two is a *hold* - `root_hold`, `sparse_stab`,
    `half_time_sub`. Those are useful as contrast, or under a busy
    arrangement, or when something else is supplying the movement. As the
    primary figure for a whole song they are what "the bass is boring"
    sounds like.
    """
    return len({degree % 7 for _step, degree, _length in fig.notes})


def is_melodic(fig: BassFigure) -> bool:
    return distinct_degrees(fig) >= 3


def figure(name: str) -> BassFigure:
    if name not in BY_NAME:
        raise ValueError(
            f"unknown bass figure {name!r}; known: {', '.join(sorted(BY_NAME))}"
        )
    return BY_NAME[name]


def from_figure(
    fig: BassFigure,
    root_degree: int = 0,
    velocity: int = 105,
    steps_per_bar: int = STEPS_PER_BAR,
) -> Motif:
    """Realise a figure as a Motif, re-pointed at `root_degree`.

    Degrees inside the figure are relative to the current chord root, so the
    same figure follows a progression without being rewritten.
    """
    tones = [
        Tone(
            degree=root_degree + deg,
            step=step,
            length=length,
            velocity=int(velocity * (0.85 + 0.15 * (metric_weight(int(step)) / 4.0))),
        )
        for step, deg, length in fig.notes
    ]
    return Motif(
        sorted(tones, key=lambda t: t.step),
        steps_per_bar * fig.bars,
        steps_per_bar,
        fig.name,
    )


def generate(
    rhythm: Pattern,
    root_degree: int = 0,
    rng: random.Random | None = None,
    leap_chance: float = 0.25,
    octave_chance: float = 0.12,
    velocity: int = 105,
) -> Motif:
    """Build a bassline onto an arbitrary rhythm.

    Strong beats get the root or fifth; weak beats may move to the b7 or 4th;
    occasionally the line jumps an octave. Deliberately conservative - a bass
    part that is too interesting stops being a foundation.
    """
    rng = rng or random.Random()
    tones: list[Tone] = []
    previous = root_degree

    for hit in rhythm.sorted_hits():
        weight = metric_weight(int(hit.step), rhythm.steps_per_bar)
        if weight >= 3.0:
            degree = root_degree
        elif weight >= 2.0:
            degree = root_degree + rng.choice((0, 0, 4))
        elif rng.random() < leap_chance:
            degree = root_degree + rng.choice(BASS_DEGREES)
        else:
            degree = root_degree + rng.choice(SAFE_DEGREES)

        if rng.random() < octave_chance:
            degree += 7 if previous <= root_degree else -7

        tones.append(Tone(degree, hit.step, hit.length, hit.velocity or velocity))
        previous = degree

    return Motif(tones, rhythm.length, rhythm.steps_per_bar, "bass")


def rhythm_for(
    style: str = "half_time",
    rng: random.Random | None = None,
    density: float = 0.4,
) -> Pattern:
    """A bass rhythm that leaves the right holes for a given drum style."""
    from . import rhythm as rhythm_mod

    rng = rng or random.Random()
    if style in ("one_drop", "rockers"):
        base = rhythm_mod.clave("rumba_3_2")
    elif style == "steppers":
        base = rhythm_mod.euclid(5, 16, rotation=0)
    else:  # half-time family
        base = rhythm_mod.from_string("x.....x.x......x")

    target = max(1, int(round(base.length * density)))
    if len(base.hits) > target:
        base = rhythm_mod.thin(base, target / len(base.hits), rng)
    return base


def constrain_register(
    motif: Motif, scale: Scale, low: int, high: int
) -> list[tuple[int, float, float, int]]:
    """Realise a bass motif into MIDI notes forced inside [low, high].

    Returns (midi, step, length, velocity). Octave-folds rather than clamps,
    so the line keeps its shape instead of flattening against the ceiling.
    """
    out = []
    for midi, step, length, vel in motif.notes(scale):
        out.append((pitch.transpose_into(midi, low, high), step, length, vel))
    return out


def monophonic(notes: list[tuple[int, float, float, int]]):
    """Truncate overlaps so no two notes sound at once.

    The Coral's bass voice is one oscillator. Overlapping notes on a mono
    voice produce whatever the synth's note-priority rule happens to be,
    which is not a musical decision we want to leave to chance.
    """
    ordered = sorted(notes, key=lambda n: n[1])
    out = []
    for i, (midi, step, length, vel) in enumerate(ordered):
        if i + 1 < len(ordered):
            next_step = ordered[i + 1][1]
            length = min(length, max(0.125, next_step - step))
        out.append((midi, step, length, vel))
    return out


def to_pattern(motif: Motif) -> Pattern:
    """View a bass motif as a rhythm pattern (for analysis or display)."""
    return Pattern(
        motif.length,
        [Hit(t.step, t.velocity, t.length) for t in motif.sorted_tones()],
        motif.steps_per_bar,
        motif.name,
    )


# --------------------------------------------------------------------------
# Generative: rhythm from a groove spec, pitches from a bass model, phrases
# from operations on a cell
# --------------------------------------------------------------------------

from . import groove as groove_mod
from .melody import CONTOURS, sample_contour


@dataclass(frozen=True)
class BassParams:
    """Every knob of the bass generator.

    These are musical quantities, not presets. Turning `approach` up adds
    passing and neighbour tones into the gaps before strong beats; turning
    `syncopation` up moves the rhythm off the grid; `anticipate` puts the
    push on the & of 4. A named figure from the old library is one point in
    this space, and most of the space has no name.
    """

    density: float = 0.3          # fraction of 16ths struck
    syncopation: float = 0.45
    anticipate: bool = False      # hit on the & of 4, tied over the barline
    anchor_root: bool = True      # always a note on the downbeat
    approach: float = 0.45        # chance a weak-beat note approaches the next strong one
    octave_jumps: float = 0.12    # chance a weak-beat note jumps an octave
    fifth_weight: float = 0.35    # how often a beat lands on the 5th instead of the root
    contour: str = "static"       # slow pitch drift across the bar
    feel: str = "straight"
    # Duration. A bass note sustains toward the next one: length is the gap
    # to the next hit times `gate`, capped at `max_hold` steps. Gate near 1
    # is legato sub bass; near 0.5 is a plucked, staccato line. The strong
    # beats hold a little longer than the weak ones regardless.
    gate: float = 0.8
    max_hold: float = 12.0
    # Phrasing: how many bars a phrase spans, and which bars answer rather
    # than restate. (3,) with phrase_bars=4 is state-state-state-answer.
    phrase_bars: int = 4
    answer_bars: tuple[int, ...] = (3,)
    fill_density_boost: float = 0.1

    def groove(self) -> groove_mod.GrooveSpec:
        return groove_mod.bass_spec(
            self.feel, self.density, self.syncopation,
            anticipate=self.anticipate, anchor_root=self.anchor_root,
        )


# Degrees a weak beat may wander to, and how welcome each is. b7 and 5 are
# the workhorses; b3 and 4 colour; the octave is a jump not a step.
_WEAK_DEGREES = ((6, 1.0), (4, 0.9), (2, 0.45), (3, 0.4), (0, 0.6))


def generate_line(
    rhythm: Pattern,
    root_degree: int,
    rng: random.Random,
    params: BassParams,
    next_root_degree: int | None = None,
) -> Motif:
    """Put bass pitches on a rhythm.

    The model, in order of what decides each note:

    1. **Downbeat and half-bar** (metric weight 3+): the root, or the fifth
       with probability `fifth_weight`. The bass's job is to say what the
       chord is, and it says it here.
    2. **Beats** (weight 2): root, fifth or b7 - the strong colours.
    3. **Weak positions**: with probability `approach`, a diatonic neighbour
       of whatever the *next* strong note will be, so the line leads into
       it; otherwise a colour tone pulled by the contour. Occasionally an
       octave displacement.
    4. **A hit on the & of 4** is an anticipation of the next chord, so it
       takes the next root, not this one.
    """
    hits = rhythm.sorted_hits()
    if not hits:
        return Motif([], rhythm.length, rhythm.steps_per_bar, "bass")

    # First pass: decide the strong-beat notes, because weak beats need to
    # know what they are approaching.
    strong: dict[int, int] = {}
    for h in hits:
        step, w = int(h.step), metric_weight(int(h.step), rhythm.steps_per_bar)
        if step == 0:
            # The downbeat says what the chord is. Always the root.
            strong[step] = 0
        elif w >= 3.0:
            strong[step] = 4 if rng.random() < params.fifth_weight * 0.6 else 0
        elif w >= 2.0:
            strong[step] = rng.choices([0, 4, 6], [1.0, params.fifth_weight, 0.3])[0]

    tones: list[Tone] = []
    previous = 0
    last_was_approach = False
    n = len(hits)
    for i, h in enumerate(hits):
        step = int(h.step)
        position = i / max(1, n - 1)
        drift = (sample_contour(params.contour, position) - 0.5) * 4  # -2..2 degrees

        if step >= rhythm.steps_per_bar - 2 and next_root_degree is not None \
                and params.anticipate:
            rel = next_root_degree - root_degree
        elif step in strong:
            rel = strong[step]
        else:
            # Find the next strong note this one could lead into.
            upcoming = [s for s in strong if s > step]
            target = strong[min(upcoming)] if upcoming else 0
            # An approach tone twice in a row is just a repeated note; the
            # second weak beat before a target walks instead.
            if rng.random() < params.approach and not last_was_approach:
                rel = target + rng.choice((-1, 1)) if target != 0 else rng.choice((-1, 1, 6))
                if rel == -1:
                    rel = 6 - 7  # the b7 below the root: leading up into it
                last_was_approach = True
            else:
                degs, ws = zip(*_WEAK_DEGREES)
                rel = rng.choices(degs, ws)[0]
                rel += int(round(drift * 0.5))
                if rel == previous:
                    rel = 4 if previous != 4 else 6
                last_was_approach = False
            if rng.random() < params.octave_jumps:
                rel += 7 if previous <= 0 else -7

        if step in strong:
            last_was_approach = False
        tones.append(Tone(root_degree + rel, h.step, h.length, h.velocity))
        previous = rel

    # Durations from spacing. Each note holds toward the next hit, so a
    # sparse line is long held notes and a dense one is short ones - the
    # way a bass player actually plays, rather than every note being a
    # stab whatever the gap after it.
    steps_per_bar = rhythm.steps_per_bar
    for i, tone in enumerate(tones):
        nxt = tones[i + 1].step if i + 1 < len(tones) else float(steps_per_bar)
        gap = nxt - tone.step
        if i + 1 == len(tones) and params.anticipate and tone.step >= steps_per_bar - 2:
            gap = 4.0                       # the push holds over the barline
        w = metric_weight(int(tone.step), steps_per_bar) / 4.0
        gate = min(1.0, params.gate + 0.12 * w)
        length = max(0.5, min(params.max_hold, gap * gate))
        tones[i] = Tone(tone.degree, tone.step, length, tone.velocity)

    # A two-note line must not be root-root: with only two notes a bar the
    # second one is the entire melodic content, so it takes the fifth or the
    # b7 - the two notes a sub can say after the root.
    if len(tones) == 2 and all(tn.degree == root_degree for tn in tones):
        second = tones[1]
        tones[1] = Tone(root_degree + rng.choice((4, 6, 6)), second.step, second.length, second.velocity)

    # Colour guarantee. A line of three or more notes that only ever says
    # root and fifth is harmonically correct and completely inert - and it
    # happens easily when every hit lands on a strong position, because
    # strong positions only choose from root, fifth and (rarely) b7. Make
    # the weakest-placed note a colour tone.
    if len(tones) >= 3:
        rels = {t.degree - root_degree for t in tones}
        if len({r % 7 for r in rels}) < 3:
            weakest = min(
                (t for t in tones if t.step != 0.0),
                key=lambda t: metric_weight(int(t.step), rhythm.steps_per_bar),
                default=None,
            )
            if weakest is not None:
                colour = rng.choice((6, 6, 2, 3))
                tones[tones.index(weakest)] = Tone(
                    root_degree + colour, weakest.step, weakest.length, weakest.velocity)

    return Motif(tones, rhythm.length, rhythm.steps_per_bar, "bass")


def generate_phrase(
    params: BassParams,
    rng: random.Random,
    root_degrees: list[int],
) -> list[Motif]:
    """A multi-bar bass phrase: a cell, restated, then answered.

    `root_degrees` gives the chord root for each bar of the phrase (so the
    line follows the harmony) and its length sets the phrase length. Bars
    listed in `params.answer_bars` get a fresh, denser rhythm and new pitches
    - the fill - rather than a restatement. Everything else repeats the cell
    re-pointed at that bar's chord, which is what keeps a phrase a phrase.
    """
    bars = len(root_degrees)
    cell_rhythm = groove_mod.generate(params.groove(), rng)
    fill_params = BassParams(
        **{**params.__dict__, "density": min(0.6, params.density + params.fill_density_boost),
           "syncopation": min(1.0, params.syncopation + 0.15),
           "approach": min(1.0, params.approach + 0.2)}
    )
    fill_rhythm = groove_mod.generate(fill_params.groove(), rng)
    # Freeze the cell's pitch choices so restatements are restatements.
    cell_seed = rng.random()

    out = []
    for i, root in enumerate(root_degrees):
        nxt = root_degrees[(i + 1) % bars]
        if i in params.answer_bars:
            out.append(generate_line(fill_rhythm, root, rng, fill_params, nxt))
        else:
            out.append(generate_line(cell_rhythm, root, random.Random(cell_seed), params, nxt))
    return out


def params_for_feel(feel: str, rng: random.Random, energy: float = 0.6) -> BassParams:
    """Draw bass parameters appropriate to a feel and an energy level.

    Ranges, not presets: every field is sampled from a musically sensible
    interval, so no two songs share a bassline unless the seed does.
    """
    if feel == "half_time":
        return BassParams(
            # Two to four notes a bar. The sub is long notes with air around
            # them; the movement is in where they land, not how many there are.
            density=rng.uniform(0.08, 0.18) + 0.04 * energy,
            syncopation=rng.uniform(0.4, 0.75),
            anticipate=rng.random() < 0.25,
            approach=rng.uniform(0.3, 0.7),
            octave_jumps=rng.uniform(0.05, 0.25),
            fifth_weight=rng.uniform(0.2, 0.5),
            contour=rng.choice(("static", "falling_tail", "arch", "descending")),
            feel="half_time",
            answer_bars=rng.choice(((3,), (3,), (1, 3), (2, 3))),
            gate=rng.uniform(0.8, 0.98),
            max_hold=rng.choice((12.0, 16.0, 16.0)),
        )
    return BassParams(
        density=rng.uniform(0.16, 0.3) + 0.06 * energy,
        syncopation=rng.uniform(0.35, 0.7),
        anticipate=rng.random() < 0.55,
        anchor_root=rng.random() < 0.6,    # one-drop basslines often skip the 1
        approach=rng.uniform(0.35, 0.75),
        octave_jumps=rng.uniform(0.08, 0.3),
        fifth_weight=rng.uniform(0.25, 0.55),
        contour=rng.choice(("static", "arch", "valley", "zigzag", "descending")),
        feel="straight",
        answer_bars=rng.choice(((3,), (3,), (1, 3))),
        gate=rng.uniform(0.55, 0.9),
        max_hold=rng.choice((6.0, 8.0, 12.0)),
    )
