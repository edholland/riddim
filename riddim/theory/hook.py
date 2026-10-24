"""Hooks: the part of a song people walk out humming.

A hook is not a random walk that happens to stay in key. It is a **motive** -
a rhythm of two to six notes with a pitch shape - said again and again,
moved, and answered. That is how every riff people remember is built: the
rhythm is learned first, the shape second, and the *return* of both is what
turns a line into a hook. So this generator starts from the motive:

1. **A rhythmic idiom.** `rhythm` picks a family - dotted 3-3-2, repeated
   stabs, pickups into the beat, long-then-short, all-offbeat, three
   against four, flowing eighths, syncopated - and `density` picks how full
   a figure of that family is. `syncopation` displaces the figure off the
   beat by a sixteenth or an eighth.
2. **A pitch shape.** `motion` decides between steps and leaps (`leap` is
   how far a leap goes), `repeat` how often a note is struck again - the
   percussive repeated-note riff - rather than moved.
3. **A trajectory.** The motive is stated several times across a unit,
   each statement moved along the `contour`, over a `span` of scale
   degrees. Same rhythm, new height: the sequence.
4. **A form.** Four units: sentence (a a′ b c - the climb in b is the
   motive moving up step by step to the one high point), period (a x a y),
   riff (a a a a′), call-and-answer (a — a′ —, the rests for the band).
   Units end open until the last one lands on the tonic. Every unit leaves
   its tail empty (`breath`) for the band's answer.

Candidates are generated from the dials and scored on two things: whether
the hook is sound (fits the chords, one climax, closes, its rhythm comes
back) and whether it is **what the dials asked for** (span, motion,
syncopation). Turning a dial changes the hook in that direction; the
selector does not fight it back to an average.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace

from .chord import Chord
from .harmony import sits_on as _sits_on
from .melody import Motif, Tone, sample_contour
from .rhythm import metric_weight
from .scale import Scale

FORMS = ("sentence", "period", "riff", "call_answer")

# What kind of lead this is - the gesture of a unit, which is what a
# listener hears first, before any rhythm or interval:
#   call    a quick figure, one long held note, space for the band's reply
#   riff    the figure back to back - an ostinato, short notes
#   song    a singable line across the unit: steps, mixed note lengths
#   stabs   one or two punchy hits per statement, wide leaps, lots of space
#   drone   two or three long notes a unit - a slow melody over the riddim
#   arp     chord tones rolling in steady eighths or sixteenths
#   chop    a garage vocal chop: a stuttered syllable, a step, a clipped
#           vowel - a sampler cutting up a phrase, a few notes wide
PHRASINGS = ("call", "riff", "song", "stabs", "drone", "arp", "chop")

# The tone that makes each mode itself, as a scale degree (0-indexed), and
# the degree it most naturally resolves to by step.
MODE_COLOUR = {
    "phrygian": (1, 0),           # b2 -> 1
    "phrygian_dominant": (1, 0),
    "dorian": (5, 4),             # natural 6 -> 5
    "harmonic_minor": (6, 7),     # leading tone -> octave
    "minor": (5, 4),              # b6 -> 5
    "aeolian": (5, 4),
    "melodic_minor": (6, 7),
    "major": (3, 2),
    "mixolydian": (6, 7),
    "lydian": (3, 4),
}

HOOK_CONTOURS = ("arch", "ascending", "valley", "falling_tail", "zigzag", "descending")

# The five notes each mode's tunes are actually built from, as 0-indexed
# degrees of the seven-note mode. Dub melodica lines, 140 leads and garage
# chops are overwhelmingly pentatonic - the minor pentatonic over the minor
# modes - with the mode's one colour tone (MODE_COLOUR) let in on top. A
# generator allowed every scale tone uses every scale tone, and a line that
# walks the whole mode is the plainest sign that nobody wrote it.
PALETTES = {
    "minor": (0, 2, 3, 4, 6), "aeolian": (0, 2, 3, 4, 6),
    "dorian": (0, 2, 3, 4, 6), "phrygian": (0, 2, 3, 4, 6),
    "harmonic_minor": (0, 2, 3, 4, 5), "melodic_minor": (0, 2, 3, 4, 5),
    "phrygian_dominant": (0, 1, 2, 4, 5),
    "major": (0, 1, 2, 4, 5), "ionian": (0, 1, 2, 4, 5),
    "lydian": (0, 1, 2, 4, 5), "mixolydian": (0, 1, 2, 4, 6),
}

# Rhythmic idioms: (motive length in steps, onsets), strictly sparse to
# dense - notes per step rise along each row, so `density` always means
# "more notes". A
# motive is stated every `length` steps through a unit, so its rhythm is
# the hook's pulse.
RHYTHMS: dict[str, tuple[tuple[int, tuple[int, ...]], ...]] = {
    "dotted": ((8, (0, 3)), (8, (0, 3, 6)), (8, (0, 3, 5, 6)), (8, (0, 2, 3, 5, 6))),
    "stabs": ((8, (0, 1)), (8, (0, 1, 2)), (8, (0, 1, 3, 4)), (4, (0, 1, 2))),
    "pickup": ((8, (0, 7)), (8, (0, 6, 7)), (8, (0, 5, 6, 7)), (8, (0, 2, 5, 6, 7))),
    "longshort": ((16, (0, 12, 14)), (16, (0, 10, 12, 14)), (8, (0, 5, 6)), (8, (0, 4, 6, 7))),
    "offbeat": ((8, (2,)), (8, (2, 6)), (8, (2, 5, 6)), (4, (2, 3))),
    "triplet": ((12, (0, 6)), (12, (0, 3, 6)), (12, (0, 3, 6, 9)), (12, (0, 2, 3, 6, 9))),
    "flowing": ((8, (0, 4)), (8, (0, 2, 4, 6)), (8, (0, 2, 3, 4, 6)), (8, (0, 1, 2, 3, 4, 6))),
    "syncopated": ((8, (0, 3)), (8, (0, 3, 5)), (8, (1, 3, 5, 6)), (8, (0, 1, 3, 5, 6))),
}
RHYTHM_NAMES = tuple(RHYTHMS)


@dataclass(frozen=True)
class HookParams:
    """Every knob of the hook generator - each one audible on its own.

    `unit_bars` is the length of one unit of the form; the whole hook is four
    units. `density` picks how full the rhythmic figure is; `breath` is the
    fraction of each unit left empty for the answer.
    """

    form: str = "sentence"
    phrasing: str = "call"       # the kind of lead: see PHRASINGS
    rhythm: str = "dotted"       # the rhythmic idiom of the motive
    unit_bars: int = 2
    density: float = 0.5         # sparse .. full figure
    syncopation: float = 0.4     # on the beat .. displaced off it
    motion: float = 0.45         # steps .. leaps
    repeat: float = 0.25         # how often a note is struck again rather than moved
    leap: int = 4                # how far a leap goes, in scale degrees
    climax: float = 0.65         # where the high point sits in the phrase, 0..1
    span: int = 8                # range of the phrase in scale degrees
    repetition: float = 0.75     # 1 = restatements are literal; 0 = their endings bend
    colour: float = 0.6          # how willingly the mode's colour tone is featured
    breath: float = 0.25         # fraction of each unit left for the answer
    contour: str = "arch"
    candidates: int = 32         # how many hooks are auditioned
    contrast: float = 0.7        # how often a later drop brings a B-theme before the hook
    outside: float = 0.15        # how willingly the tune leaves its pentatonic palette


@dataclass
class Unit:
    """One quarter of the hook phrase."""

    role: str                    # a, a2, b, c, x, y, rest
    tones: list[Tone]            # steps relative to the unit start
    steps: int

    @property
    def sung(self) -> bool:
        return bool(self.tones)


@dataclass
class Hook:
    """A hook phrase in diatonic space, plus its score."""

    units: list[Unit]
    params: HookParams
    unit_steps: int
    contour: str = "arch"
    score: float = 0.0
    report: dict = field(default_factory=dict)

    @property
    def length(self) -> int:
        return self.unit_steps * len(self.units)

    @property
    def cell(self) -> Unit:
        return self.units[0]

    def phrase(self) -> Motif:
        """The whole phrase as one motif, steps from the phrase start."""
        tones = []
        for i, u in enumerate(self.units):
            off = i * self.unit_steps
            tones += [replace(t, step=t.step + off) for t in u.tones]
        return Motif(tones, self.length, 16, "hook")

    def form_string(self) -> str:
        return " ".join(u.role for u in self.units)

    def rhythm_string(self, unit: int = 0) -> str:
        u = self.units[unit]
        grid = ["."] * u.steps
        for t in u.tones:
            s = int(t.step)
            if 0 <= s < u.steps:
                grid[s] = "x"
        return "".join(grid)

    def onset_steps(self) -> list[int]:
        """Absolute onset steps of every note in the phrase - the activity
        the rest of the band listens to."""
        out = []
        for i, u in enumerate(self.units):
            out += [int(t.step) + i * self.unit_steps for t in u.tones]
        return sorted(out)


# --------------------------------------------------------------------------
# The motive
# --------------------------------------------------------------------------

def motive_rhythm(params: HookParams, rng: random.Random) -> tuple[int, list[int]]:
    """(length, onsets) of the motive: the idiom's figure at the asked-for
    density, as written - displacement comes later, once the figure has its
    final size."""
    variants = RHYTHMS.get(params.rhythm, RHYTHMS["dotted"])
    pos = params.density * (len(variants) - 1) + rng.choice((-0.35, 0.0, 0.35))
    length, onsets = variants[max(0, min(len(variants) - 1, int(round(pos))))]
    return length, list(onsets)


def displace(onsets: list[int], length: int, syncopation: float) -> list[int]:
    """Sit the figure on the grid as the dial asks. Of four displacements -
    on the beat, a sixteenth late, an eighth late, a sixteenth early - the one
    whose share of off-beat notes is nearest the target. (A figure that is
    syncopated already is not shoved back onto the beat by asking for more.)
    Asked for straight, the figure is pulled onto the eighth-note grid."""
    target = 0.15 + 0.8 * syncopation

    def placed(shift):
        return sorted({(o + shift) % length for o in onsets})

    def off_share(shift):
        moved = placed(shift)
        return sum(1 for o in moved if o % 4) / len(moved)
    shift = min(range(4), key=lambda sh: (abs(off_share(sh) - target), sh))
    out = placed(shift)
    if syncopation < 0.3:
        out = sorted({o - (o % 2) for o in out})
    return out


def motive_shape(k: int, params: HookParams, rng: random.Random) -> list[int]:
    """Relative scale degrees for the motive's notes.

    Each move is a leap with probability ~`motion` (an octave, now and then,
    when motion is high - the 140 lead's favourite gesture), a third some of
    the time, otherwise a step. After a leap the line turns back: the gap is
    filled from the other side.
    """
    shape = [0]
    direction = 1 if rng.random() < 0.65 else -1
    for _ in range(1, k):
        if rng.random() < params.repeat * 0.85:
            shape.append(shape[-1])
            continue
        r = rng.random()
        if r < params.motion * 0.8:
            move = 7 if rng.random() < params.motion * 0.25 else max(2, params.leap)
        elif r < params.motion * 0.8 + 0.3 * (1 - params.motion) + 0.1:
            move = 2
        else:
            move = 1
        shape.append(shape[-1] + direction * move)
        if move >= 3 or rng.random() < 0.35:
            direction = -direction
    # A motive that never moves is a drum, not a hook: the last note of a
    # repeated-note figure goes somewhere, by leap when motion asks for it.
    if k >= 2 and len(set(shape)) == 1 and params.repeat < 0.85:
        shape[-1] += (max(2, params.leap) if rng.random() < params.motion else 1) * \
            (1 if rng.random() < 0.6 else -1)
    return shape


def _trajectory(n: int, params: HookParams, shape: list[int], contour: str) -> list[int]:
    """Where each statement of the motive sits, following the contour over
    whatever span the motive's own shape leaves."""
    if n <= 1:
        return [0]
    # How far apart the statements sit: a stepwise hook moves its motive by
    # small amounts, a leaping one throws it across the span. (The climb in
    # a sentence still reaches the full span either way.)
    room = max(1, params.span - (max(shape) - min(shape))) * (0.35 + 0.65 * params.motion)
    first = sample_contour(contour, 0.0)
    return [int(round(room * (sample_contour(contour, p / (n - 1)) - first)))
            for p in range(n)]


def _vel(onset_index: int, step: float, pos: float, contour: str, rng) -> int:
    lead = 14 if onset_index == 0 else 0      # the motive's first note leads
    w = metric_weight(int(step) % 16) / 4.0
    v = 76 + lead + 14 * w + 12 * sample_contour(contour, pos) + rng.uniform(-4, 4)
    return max(50, min(127, int(v)))


def _statements(length, onsets, shape, trajectory, start, sung, contour, rng,
                lengths_to=None) -> list[Tone]:
    """The motive stated at each trajectory point, every `length` steps,
    inside the sung part of a unit."""
    tones = []
    n = len(trajectory)
    for p, t in enumerate(trajectory):
        for i, (o, d) in enumerate(zip(onsets, shape)):
            step = p * length + o
            if step >= sung:
                continue
            tones.append(Tone(start + t + d, float(step), 1.0,
                              _vel(i, step, p / max(1, n - 1), contour, rng)))
    return _lengths(tones, sung)


def _lengths(tones: list[Tone], sung: float) -> list[Tone]:
    """Each note holds to the next; the last rings to the end of the sung
    part. (The lead's `gate` shortens them when it is played.)"""
    tones = sorted(tones, key=lambda t: t.step)
    out = []
    for i, t in enumerate(tones):
        end = tones[i + 1].step if i + 1 < len(tones) else max(t.step + 1.0, sung)
        out.append(replace(t, length=max(0.5, end - t.step)))
    return out


def _land(tones: list[Tone], target: int, size: int) -> list[Tone]:
    """End on `target` (nearest octave), stepping into it."""
    if not tones:
        return tones
    out = list(tones)
    last = out[-1].degree
    goal = target + round((last - target) / size) * size
    out[-1] = replace(out[-1], degree=goal)
    if len(out) >= 2 and abs(out[-2].degree - goal) > 2:
        out[-2] = replace(out[-2], degree=goal + (1 if out[-2].degree > goal else -1))
    return out


def _colour(tones: list[Tone], scale: Scale, amount: float, rng) -> list[Tone]:
    """Put the mode's colour tone in once, on a weak beat, by step."""
    if scale.name not in MODE_COLOUR or rng.random() > amount or len(tones) < 3:
        return tones
    colour, _ = MODE_COLOUR[scale.name]
    size = scale.size
    out = list(tones)
    order = list(range(1, len(out) - 1))
    rng.shuffle(order)
    for i in order:
        if metric_weight(int(out[i].step) % 16) >= 2.0:
            continue
        octave = (out[i].degree // size) * size
        for c in (octave + colour, octave + colour + size, octave + colour - size):
            if abs(c - out[i - 1].degree) <= 2 and abs(c - out[i + 1].degree) <= 2:
                out[i] = replace(out[i], degree=c)
                return out
    return out


def palette(scale: Scale) -> set[int] | None:
    """The degrees (mod the scale's size) a hook in this mode is built
    from: its pentatonic. None for a scale that is already that small.
    (The mode's colour tone is not in it: `_colour` places that once, where
    it means something, and those placements are kept.)"""
    if scale.name not in PALETTES or scale.size != 7:
        return None
    return set(PALETTES[scale.name])


def _to_palette(tones: list[Tone], scale: Scale, outside: float, rng,
                keep: set | frozenset = frozenset()) -> list[Tone]:
    """Bring each tone outside the palette onto its nearest palette note -
    continuing the line's direction, so a passing step becomes a step of
    the pentatonic (a third) rather than a repeated note. `outside` is the
    chance a stray tone is kept: the passing note a player lets through."""
    pal = palette(scale)
    if pal is None:
        return tones
    size = scale.size
    colour = MODE_COLOUR[scale.name][0] if scale.name in MODE_COLOUR else None
    out: list[Tone] = []
    prev = None
    for t in sorted(tones, key=lambda t: t.step):
        # `keep` holds the steps where `_colour` put the mode's colour tone
        # on purpose: that one stays, wherever the motive states it.
        if t.alter or t.degree % size in pal or t.degree % size == colour \
                and t.step in keep or rng.random() < outside:
            out.append(t)
            prev = t.degree
            continue
        up = prev is None or t.degree >= prev
        for off in ((1, -1, 2, -2) if up else (-1, 1, -2, 2)):
            if (t.degree + off) % size in pal:
                t = replace(t, degree=t.degree + off)
                break
        out.append(t)
        prev = t.degree
    return out


def generate(
    params: HookParams,
    scale: Scale,
    rng: random.Random,
    unit_steps: int | None = None,
    contour: str | None = None,
) -> Hook:
    """One hook candidate - not yet scored."""
    steps = unit_steps or params.unit_bars * 16
    contour = contour or params.contour
    size = scale.size
    sung = max(4.0, round(steps * (1.0 - params.breath)))

    # One stream per decision, all from this candidate's seed: turning a dial
    # changes the decision it governs and leaves the others where they were,
    # so the same idea comes back adjusted rather than replaced.
    seed = rng.random()
    sub = {k: random.Random(f"{seed}:{k}") for k in
           ("rhythm", "shape", "start", "colour", "vel", "land", "form", "palette")}
    length, onsets = motive_rhythm(params, sub["rhythm"])
    shape = motive_shape(len(onsets), params, sub["shape"])
    # Re-strikes: with `repeat` up, a note with room after it is struck again
    # a sixteenth later - the stab riff's stutter.
    restrike = sub["shape"]
    o2, s2 = [], []
    for i, (o, d) in enumerate(zip(onsets, shape)):
        o2.append(o)
        s2.append(d)
        nxt = onsets[i + 1] if i + 1 < len(onsets) else length
        if nxt - o >= 3 and restrike.random() < params.repeat * 0.8:
            o2.append(o + 1)
            s2.append(d)
    onsets, shape = o2, s2

    ph = params.phrasing if params.phrasing in PHRASINGS else "call"
    prng = sub["shape"]
    cap = None                      # a ceiling on note length, for stabs
    if ph != "call":
        onsets, shape, length, n, cap = _phrase(ph, params, onsets, shape, length,
                                                sung, contour, prng)
        if ph not in ("riff", "call", "chop"):
            # Re-strikes for every kind of lead: `repeat` strikes a note
            # again a sixteenth later where there is room.
            o2, s2 = [], []
            for i, (o, d) in enumerate(zip(onsets, shape)):
                o2.append(o)
                s2.append(d)
                nxt = onsets[i + 1] if i + 1 < len(onsets) else length
                if nxt - o >= 2 and prng.random() < params.repeat * 0.8:
                    o2.append(o + 1)
                    s2.append(d)
            onsets, shape = o2, s2
        if ph == "chop":
            # A chop keeps its syllables together - the stutter is the sound -
            # so syncopation moves the whole cut off the beat, not its parts.
            onsets = [o + (2 if params.syncopation > 0.55 else 0) for o in onsets]
        else:
            moved = displace(onsets, max(length, max(onsets) + 1), params.syncopation)
            if len(moved) == len(onsets):
                onsets = moved
        if ph == "stabs":
            # Home, leap, home, leap: the stab lead's whole personality.
            jump = 7 if prng.random() < params.motion else max(2, params.leap)
            trajectory = [0 if i % 2 == 0 else jump for i in range(n)]
        elif ph in ("riff", "chop"):
            trajectory = _trajectory(n, params, shape, contour)
        else:
            trajectory = [0]
        start = sub["start"].choice((0, 2, 4, 4, 7)) - min(0, min(shape) + min(trajectory))
        vrng = sub["vel"]
        return _assemble(params, scale, steps, sung, length, onsets, shape, trajectory,
                         start, contour, sub, vrng, cap)

    # Phrase rhythm. A unit is a *call*: one to three statements of the
    # motive, each a quick figure that lands on a held note - the note a
    # listener sings - and then the answer space (`breath`) where the band
    # replies. Back-to-back figures with no landing are a stream of blips.
    # One statement - figure, then a long sung note - unless density asks for
    # a second (or, in a long unit, a third).
    n = 1
    if params.density > 0.55 and sung >= 16:
        n = 2
    if params.density > 0.85 and sung >= 24:
        n = 3
    # Statements start on beats, so a figure sits the same way every time.
    length = max(4, int(sung // n) // 4 * 4)
    room = max(2, int(length * 0.55))      # the figure is quick; the landing sings
    if max(onsets) > room:
        squeeze = room / max(onsets)
        pairs = sorted({int(round(o * squeeze)): d for o, d in zip(onsets, shape)}.items())
        onsets, shape = [o for o, _ in pairs], [d for _, d in pairs]
    # Span is how far a call reaches: with one or two statements there is
    # little room to travel, so the landing note - the last of the figure,
    # the one that is held - is pulled out to the reach the dial asks for.
    reach = max(2, int(round(params.span * 0.6 * (0.35 + 0.65 * params.motion))))
    if n <= 2 and len(shape) >= 2 and abs(shape[-1] - shape[0]) < reach:
        sign = 1 if shape[-1] >= shape[0] else -1
        shape[-1] = shape[0] + sign * reach
    # Now the figure has its size, sit it on the grid as syncopation asks.
    # (The shape follows its notes in order; displacement only moves time.)
    moved = displace(onsets, max(room + 1, max(onsets) + 1), params.syncopation)
    if len(moved) == len(onsets):
        onsets = moved
    trajectory = _trajectory(n, params, shape, contour)
    start = sub["start"].choice((0, 2, 4, 4, 7)) - min(0, min(shape) + min(trajectory))
    vrng = sub["vel"]
    return _assemble(params, scale, steps, sung, length, onsets, shape, trajectory,
                     start, contour, sub, vrng, None)


def _phrase(ph, params, onsets, shape, length, sung, contour, rng):
    """Onsets, shape, statement length, statement count and a length cap for
    each kind of lead other than the call."""
    lp = max(2, params.leap)
    if ph == "riff":
        # The figure back to back: an ostinato - at least a mid-density figure,
        # or it is not a riff but a pair of notes.
        if len(onsets) < 3:
            variants = RHYTHMS.get(params.rhythm, RHYTHMS["dotted"])
            length, onsets = variants[min(len(variants) - 1, 2)]
            onsets = list(onsets)
            shape = motive_shape(len(onsets), params, rng)
        n = max(1, int(sung // length))
        return onsets, shape, length, n, None
    if ph == "stabs":
        # One or two hits a statement; leaps (octaves when motion is high).
        k = 1 if params.density < 0.5 else 2
        hits = [0, 3 if params.syncopation > 0.5 else 2][:k]
        jump = 7 if rng.random() < params.motion else lp
        shp = [0, jump * rng.choice((1, -1))][:k]
        n = 2 if params.density < 0.7 else 3
        length = max(4, int(sung // n) // 4 * 4)
        return hits, shp, length, n, 1.5
    if ph == "chop":
        # A syllable cell - the stutter (one note struck twice), a step or
        # two away, and a held vowel - stated every half bar or bar with air
        # around it. Narrow: a chopped vocal moves a few notes, not an
        # octave, unless motion asks for the leap.
        cells = ((0, 1, 4), (0, 1, 3), (0, 1, 2, 6), (0, 1, 3, 6), (0, 1, 3, 4))
        ons = list(cells[min(len(cells) - 1, int(params.density * len(cells)))])
        shp, d, up = [0], 0, rng.random() < 0.5
        # The stutter is the repeat dial: none (the syllable steps), the
        # double hit, or - turned right up - a triple.
        stutters = 0 if params.repeat < 0.15 else (2 if params.repeat > 0.55 else 1)
        for i in range(1, len(ons)):
            if i <= stutters and ons[i] - ons[i - 1] <= 2:
                shp.append(d)                   # the stutter
                continue
            # span widens the cut: a narrow chop moves by step, a wide one
            # by thirds and fourths
            steps = (1, 1, 2) if params.span < 7 else ((1, 2, 2) if params.span < 10 else (2, 3, 3))
            move = lp if rng.random() < params.motion * 0.5 else rng.choice(steps)
            d += move if up else -move
            up = not up if abs(d) >= 3 else up
            shp.append(d)
        # The chop and then the air: one a bar, two only when density asks.
        length = 8 if params.density >= 0.75 else 16
        # Every bar (every half bar when dense): a chop's breath is the air
        # between cuts, so a statement that starts inside the sung part is
        # made even if its tail is trimmed.
        return ons, shp, length, max(1, int(math.ceil(sung / length))), 2.5
    if ph == "drone":
        # Two or three long notes across the unit, moving by step.
        count = 2 if params.density < 0.6 else 3
        gap = max(4, int(sung // count) // 4 * 4)
        ons = [i * gap for i in range(count)]
        shp = [0]
        for _ in range(count - 1):
            move = lp if rng.random() < params.motion else rng.choice((1, 2))
            shp.append(shp[-1] + move * rng.choice((1, -1)))
        return ons, shp, int(sung), 1, None
    if ph == "arp":
        # Chord tones rolling: up, down or up-and-back, in eighths (or
        # sixteenths when dense).
        every = 1 if params.density > 0.7 else 2
        ons = list(range(0, int(sung), every))
        # How wide the arpeggio reaches: motion and span together.
        reach = (params.motion + min(1.0, max(0.0, (params.span - 4) / 8.0))) / 2.0
        if reach < 0.35:              # a scale run rather than an arpeggio
            cycle = rng.choice(((0, 1, 2, 3), (3, 2, 1, 0), (0, 1, 2, 1), (0, 2, 1, 3)))
        elif reach < 0.7:             # the triad
            cycle = rng.choice(((0, 2, 4, 7), (7, 4, 2, 0), (0, 2, 4, 7, 4, 2), (0, 4, 7, 4)))
        else:                         # wide: octaves and the seventh
            cycle = rng.choice(((0, 4, 7, 11), (0, 7, 4, 9), (0, 7, 14, 7), (11, 7, 4, 0)))
        shp = [cycle[i % len(cycle)] for i in range(len(ons))]
        return ons, shp, int(sung), 1, None
    # song: a singable line - mixed note values filling the unit, mostly
    # steps, the contour carried by the line itself.
    values = (2, 2, 4, 4, 3, 6, 8) if params.density < 0.5 else (2, 2, 2, 4, 1, 3)
    ons, pos = [], 0
    while pos < sung - 1:
        ons.append(pos)
        pos += rng.choice(values)
    shp, d = [], 0
    first = sample_contour(contour, 0.0)
    for i in range(len(ons)):
        target = (sample_contour(contour, i / max(1, len(ons) - 1)) - first) * params.span * 0.8
        if i:
            r = rng.random()
            if r < params.repeat * 0.5:
                pass
            elif r < 0.75 + 0.2 * (1 - params.motion):
                d += 1 if target > d else -1
            else:
                d += (2 if target > d else -2) if rng.random() > params.motion else (lp if target > d else -lp)
        shp.append(d)
    return ons, shp, int(sung), 1, None


def _assemble(params, scale, steps, sung, length, onsets, shape, trajectory, start,
              contour, sub, vrng, cap) -> Hook:
    """Build the four units of the form from a unit's material."""
    size = scale.size
    n = len(trajectory)

    def unit(role, tones):
        return Unit(role, tones, steps)

    a_tones = _statements(length, onsets, shape, trajectory, start, sung, contour, vrng)
    if cap is not None:
        a_tones = [replace(t, length=min(t.length, cap)) for t in a_tones]
    plain = a_tones
    a_tones = _colour(a_tones, scale, params.colour, sub["colour"])
    coloured = frozenset(t.step for t, o in zip(a_tones, plain) if t.degree != o.degree)
    # The opening unit ends open - on the 2nd, 4th or 5th - asking a question.
    a_open = _land(a_tones, sub["land"].choice((1, 3, 4, 4)), size)

    def restated(role):
        """The same rhythm and shape. Literal by `repetition`; otherwise the
        ending bends to a different open degree."""
        if sub["land"].random() <= params.repetition:
            return unit(role, list(a_open))
        return unit(role, _land(a_tones, sub["land"].choice((2, 4, 6)), size))

    def climb():
        """The motive moving up a step (or a third) per statement, twice as
        often - the sequence that carries the phrase to its one high point."""
        spacing = max(max(onsets) + 3, int(sung // 2) // 2 * 2) if n <= 2 else length
        count = max(2, int(sung // spacing))
        top = start + max(trajectory)
        # The climb reaches the top of the span: a wide hook climbs in
        # bigger steps, a narrow one creeps.
        reach = max(1, params.span - (top - start) - (max(shape) - min(shape)))
        rise = max(1, min(3, round(reach / count)))
        tones = []
        for p in range(count):
            for i, (o, d) in enumerate(zip(onsets, shape)):
                step = p * spacing + o
                if step < sung:
                    tones.append(Tone(top + p * rise + d - rise, float(step), 1.0,
                                      _vel(i, step, 0.5 + 0.5 * p / count, "ascending", vrng)))
        return unit("b", _lengths(tones, sung))

    def cadence(role, target):
        """The first statement, then a stepwise fall onto `target`, held on a
        strong beat - where the phrase closes."""
        head = [t for t in a_tones if t.step < length] or a_tones[:2]
        last = head[-1]
        goal = target + ((last.degree - target) // size) * size
        walk = []
        d = last.degree
        while abs(d - goal) > 1 and len(walk) < 3:
            d += -1 if d > goal else 1
            walk.append(d)
        landing = float(((int(last.step) // 4) + 2 + len(walk)) * 4)
        landing = min(landing, sung - 2) if sung - 2 > last.step + len(walk) + 1 else last.step + len(walk) + 1
        gap = (landing - last.step) / (len(walk) + 1)
        tones = list(head)
        for i, deg in enumerate(walk):
            tones.append(Tone(deg, float(round(last.step + gap * (i + 1))), 1.0, 84))
        tones.append(Tone(goal, landing, 4.0, 96))
        return unit(role, _lengths(tones, sung))

    def answer(target):
        """The first statement again, then the motive turned upside down -
        related, but a reply rather than a repeat."""
        head = [t for t in a_tones if t.step < length]
        inv = [-d for d in shape]
        tail = _statements(length, onsets, inv, [0] * max(1, n - 1),
                           (head[-1].degree if head else start), sung - length,
                           contour, vrng)
        tones = head + [replace(t, step=t.step + length) for t in tail]
        tones = [t for t in tones if t.step < sung]
        return unit("x" if target != 0 else "y", _land(_lengths(tones, sung), target, size))

    form = params.form if params.form in FORMS else "sentence"
    a = unit("a", a_open)
    if form == "sentence":
        units = [a, restated("a2"), climb(), cadence("c", 0)]
    elif form == "period":
        units = [a, answer(sub["form"].choice((4, 1, 3))), restated("a2"), answer(0)]
    elif form == "riff":
        units = [a, restated("a"), restated("a"), unit("a2", _land(a_tones, 0, size))]
    else:   # call_answer: the second call closes; the rests are the band's.
        units = [a, Unit("rest", [], steps), unit("a2", _land(a_tones, 0, size)),
                 Unit("rest", [], steps)]
    if params.phrasing != "arp":       # an arpeggio is its chord's tones already
        prng = sub["palette"]
        units = [replace(u, tones=_to_palette(u.tones, scale, params.outside, prng, coloured))
                 for u in units]
    return Hook(units, params, steps, contour)


# --------------------------------------------------------------------------
# Realisation against harmony, and the score
# --------------------------------------------------------------------------

ChordAt = "callable[[float], Chord | None]"   # step in phrase -> chord


def realise(hook: Hook, scale: Scale, chord_at, follow: bool = True) -> list[Tone]:
    """The phrase fitted to the chords it will sound over.

    Each unit may be shifted diatonically to follow the chord under it (a
    sequence) - but only if that agrees with the chord better than the
    literal statement does, so repetition survives wherever the harmony
    allows it. Then any strong-beat tone a semitone from a chord tone - the
    clash that sounds wrong rather than tense - is moved onto the chord.
    """
    out: list[Tone] = []
    base = None
    for i, u in enumerate(hook.units):
        if not u.tones:
            continue
        off = i * hook.unit_steps
        chord = chord_at(off)
        root = chord.degree if chord is not None else 0
        if base is None:
            base = root
        candidates = [0]
        if follow and chord is not None and root != base:
            shift = root - base
            while shift > 3:
                shift -= scale.size
            while shift < -3:
                shift += scale.size
            candidates.append(shift)
        best = None
        for shift in candidates:
            tones = [replace(t, degree=t.degree + shift, step=t.step + off) for t in u.tones]
            agree = _agreement(tones, scale, chord_at)
            # A literal restatement wins ties: the ear wants the same notes.
            key = agree - (0.08 if shift else 0.0)
            if best is None or key > best[0]:
                best = (key, tones)
        out += best[1]
    return [_avoid_clash(t, scale, chord_at(t.step)) for t in out]


def _agreement(tones, scale, chord_at) -> float:
    strong = [t for t in tones if metric_weight(int(t.step) % 16) >= 2.0]
    if not strong:
        return 1.0
    ok = 0
    for t in strong:
        c = chord_at(t.step)
        if c is None or _sits_on(t.midi(scale), c):
            ok += 1
    return ok / len(strong)


def _avoid_clash(t: Tone, scale: Scale, chord: Chord | None) -> Tone:
    """Move a tone off a semitone clash with the chord under it.

    Only where the clash is *heard*: on a strong beat, or on a note long
    enough to sit against the chord (two steps or more). A short passing
    tone a semitone off is a passing tone. When the chord has borrowed a
    tone from outside the scale - the leading tone of a major V - the line
    borrows it too, by alteration, rather than fleeing to a different note.
    """
    if chord is None:
        return t
    if metric_weight(int(t.step) % 16) < 2.0 and t.length < 2.0:
        return t
    pc = t.midi(scale) % 12
    chord_pcs = chord.pitch_classes()
    if pc in chord_pcs:
        return t
    near = [c for c in chord_pcs if (pc - c) % 12 in (1, 11)]
    if not near:
        # A long note on the downbeat or the half-bar is where the ear
        # decides what the harmony is; a whole step off there is a wrong
        # note held, not a tension. Anywhere else it is colour, and stays.
        if metric_weight(int(t.step) % 16) >= 3.0 and t.length >= 3.0:
            for off in (-1, 1, -2, 2):
                if scale.note(t.degree + off) % 12 in chord_pcs:
                    return replace(t, degree=t.degree + off, alter=0)
        return t
    in_scale = set(scale.pitch_classes)
    for c in near:
        if c not in in_scale:
            step = 1 if (c - pc) % 12 == 1 else -1
            return replace(t, alter=t.alter + step)
    for off in (-1, 1, -2, 2):
        if scale.note(t.degree + off) % 12 in chord_pcs:
            return replace(t, degree=t.degree + off, alter=0)
    return t




def vocabulary(params: HookParams) -> int:
    """How many different pitches a hook of this kind is built from: three
    for a stab, a drone or a chop, four for a call or a riff, five for a
    sung line or an arpeggio - and one more for each two degrees of span
    past seven, and up to two more as `outside` opens the tune to the whole
    mode, so neither dial is auditioned away by this one."""
    base = {"stabs": 3, "drone": 3, "chop": 3, "song": 5, "arp": 5}.get(params.phrasing, 4)
    return base + max(0, params.span - 7) // 2 + int(round(2 * params.outside))


def score(hook: Hook, tones: list[Tone], scale: Scale, chord_at) -> tuple[float, dict]:
    """How sound the realised hook is, and how faithfully it is what its
    dials asked for, 0..1, with reasons."""
    if len(tones) < 4:
        return 0.0, {"too_short": len(tones)}
    p = hook.params
    pitches = [t.midi(scale) for t in tones]
    rep: dict[str, float] = {}

    # Sound.
    rep["chord_fit"] = _agreement(tones, scale, chord_at)
    top = max(pitches)
    tops = [i for i, x in enumerate(pitches) if x == top]
    pos = tops[0] / max(1, len(pitches) - 1)
    unique = 1.0 if len(tops) == 1 else (0.7 if len(tops) == 2 else 0.35)
    rep["climax"] = unique * (1.0 - min(1.0, abs(pos - p.climax) * 1.4))
    if p.phrasing in ("riff", "arp", "stabs", "chop"):
        # A repeating figure repeats its top note by nature: judge it on
        # everything else.
        rep["climax"] = max(rep["climax"], 0.8)
    last = tones[-1].degree % scale.size
    rep["closes"] = 1.0 if last == 0 else (0.6 if last == 4 else 0.2)
    if p.form == "call_answer":
        rep["closes"] = max(rep["closes"], 0.6)
    a = [t.step for t in hook.units[0].tones]
    rep["returns"] = 1.0 if any([t.step for t in u.tones] == a for u in hook.units[1:]) else 0.4
    moves = [abs(b - a) for a, b in zip(pitches, pitches[1:])]
    rep["sane"] = 0.0 if any(m == 6 or m > 12 for m in moves) else 1.0
    # Economy: a hook people hum is built from a handful of pitches, said
    # again - not the whole scale walked through. Three notes make a stab or
    # a chop, four a riff or a call, five a sung line; a wider span earns a
    # note more. Fewer than three is a drum, not a tune. (The old measure
    # rewarded four *or more*, and the selector duly reached for every note
    # the mode allowed - the clearest sign that nobody wrote it.)
    vocab = len({x % 12 for x in pitches})
    rep["economy"] = (0.6 if vocab < 3 else
                      max(0.0, 1.0 - 0.35 * max(0, vocab - vocabulary(p))))
    # The opening unit is what is remembered: it has to move. Three
    # different notes for most leads; two for stabs and drones, which are
    # built from fewer.
    cell = {t.degree for t in hook.units[0].tones}
    need = 2 if p.phrasing in ("stabs", "drone", "chop") else 3
    rep["cell_moves"] = 1.0 if len(cell) >= need else (0.4 if len(cell) == need - 1 else 0.0)

    # Fidelity: is this what the dials asked for?
    degrees = [t.degree for t in tones]
    span = max(degrees) - min(degrees)
    rep["f_span"] = max(0.0, 1.0 - abs(span - p.span) / max(3.0, p.span))
    dmoves = [abs(b - a) for a, b in zip(degrees, degrees[1:])]
    moving = [m for m in dmoves if m > 0] or [1]
    motion = min(1.0, max(0.0, (sum(moving) / len(moving) - 1.0) / 2.0))
    rep["f_motion"] = 1.0 - abs(motion - p.motion)
    off = sum(1 for t in tones if metric_weight(int(t.step) % 16) < 2.0) / len(tones)
    rep["f_sync"] = max(0.0, 1.0 - abs(off - (0.15 + 0.8 * p.syncopation)) * 1.5)
    # Repeated notes as asked: the stutter dial is judged like the others,
    # so no other term can quietly audition it away.
    same = sum(1 for a, b in zip(degrees, degrees[1:]) if a == b) / max(1, len(degrees) - 1)
    target = 0.05 + 0.45 * p.repeat + (0.2 if p.phrasing == "chop" else 0.0)   # a chop is its stutter
    rep["f_repeat"] = max(0.0, 1.0 - abs(same - target) * 1.5)

    # The tune over the bass: on the strong beats a melody sings the third,
    # fifth or seventh of the chord - the root is the bass's note, and a
    # tune that keeps landing on it doubles the bass in octaves and sounds
    # hollow. The root is for cadences: the final note is exempt.
    strong = [t for t in tones[:-1] if metric_weight(int(t.step) % 16) >= 2.0]
    on_root = 0
    for t in strong:
        c = chord_at(t.step)
        if c is not None and t.midi(scale) % 12 == c.root_pc:
            on_root += 1
    share = on_root / len(strong) if strong else 0.0
    rep["over_bass"] = 1.0 if share <= 0.34 else max(0.0, 1.0 - (share - 0.34) * 2.5)

    weights = {"chord_fit": 2.0, "climax": 1.2, "closes": 0.8, "returns": 0.8,
               "sane": 1.0, "economy": 1.4, "cell_moves": 1.5, "over_bass": 1.2,
               "f_span": 1.6, "f_motion": 1.0, "f_sync": 1.0, "f_repeat": 0.8}
    total = sum(weights[k] * max(0.0, rep[k]) for k in weights) / sum(weights.values())
    return round(total, 4), rep


def best(
    params: HookParams,
    scale: Scale,
    rng: random.Random,
    chord_at,
    unit_steps: int | None = None,
) -> Hook:
    """Audition `params.candidates` hooks against the harmony; keep the best.

    Every candidate is drawn from the same dials, and the score rewards
    matching them - so the winner is the best hook *of the kind asked for*.
    """
    winner: Hook | None = None
    for _ in range(max(1, params.candidates)):
        h = generate(params, scale, rng, unit_steps)
        tones = realise(h, scale, chord_at)
        s, rep = score(h, tones, scale, chord_at)
        h.score, h.report = s, rep
        if winner is None or s > winner.score:
            winner = h
    return winner


def params_for(feel: str, rng: random.Random, family: str = "") -> HookParams:
    """Draw hook params for a feel. Ranges, not presets.

    Garage (and dubstep, its child) reach for the chopped vocal hook; a
    garage record leans on syncopated, offbeat figures that dance round the
    2-step kick."""
    half = feel == "half_time"
    garage = family == "garage" or (family == "dubstep" and not half)
    if garage:
        rhythms = (("syncopated", 4), ("offbeat", 3), ("dotted", 3), ("stabs", 2),
                   ("pickup", 2), ("flowing", 1), ("longshort", 1), ("triplet", 1))
    elif half:
        rhythms = (("dotted", 4), ("syncopated", 4), ("longshort", 3), ("triplet", 2),
                   ("stabs", 3), ("pickup", 2), ("offbeat", 1), ("flowing", 1))
    else:
        rhythms = (("offbeat", 3), ("pickup", 3), ("dotted", 3), ("flowing", 3),
                   ("longshort", 2), ("syncopated", 2), ("triplet", 1), ("stabs", 1))
    names, weights = zip(*rhythms)
    #                   call riff song stabs drone arp chop
    phrasing_weights = ((2, 3, 1, 1, 0, 1, 5) if garage else
                        (3, 2, 2, 3, 1, 1, 1) if half else (3, 2, 3, 1, 2, 1, 0))
    return HookParams(
        form=rng.choices(FORMS, (4, 3, 3, 3))[0],
        phrasing=rng.choices(PHRASINGS, phrasing_weights)[0],
        rhythm=rng.choices(names, weights)[0],
        unit_bars=2 if half else rng.choice((1, 2, 2)),
        density=round(rng.uniform(0.15, 0.65), 2),
        syncopation=round(rng.uniform(0.2, 0.85), 2),
        motion=round(rng.uniform(0.2, 0.8), 2),
        repeat=round(rng.uniform(0.05, 0.5), 2),
        leap=rng.choice((3, 4, 4, 5)),
        climax=round(rng.uniform(0.5, 0.8), 2),
        # Most hooks live inside a fifth or a sixth; a wide one is the
        # exception that makes the rule audible. Garage and 140 leads are
        # tighter still - a chop, a riff, a stab and its octave.
        span=rng.choice((4, 5, 5, 6, 6, 7) if garage or family == "dubstep"
                        else (4, 5, 5, 6, 7, 8)),
        repetition=round(rng.uniform(0.6, 0.95), 2),
        colour=round(rng.uniform(0.3, 0.9), 2),
        breath=round(rng.uniform(0.25, 0.4), 2),
        contour=rng.choice(HOOK_CONTOURS),
        contrast=round(rng.uniform(0.4, 0.9), 2),
        # Drawn last, so every draw above is what it always was.
        outside=round(rng.uniform(0.0, 0.3), 2),
    )
# The other way round: a contrasting theme moves where the hook does not.
_OPPOSITE = {"arch": "valley", "valley": "arch", "ascending": "descending",
             "descending": "ascending", "falling_tail": "ascending", "zigzag": "arch"}


def contrast_params(params: HookParams) -> HookParams:
    """A B-theme's params: the hook's rhythm idiom and kind of lead - so the
    two are plainly one song - with the contour turned around and the
    motion pushed the other way."""
    # Contrast of character, too: a busy first theme gets a singing second,
    # a sparse one a busier answer.
    density = round(max(0.15, min(0.9, 1.0 - params.density * 0.8)), 2)
    return replace(params, contour=_OPPOSITE.get(params.contour, "valley"), density=density,
                   motion=round(1.0 - params.motion * 0.6 - 0.2, 2) if params.motion > 0.5
                   else round(min(1.0, params.motion + 0.35), 2),
                   climax=round(1.0 - params.climax * 0.5, 2))


def fit(tones: list[Tone], scale: Scale, chord_at) -> list[Tone]:
    """Clear strong-beat semitone clashes from tones placed by hand (a build's
    climbing fragment, an outro's dissolving cell) against the chords under
    them - the same rule `realise` applies to a full pass."""
    return [_avoid_clash(t, scale, chord_at(t.step)) for t in tones]
