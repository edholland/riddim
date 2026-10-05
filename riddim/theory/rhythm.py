"""Rhythm: patterns, Euclidean distribution, claves and the dub riddims.

Three ideas do the work here.

1. **Metric weight.** Every step in a bar has a strength (downbeat strongest,
   odd 16ths weakest). Thinning a pattern means removing the weakest hits
   first; landing a melody note means preferring the strongest. One function
   drives both.

2. **Euclidean distribution.** Spreading N hits as evenly as possible over M
   steps produces, for free, most of the rhythms that sound good across world
   music - tresillo, cinquillo, the standard bell patterns.

3. **Rests are the material.** In dub the pattern is defined by the holes. So
   the generators here start from a full grid and take away, rather than
   starting from silence and adding, which is why they never produce the
   even-eighths mush that a naive random generator does.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace

from ..units import TICKS_PER_16TH, STEPS_PER_BAR


# --------------------------------------------------------------------------
# Metric weight
# --------------------------------------------------------------------------

def metric_weight(step: int, steps_per_bar: int = STEPS_PER_BAR) -> float:
    """Metric strength of a step, 0 (weakest) to 4 (bar downbeat).

    The standard metric hierarchy: a position is as strong as the largest
    subdivision it begins.
    """
    s = step % steps_per_bar
    if s == 0:
        return 4.0
    if steps_per_bar % 2 == 0 and s % (steps_per_bar // 2) == 0:
        return 3.0
    if steps_per_bar % 4 == 0 and s % (steps_per_bar // 4) == 0:
        return 2.0
    if s % 2 == 0:
        return 1.0
    return 0.0


def is_offbeat(step: int, steps_per_bar: int = STEPS_PER_BAR,
               beat_steps: int | None = None) -> bool:
    """True on the '&' of a beat - the upbeats.

    This is where the skank lives, and it is the single most identifying
    feature of reggae and dub.

    `beat_steps` is the length of the *felt* beat, which is not always the
    written quarter note. In half-time the backbeat falls on 3 rather than 2
    and 4, so the felt beat is a half note - eight steps, not four - and the
    upbeats that matter are the halfway points of those. Using the written
    quarter in half-time gives four stabs a bar where there should be two,
    which is what turns a skank into chatter.
    """
    if beat_steps is None:
        beat_steps = steps_per_bar // 4
    return step % beat_steps == beat_steps // 2


# --------------------------------------------------------------------------
# Patterns
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Hit:
    """One onset. `step` may be fractional to express micro-timing."""

    step: float
    velocity: int = 100
    length: float = 1.0  # in steps
    tag: str = ""

    def tick(self, ticks_per_step: int = TICKS_PER_16TH) -> int:
        return int(round(self.step * ticks_per_step))

    def tick_length(self, ticks_per_step: int = TICKS_PER_16TH) -> int:
        return max(1, int(round(self.length * ticks_per_step)))


@dataclass
class Pattern:
    """A sequence of hits over a fixed number of steps."""

    length: int  # in steps
    hits: list[Hit] = field(default_factory=list)
    steps_per_bar: int = STEPS_PER_BAR
    name: str = ""

    @property
    def bars(self) -> float:
        return self.length / self.steps_per_bar

    def sorted_hits(self) -> list[Hit]:
        return sorted(self.hits, key=lambda h: h.step)

    def steps(self) -> list[int]:
        return [int(h.step) for h in self.sorted_hits()]

    def density(self) -> float:
        """Fraction of available steps that are struck."""
        return len(self.hits) / self.length if self.length else 0.0

    def copy(self) -> "Pattern":
        return Pattern(self.length, list(self.hits), self.steps_per_bar, self.name)

    def merged(self, other: "Pattern") -> "Pattern":
        """Union of two patterns; later hits win on collision."""
        by_step: dict[float, Hit] = {h.step: h for h in self.hits}
        for h in other.hits:
            by_step[h.step] = h
        return Pattern(
            max(self.length, other.length),
            sorted(by_step.values(), key=lambda h: h.step),
            self.steps_per_bar,
            self.name,
        )

    def repeated(self, times: int) -> "Pattern":
        out: list[Hit] = []
        for i in range(times):
            offset = i * self.length
            out.extend(replace(h, step=h.step + offset) for h in self.hits)
        return Pattern(self.length * times, out, self.steps_per_bar, self.name)

    def __len__(self) -> int:
        return len(self.hits)


def from_steps(
    steps: list[int],
    length: int = STEPS_PER_BAR,
    velocity: int = 100,
    note_length: float = 1.0,
    name: str = "",
) -> Pattern:
    return Pattern(
        length,
        [Hit(float(s), velocity, note_length) for s in steps],
        name=name,
    )


def from_bools(
    flags: list[bool], velocity: int = 100, note_length: float = 1.0, name: str = ""
) -> Pattern:
    return Pattern(
        len(flags),
        [Hit(float(i), velocity, note_length) for i, f in enumerate(flags) if f],
        name=name,
    )


def from_string(text: str, velocity: int = 100, name: str = "") -> Pattern:
    """Parse 'x..x..x.' into a pattern. 'x'/'X' hit, anything else rest.

    'X' is an accent, worth +20 velocity.
    """
    hits = []
    for i, ch in enumerate(text):
        if ch in "xX":
            hits.append(Hit(float(i), min(127, velocity + (20 if ch == "X" else 0))))
    return Pattern(len(text), hits, name=name)


# --------------------------------------------------------------------------
# Euclidean rhythms
# --------------------------------------------------------------------------

def bjorklund(pulses: int, steps: int) -> list[bool]:
    """Distribute `pulses` as evenly as possible over `steps`.

    Bjorklund's algorithm - the same recursion as Euclid's GCD. E(3,8) gives
    the tresillo, E(5,8) the cinquillo, E(5,16) a bell pattern.
    """
    if steps <= 0:
        return []
    if pulses <= 0:
        return [False] * steps
    if pulses >= steps:
        return [True] * steps

    groups_a: list[list[bool]] = [[True] for _ in range(pulses)]
    groups_b: list[list[bool]] = [[False] for _ in range(steps - pulses)]

    while len(groups_b) > 1 and len(groups_a) > 1:
        n = min(len(groups_a), len(groups_b))
        merged = [groups_a[i] + groups_b[i] for i in range(n)]
        remainder = groups_a[n:] if len(groups_a) > n else groups_b[n:]
        groups_a, groups_b = merged, remainder

    return [flag for group in groups_a + groups_b for flag in group]


def euclid(
    pulses: int,
    steps: int = STEPS_PER_BAR,
    rotation: int = 0,
    velocity: int = 100,
    name: str = "",
) -> Pattern:
    """Euclidean pattern, optionally rotated left by `rotation` steps."""
    flags = bjorklund(pulses, steps)
    if rotation:
        r = rotation % steps
        flags = flags[r:] + flags[:r]
    return from_bools(flags, velocity=velocity, name=name or f"E({pulses},{steps})")


# --------------------------------------------------------------------------
# Clave and riddim templates
# --------------------------------------------------------------------------

# 16-step templates. These are the load-bearing patterns of Caribbean music
# and they are what make a programmed riddim sound played rather than typed.
CLAVES: dict[str, str] = {
    "son_3_2": "x..x..x...x.x...",
    "son_2_3": "..x.x...x..x..x.",
    "rumba_3_2": "x..x...x..x.x...",
    "rumba_2_3": "..x.x...x..x...x",
    "bossa": "x..x..x...x..x..",
    "tresillo": "x..x..x.x..x..x.",
    "cascara": "x.xx.x.xx.x.xx.x",
}

# Drum riddims. The kick is the genre signature: where it sits tells you
# immediately whether you are in one-drop, steppers or half-time.
DRUM_RIDDIMS: dict[str, dict[str, str]] = {
    # Beat 3 only. The "drop" is the missing beat 1 - the whole feel hangs
    # on that hole.
    "one_drop": {
        "kick": "........x.......",
        "snare": "........x.......",
        "hat": "..x...x...x...x.",
        "rim": "....x.......x...",
    },
    # Four on the floor, militant. Late-70s dub and anything by Sly Dunbar.
    "steppers": {
        "kick": "x...x...x...x...",
        "snare": "........x.......",
        "hat": "..x...x...x...x.",
        "rim": "....x.......x...",
    },
    # Kick on 1 and 3 with a pickup; more forward motion than one-drop.
    "rockers": {
        "kick": "x.....x.x.......",
        "snare": "........x.......",
        "hat": "..x...x...x...x.",
        "rim": "....x.......x...",
    },
    # 140bpm written, 70bpm felt. Kick on 1, snare on 3, and the whole bar
    # is mostly empty so the bass can be the event.
    "half_time": {
        "kick": "x.......x.x.....",
        "snare": "........x.......",
        "hat": "..x...x...x...x.",
        "rim": "................",
    },
    # Sparser still - for drops where the sub carries everything.
    "half_time_sparse": {
        "kick": "x...............",
        "snare": "........x.......",
        "hat": "....x.......x...",
        "rim": "................",
    },
    # The kick pushes onto the & of 2, right before the snare. That
    # anticipation is the difference between a beat that sits and one that
    # lurches forward.
    "half_time_push": {
        "kick": "x.....x.x.......",
        "snare": "........x.......",
        "hat": "..x...x...x...x.",
        "rim": "................",
    },
    # Double kick on the one. Heavier, more aggressive - a drop kick.
    "half_time_double": {
        "kick": "x.x.....x.......",
        "snare": "........x.......",
        "hat": "....x.......x...",
        "rim": "..............x.",
    },
    # Kick in a 3-3-2 grouping against the half-time snare. The Latin cell
    # under a dubstep backbeat - a lot of the best drops are this.
    "half_time_tresillo": {
        "kick": "x..x..x.x.......",
        "snare": "........x.......",
        "hat": "..x...x...x...x.",
        "rim": "................",
    },
    # Snare doubled on the & of 4 as a pickup. Rolling, almost breakbeat.
    "half_time_roll": {
        "kick": "x.........x.....",
        "snare": "........x.....x.",
        "hat": "x.x.x.x.x.x.x.x.",
        "rim": "................",
    },
    # One-drop with the rim doubled - the "flying cymbal" era feel.
    "one_drop_flyer": {
        "kick": "........x.......",
        "snare": "........x.......",
        "hat": "..x.x.x.x.x.x.x.",
        "rim": "....x.....x.x...",
    },
    # One-drop with a kick pickup on the & of 4, pushing into the next bar.
    # The drop on beat 1 is still there - the pickup makes it land harder.
    "one_drop_pickup": {
        "kick": "........x.....x.",
        "snare": "........x.......",
        "hat": "..x...x...x...x.",
        "rim": "....x.......x...",
    },
    # Steppers with a skipped third kick. Militant but breathing.
    "steppers_skip": {
        "kick": "x...x.......x...",
        "snare": "........x.......",
        "hat": "..x...x...x...x.",
        "rim": "....x.......x...",
    },
}

# Chord-stab rhythms, as 16-step strings, grouped by feel. In a straight
# feel the felt beat is a quarter (4 steps); in half-time it is a half note
# (8 steps). None of these put a stab on step 0 - a chord on the downbeat is
# the one thing that instantly stops this sounding like dub.
STAB_PATTERNS: dict[str, dict[str, str]] = {
    "straight": {
        "skank": "..x...x...x...x.",         # every upbeat - the reggae guitar
        "double_chop": "..xx..xx..xx..xx",   # the double-strum chop
        "half_skank": "......x.......x.",    # upbeats of 2 and 4 only
        "three_four": "..x.......x...x.",    # skips the & of 2 - a hole
        "late_skank": "......x...x...x.",    # sits out the first upbeat
    },
    "half_time": {
        "offbeats": "....x.......x...",      # the & of each felt beat
        "push": "......x.....x...",          # pushed onto the & before the snare
        "single": "............x...",        # one stab, after the snare
        "answer": "....x.....x.....",        # responds to kick and snare
        "tresillo": "...x..x.....x...",      # 3-3-2 against the backbeat
        "late_pair": "..........x...x.",     # two stabs in the last quarter
    },
}


def stab(feel: str, name: str, velocity: int = 90, note_length: float = 0.45) -> Pattern:
    """A named chord-stab rhythm for a feel. See STAB_PATTERNS."""
    if feel not in STAB_PATTERNS:
        raise ValueError(
            f"unknown feel {feel!r}; known: {', '.join(sorted(STAB_PATTERNS))}"
        )
    table = STAB_PATTERNS[feel]
    if name not in table:
        raise ValueError(
            f"feel {feel!r} has no stab pattern {name!r}; has: {', '.join(table)}"
        )
    hits = [
        Hit(float(i), velocity, note_length)
        for i, ch in enumerate(table[name]) if ch == "x"
    ]
    return Pattern(len(table[name]), hits, name=f"{feel}.{name}")


def clave(name: str, velocity: int = 100) -> Pattern:
    if name not in CLAVES:
        raise ValueError(f"unknown clave {name!r}; known: {', '.join(sorted(CLAVES))}")
    return from_string(CLAVES[name], velocity=velocity, name=name)


def riddim(name: str, voice: str, velocity: int = 100) -> Pattern:
    if name not in DRUM_RIDDIMS:
        raise ValueError(
            f"unknown riddim {name!r}; known: {', '.join(sorted(DRUM_RIDDIMS))}"
        )
    voices = DRUM_RIDDIMS[name]
    if voice not in voices:
        raise ValueError(
            f"riddim {name!r} has no voice {voice!r}; has: {', '.join(voices)}"
        )
    return from_string(voices[voice], velocity=velocity, name=f"{name}.{voice}")


def skank(
    steps_per_bar: int = STEPS_PER_BAR,
    velocity: int = 90,
    note_length: float = 0.5,
    beat_steps: int | None = None,
) -> Pattern:
    """The offbeat chord stab: every '&' of the felt beat, nothing on it.

    Short notes. A skank that rings into the downbeat stops being a skank.

    Pass `beat_steps=8` for a half-time feel, which gives two stabs a bar
    instead of four - see `is_offbeat`.
    """
    hits = [
        Hit(float(s), velocity, note_length)
        for s in range(steps_per_bar)
        if is_offbeat(s, steps_per_bar, beat_steps)
    ]
    return Pattern(steps_per_bar, hits, steps_per_bar, "skank")


def bubble(steps_per_bar: int = STEPS_PER_BAR, velocity: int = 80) -> Pattern:
    """The organ bubble: the other half of reggae keys.

    A shuffle figure that plays the '&' loud and the preceding 16th quiet,
    giving that rolling, breathing motion under the skank.
    """
    hits: list[Hit] = []
    quarter = steps_per_bar // 4
    for beat in range(4):
        base = beat * quarter
        hits.append(Hit(float(base + quarter // 4), velocity - 25, 0.4))
        hits.append(Hit(float(base + quarter // 2), velocity, 0.5))
    return Pattern(steps_per_bar, hits, steps_per_bar, "bubble")


# --------------------------------------------------------------------------
# Transformations
# --------------------------------------------------------------------------

def rotate(pattern: Pattern, steps: int) -> Pattern:
    """Rotate hits within the pattern length, wrapping around."""
    if pattern.length == 0:
        return pattern.copy()
    hits = [
        replace(h, step=(h.step + steps) % pattern.length) for h in pattern.hits
    ]
    return Pattern(pattern.length, sorted(hits, key=lambda h: h.step),
                   pattern.steps_per_bar, pattern.name)


def displace(pattern: Pattern, steps: float, which=None) -> Pattern:
    """Shift some hits later without wrapping - creates syncopation.

    `which` is a predicate on step index; default shifts everything.
    """
    hits = []
    for h in pattern.hits:
        if which is None or which(h.step):
            moved = h.step + steps
            if 0 <= moved < pattern.length:
                hits.append(replace(h, step=moved))
            # A hit pushed off the end is simply gone - that is a legitimate
            # rhythmic result, not an error.
        else:
            hits.append(h)
    return Pattern(pattern.length, sorted(hits, key=lambda h: h.step),
                   pattern.steps_per_bar, pattern.name)


def thin(pattern: Pattern, keep: float, rng: random.Random | None = None) -> Pattern:
    """Remove hits until `keep` (0..1) of them remain, weakest metric first.

    This is how a section drops in energy without changing identity: the
    downbeats survive, the decoration goes. Ties are broken randomly so
    repeated calls do not always strip the same 16th.
    """
    if keep >= 1.0 or not pattern.hits:
        return pattern.copy()
    keep = max(0.0, keep)
    rng = rng or random.Random()
    target = max(1, int(round(len(pattern.hits) * keep))) if keep > 0 else 0

    ranked = sorted(
        pattern.hits,
        key=lambda h: (
            -metric_weight(int(h.step), pattern.steps_per_bar),
            -h.velocity,
            rng.random(),
        ),
    )
    kept = sorted(ranked[:target], key=lambda h: h.step)
    return Pattern(pattern.length, kept, pattern.steps_per_bar, pattern.name)


def thicken(
    pattern: Pattern,
    add: int,
    rng: random.Random | None = None,
    velocity: int = 70,
    prefer_offbeat: bool = True,
) -> Pattern:
    """Add `add` hits on currently empty steps.

    Prefers offbeats, because filling in the strong beats just makes the
    pattern square - the energy should come from syncopation.
    """
    rng = rng or random.Random()
    taken = {int(h.step) for h in pattern.hits}
    free = [s for s in range(pattern.length) if s not in taken]
    if not free:
        return pattern.copy()

    def score(step: int) -> float:
        w = metric_weight(step, pattern.steps_per_bar)
        bias = -w if prefer_offbeat else w
        return bias + rng.random() * 0.5

    chosen = sorted(free, key=score)[: max(0, add)]
    hits = list(pattern.hits) + [Hit(float(s), velocity, 1.0) for s in chosen]
    return Pattern(pattern.length, sorted(hits, key=lambda h: h.step),
                   pattern.steps_per_bar, pattern.name)


def accent(
    pattern: Pattern,
    template: str | None = None,
    strong: int = 112,
    weak: int = 78,
) -> Pattern:
    """Re-velocity a pattern.

    With no template, velocity follows metric weight, which is what a human
    does without thinking about it.
    """
    hits = []
    for h in pattern.hits:
        if template:
            ch = template[int(h.step) % len(template)]
            vel = strong if ch in "xX" else weak
        else:
            w = metric_weight(int(h.step), pattern.steps_per_bar)
            vel = int(weak + (strong - weak) * (w / 4.0))
        hits.append(replace(h, velocity=max(1, min(127, vel))))
    return Pattern(pattern.length, hits, pattern.steps_per_bar, pattern.name)


def swing(pattern: Pattern, amount: float = 0.2) -> Pattern:
    """Delay every odd 16th by `amount` of a step. 0 is straight, 0.33 triplet.

    Applied as fractional step offsets, which survive all the way to the
    Deluge's tick grid - 24 ticks per step means we can resolve about 4% of a
    step, plenty for feel.
    """
    hits = []
    for h in pattern.hits:
        offset = amount if int(h.step) % 2 == 1 else 0.0
        hits.append(replace(h, step=h.step + offset))
    return Pattern(pattern.length, hits, pattern.steps_per_bar, pattern.name)


def lay_back(pattern: Pattern, steps: float = 0.06, tags=("snare",)) -> Pattern:
    """Nudge selected voices late. A snare a hair behind the beat is the
    difference between 'programmed' and 'played'."""
    hits = [
        replace(h, step=h.step + steps) if (not tags or h.tag in tags) else h
        for h in pattern.hits
    ]
    return Pattern(pattern.length, hits, pattern.steps_per_bar, pattern.name)


def humanize(
    pattern: Pattern,
    timing: float = 0.03,
    velocity: int = 8,
    rng: random.Random | None = None,
) -> Pattern:
    """Small random timing and velocity jitter."""
    rng = rng or random.Random()
    hits = []
    for h in pattern.hits:
        step = max(0.0, h.step + rng.uniform(-timing, timing))
        vel = max(1, min(127, h.velocity + rng.randint(-velocity, velocity)))
        hits.append(replace(h, step=step, velocity=vel))
    return Pattern(pattern.length, hits, pattern.steps_per_bar, pattern.name)


def mask(pattern: Pattern, keep_steps) -> Pattern:
    """Keep only hits whose step passes `keep_steps` (a set or predicate)."""
    if callable(keep_steps):
        test = keep_steps
    else:
        allowed = set(keep_steps)
        test = lambda s: int(s) in allowed  # noqa: E731
    return Pattern(
        pattern.length,
        [h for h in pattern.hits if test(h.step)],
        pattern.steps_per_bar,
        pattern.name,
    )


def every_other_bar(pattern: Pattern, variant: Pattern, bars: int = 2) -> Pattern:
    """Alternate between two one-bar patterns to make a longer phrase.

    Two-bar phrasing is the cheapest possible cure for loop fatigue.
    """
    out: list[Hit] = []
    for i in range(bars):
        source = pattern if i % 2 == 0 else variant
        out.extend(replace(h, step=h.step + i * pattern.length) for h in source.hits)
    return Pattern(pattern.length * bars, out, pattern.steps_per_bar, pattern.name)
