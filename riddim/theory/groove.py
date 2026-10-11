"""Parametric rhythm generation.

This replaces the tables of hand-written 16-step strings. A groove is
*specified* by musical quantities and *generated* from them:

- **density** - what fraction of the grid is struck.
- **syncopation** - how much the hits prefer weak positions over strong
  ones. 0 sits square on the beat; 1 avoids it.
- **anchors** - positions that must be struck. This is where genre lives: a
  one-drop is a kick anchored on beat 3 and nowhere else; steppers is a kick
  anchored on every beat; half-time is a snare anchored on 3.
- **avoid** - positions that must not be struck. A chord stab that never
  lands on the downbeat is `avoid={0}`.
- **min_gap** - the closest two hits may be, in steps. A sub bass cannot
  retrigger every 16th and still be a sub.

Given those, hits are placed by scoring every step - metric weight blended
against its inverse by the syncopation amount, plus a little noise - and
sampling in score order until the density is met. The named riddims of the
genre are just particular settings of these knobs, and so are a great many
riddims nobody has named yet.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace

from .rhythm import Hit, Pattern, metric_weight
from ..units import STEPS_PER_BAR


@dataclass(frozen=True)
class GrooveSpec:
    """Everything needed to generate one voice's rhythm for one bar."""

    density: float = 0.25
    syncopation: float = 0.3
    anchors: frozenset[int] = frozenset()
    avoid: frozenset[int] = frozenset()
    min_gap: int = 1
    steps: int = STEPS_PER_BAR
    # Extra pull toward the upbeats specifically (the '&' of each beat),
    # independent of general syncopation. This is the skank dimension.
    offbeat_bias: float = 0.0
    # Felt beat length in steps: 4 straight, 8 half-time. Sets which
    # positions count as upbeats.
    beat_steps: int = 4
    # Velocity range; strong positions get the top of it.
    velocity: tuple[int, int] = (78, 112)
    note_length: float = 1.0

    def target_hits(self) -> int:
        return max(len(self.anchors), int(round(self.density * self.steps)))


def _score(step: int, spec: GrooveSpec, rng: random.Random) -> float:
    weight = metric_weight(step, spec.steps) / 4.0          # 0..1
    base = (1.0 - spec.syncopation) * weight + spec.syncopation * (1.0 - weight)
    if spec.offbeat_bias:
        half = spec.beat_steps // 2
        is_up = step % spec.beat_steps == half
        base += spec.offbeat_bias * (1.0 if is_up else -0.5)
    return base + rng.uniform(0.0, 0.35)


def generate(spec: GrooveSpec, rng: random.Random | None = None) -> Pattern:
    """Generate one bar of rhythm from a spec. Deterministic for an rng."""
    rng = rng or random.Random()
    chosen: set[int] = set(s for s in spec.anchors if 0 <= s < spec.steps)
    target = spec.target_hits()

    def too_close(step: int) -> bool:
        return any(abs(step - c) < spec.min_gap or abs(step - c) > spec.steps - spec.min_gap
                   for c in chosen)

    candidates = [
        s for s in range(spec.steps)
        if s not in chosen and s not in spec.avoid
    ]
    ranked = sorted(candidates, key=lambda s: -_score(s, spec, rng))
    for step in ranked:
        if len(chosen) >= target:
            break
        if too_close(step):
            continue
        chosen.add(step)

    lo, hi = spec.velocity
    hits = []
    for step in sorted(chosen):
        w = metric_weight(step, spec.steps) / 4.0
        vel = int(round(lo + (hi - lo) * (0.35 + 0.65 * w)))
        hits.append(Hit(float(step), max(1, min(127, vel)), spec.note_length))
    return Pattern(spec.steps, hits, spec.steps)


# --------------------------------------------------------------------------
# Drum kit as a set of specs
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class DrumSpec:
    """The parameters that define a drum feel, voice by voice.

    `feel` is "straight" (backbeat on 2 and 4) or "half_time" (backbeat on
    3). The kick's anchors are the single most genre-defining choice: {8}
    alone is one-drop, {0,4,8,12} is steppers, {0} with the backbeat on 8 is
    dubstep half-time.
    """

    feel: str = "straight"
    kick_anchors: frozenset[int] = frozenset({0})
    kick_density: float = 0.12
    kick_syncopation: float = 0.4
    snare_ghosts: float = 0.0          # density of extra snare hits
    hat_density: float = 0.25
    hat_offbeat_bias: float = 0.6
    rim_density: float = 0.0
    swing: float = 0.0
    # Where the snare lands. Defaults from the feel (2 and 4 straight, 3 in
    # half-time) but a one-drop is straight-feel hats with the snare on 3,
    # together with the kick - that coincidence *is* the one-drop.
    backbeat_override: frozenset[int] | None = None
    # Which kit vocabulary the hats, ghosts and rims speak: "roots" (the
    # sound-system kit - chick on the offbeat, rim on the clave) or
    # "garage" (2-step - hats skipping into the backbeat, the ghost snare
    # that pushes the one, rimshots on the in-between sixteenths).
    idiom: str = "roots"

    @property
    def beat_steps(self) -> int:
        return 8 if self.feel == "half_time" else 4

    @property
    def backbeat(self) -> frozenset[int]:
        if self.backbeat_override is not None:
            return self.backbeat_override
        return frozenset({8}) if self.feel == "half_time" else frozenset({4, 12})

    def specs(self) -> dict[str, GrooveSpec]:
        bs = self.beat_steps
        return {
            "kick": GrooveSpec(
                density=self.kick_density, syncopation=self.kick_syncopation,
                anchors=self.kick_anchors, avoid=self.backbeat, min_gap=2,
                beat_steps=bs, velocity=(96, 118), note_length=0.9,
            ),
            "snare": GrooveSpec(
                density=self.snare_ghosts, syncopation=0.7,
                anchors=self.backbeat, avoid=self.kick_anchors - self.backbeat,
                min_gap=2, beat_steps=bs, velocity=(60, 116), note_length=0.9,
            ),
            "hat": GrooveSpec(
                density=self.hat_density, syncopation=0.5,
                offbeat_bias=self.hat_offbeat_bias, min_gap=1, beat_steps=bs,
                velocity=(58, 92), note_length=0.5,
            ),
            "rim": GrooveSpec(
                density=self.rim_density, syncopation=0.2,
                avoid=self.backbeat | self.kick_anchors, min_gap=3,
                beat_steps=bs, velocity=(66, 96), note_length=0.5,
            ),
        }


# The classic feels, expressed as parameters rather than as strings - so
# they are reachable points in a continuous space, not the only points.
ONE_DROP = DrumSpec("straight", kick_anchors=frozenset({8}), kick_density=0.06,
                    hat_density=0.25, rim_density=0.12,
                    backbeat_override=frozenset({8}))
STEPPERS = DrumSpec("straight", kick_anchors=frozenset({0, 4, 8, 12}),
                    kick_density=0.25, hat_density=0.25, rim_density=0.12)
ROCKERS = DrumSpec("straight", kick_anchors=frozenset({0, 8}), kick_density=0.18,
                   kick_syncopation=0.6, hat_density=0.25, rim_density=0.12)
HALF_TIME = DrumSpec("half_time", kick_anchors=frozenset({0}), kick_density=0.12,
                     kick_syncopation=0.55, hat_density=0.25)
HALF_TIME_ROLL = DrumSpec("half_time", kick_anchors=frozenset({0}), kick_density=0.12,
                          snare_ghosts=0.12, hat_density=0.5, hat_offbeat_bias=0.0)

# The rest of each idiom's family. Each is a different *anchor* - the kick
# positions that make a feel itself - not a density tweak of another one.
ONE_DROP_FLYER = DrumSpec("straight", kick_anchors=frozenset({8}), kick_density=0.06,
                          hat_density=0.55, hat_offbeat_bias=0.0, rim_density=0.08,
                          backbeat_override=frozenset({8}))         # running hats
ONE_DROP_PICKUP = DrumSpec("straight", kick_anchors=frozenset({8, 14}), kick_density=0.12,
                           hat_density=0.25, rim_density=0.12,
                           backbeat_override=frozenset({8}))        # kicks into the next bar
STEPPERS_SKIP = DrumSpec("straight", kick_anchors=frozenset({0, 4, 8, 12}),
                         kick_density=0.3, kick_syncopation=0.8, hat_density=0.35,
                         hat_offbeat_bias=0.9, rim_density=0.15)   # a skipped extra kick
HALF_TIME_PUSH = DrumSpec("half_time", kick_anchors=frozenset({0, 14}), kick_density=0.14,
                          kick_syncopation=0.5, hat_density=0.25)  # pushes into the bar
HALF_TIME_DOUBLE = DrumSpec("half_time", kick_anchors=frozenset({0, 10}), kick_density=0.14,
                            kick_syncopation=0.4, hat_density=0.3)  # boom ... clap-boom
HALF_TIME_TRESILLO = DrumSpec("half_time", kick_anchors=frozenset({0, 6}), kick_density=0.14,
                              kick_syncopation=0.3, hat_density=0.3,
                              hat_offbeat_bias=0.2)                # 3-3-2 against the snare
HALF_TIME_SPARSE = DrumSpec("half_time", kick_anchors=frozenset({0}), kick_density=0.06,
                            kick_syncopation=0.6, hat_density=0.15)

# UK garage - the other parent of dubstep. A 2-step bar leaves out the
# second and fourth kicks a house record would have and puts one on the &
# of 3 instead: boom . clap . . boom-clap. The shuffle is the Deluge's job;
# the skip in the hats is ours.
TWO_STEP = DrumSpec("straight", kick_anchors=frozenset({0, 10}), kick_density=0.12,
                    kick_syncopation=0.6, snare_ghosts=0.1, hat_density=0.3,
                    hat_offbeat_bias=0.8, rim_density=0.1, idiom="garage")
TWO_STEP_SKIP = DrumSpec("straight", kick_anchors=frozenset({0, 11}), kick_density=0.12,
                         kick_syncopation=0.7, snare_ghosts=0.2, hat_density=0.45,
                         hat_offbeat_bias=0.8, rim_density=0.15, idiom="garage")  # the late kick
TWO_STEP_ROLL = DrumSpec("straight", kick_anchors=frozenset({0, 6, 10}), kick_density=0.19,
                         kick_syncopation=0.5, snare_ghosts=0.1, hat_density=0.3,
                         hat_offbeat_bias=0.8, rim_density=0.1, idiom="garage")  # boom . . b-boom
GARAGE_4X4 = DrumSpec("straight", kick_anchors=frozenset({0, 4, 8, 12}), kick_density=0.25,
                      snare_ghosts=0.1, hat_density=0.3, hat_offbeat_bias=1.0,
                      rim_density=0.1, idiom="garage")                       # speed garage
# Dubstep's garage inheritance: half-time with the 2-step's late kick and
# skipping hats - Skream and Benga rather than the sound system.
HALF_TIME_SKIP = DrumSpec("half_time", kick_anchors=frozenset({0, 11}), kick_density=0.14,
                          kick_syncopation=0.5, snare_ghosts=0.15, hat_density=0.45,
                          hat_offbeat_bias=0.6, rim_density=0.1, idiom="garage")
HALF_TIME_GARAGE = DrumSpec("half_time", kick_anchors=frozenset({0, 3}), kick_density=0.14,
                            kick_syncopation=0.4, snare_ghosts=0.1, hat_density=0.3,
                            hat_offbeat_bias=0.8, rim_density=0.15, idiom="garage")

FEELS = {
    "one_drop": ONE_DROP, "one_drop_flyer": ONE_DROP_FLYER,
    "one_drop_pickup": ONE_DROP_PICKUP, "steppers": STEPPERS,
    "steppers_skip": STEPPERS_SKIP, "rockers": ROCKERS,
    "half_time": HALF_TIME, "half_time_roll": HALF_TIME_ROLL,
    "half_time_push": HALF_TIME_PUSH, "half_time_double": HALF_TIME_DOUBLE,
    "half_time_tresillo": HALF_TIME_TRESILLO, "half_time_sparse": HALF_TIME_SPARSE,
    "two_step": TWO_STEP, "two_step_skip": TWO_STEP_SKIP, "two_step_roll": TWO_STEP_ROLL,
    "garage_4x4": GARAGE_4X4, "half_time_skip": HALF_TIME_SKIP,
    "half_time_garage": HALF_TIME_GARAGE,
}


# Hat figures, sparse to dense: a drummer's ladder, not scattered hits.
_HATS = (
    (2, 6, 10, 14),                                   # the chick on the offbeats
    (0, 2, 4, 6, 8, 10, 12, 14),                      # eighths
    (0, 2, 4, 6, 7, 8, 10, 12, 14, 15),               # eighths with a skip
    tuple(range(16)),                                 # running sixteenths
)
# The garage ladder: the offbeat hat, then the sixteenth after it - the
# skip that, shuffled on the Deluge, is the 2-step sound - then both.
_GARAGE_HATS = (
    (2, 6, 10, 14),
    (2, 6, 7, 10, 14, 15),                            # skips into 3 and the one
    (2, 3, 6, 7, 10, 11, 14, 15),                     # paired: tss-ts tss-ts
    (0, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15),
)
# Ghost-snare and rim positions in the order a player reaches for them.
_GHOSTS = {"half_time": (15, 6, 11), "straight": (7, 15, 10)}
_RIMS = {"half_time": (12, 4, 14), "straight": (6, 14, 10)}
# Garage ghosts push into the one and answer the late kick; its rims sit
# on the sixteenths between - the woodblock and rimshot of a 2-step kit.
_GARAGE_GHOSTS = {"half_time": (15, 13, 5), "straight": (15, 9, 7)}
_GARAGE_RIMS = {"half_time": (3, 13, 6), "straight": (3, 13, 9)}


def drums(spec: DrumSpec, rng: random.Random | None = None) -> dict[str, Pattern]:
    """One bar of every kit voice, idiomatically.

    Kick is the spec's anchors plus its extra hits (placed by the scoring
    generator, then locked to the bass by the composer); snare is the
    backbeat plus ghosts from the idiomatic positions; hats are a figure
    from the ladder by density, offbeat-accented by `hat_offbeat_bias`; rim
    takes clave positions. Regular figures, not scattered hits - in a dense
    bar, scatter is what sounds out of time.
    """
    rng = rng or random.Random()
    specs = spec.specs()
    out = {"kick": generate(specs["kick"], rng)}
    feel = "half_time" if spec.feel == "half_time" else "straight"

    def hits(steps, lo, hi, accent):
        return Pattern(16, [Hit(float(s), hi if accent(s) else lo, 0.5) for s in steps], 16)

    garage = spec.idiom == "garage"
    backbeat = sorted(spec.backbeat)
    ghosts = list((_GARAGE_GHOSTS if garage else _GHOSTS)[feel][: min(3, int(round(spec.snare_ghosts * 10)))])
    snare = [Hit(float(s), 112, 0.9) for s in backbeat]
    snare += [Hit(float(s), 58, 0.9) for s in ghosts if s not in backbeat]
    out["snare"] = Pattern(16, sorted(snare, key=lambda h: h.step), 16)

    # Half-time is felt at half speed: running sixteenths at 140 are a blur,
    # so its ladder stops at eighths with a skip.
    # Garage stops at the paired skip too: past it the hats fill the
    # beats and the skip - the whole point - is gone.
    out["hat"] = hat_pattern(spec)

    rims = [s for s in (_GARAGE_RIMS if garage else _RIMS)[feel][: min(3, int(round(spec.rim_density * 10)))]
            if s not in backbeat and s not in spec.kick_anchors]
    out["rim"] = hits(rims, 72, 90, lambda s: s % 4 == 2)
    return out


def hat_pattern(spec: DrumSpec, climb: int = 0) -> Pattern:
    """The hat figure from the ladder by density - `climb` rungs higher,
    for a build tightening into a drop. Half-time stops at eighths with a
    skip (running sixteenths at 140 are a blur), and garage at the paired
    skip (past it the hats fill the beats and the skip is gone)."""
    garage = spec.idiom == "garage"
    backbeat = sorted(spec.backbeat)
    top = 2 if spec.feel == "half_time" or garage else len(_HATS) - 1
    tier = min(top, int(spec.hat_density * 5) + climb)
    figure = [s for s in (_GARAGE_HATS if garage else _HATS)[tier] if s not in backbeat or tier >= 1]
    off = spec.hat_offbeat_bias > 0.4
    accent = (lambda s: s % 4 == 2) if off else (lambda s: s % 4 == 0)

    # Three levels, the way a drummer plays them: the accent, the other
    # eighths at a medium touch, and the in-between sixteenths ghosted -
    # the garage skip just audible, the push into the next eighth.
    def vel(s):
        if accent(s):
            return 88
        if s % 2 == 0:
            return 68
        return 54 if garage else 50
    return Pattern(16, [Hit(float(s), vel(s), 0.5) for s in figure], 16)


def _kick_cap(spec: DrumSpec) -> float:
    """Half-time lives on its emptiness, and 2-step on the kicks it leaves
    out: too many and it is not that feel any more. A 2-step bar takes one
    kick beyond its anchors at most; four-to-the-floor garage, none to speak
    of."""
    if spec.idiom == "garage":
        return max(0.19, (len(spec.kick_anchors) + 1) / 16)
    return 0.22 if spec.feel == "half_time" else 0.4


def vary(spec: DrumSpec, rng: random.Random, amount: float = 0.15) -> DrumSpec:
    """Perturb a drum spec's continuous parameters.

    This is how a genre becomes a distribution: start at a named feel and
    move a little in every direction. The anchors stay - they are the
    identity - and everything else drifts.
    """
    def jitter(v, lo=0.0, hi=1.0):
        return max(lo, min(hi, v + rng.uniform(-amount, amount)))
    kick_cap = _kick_cap(spec)
    return replace(
        spec,
        kick_density=jitter(spec.kick_density, 0.0, kick_cap),
        kick_syncopation=jitter(spec.kick_syncopation),
        snare_ghosts=jitter(spec.snare_ghosts, 0.0, 0.3),
        hat_density=jitter(spec.hat_density, 0.1, 0.6),
        hat_offbeat_bias=jitter(spec.hat_offbeat_bias, -0.5, 1.0),
        rim_density=jitter(spec.rim_density, 0.0, 0.25),
    )


# --------------------------------------------------------------------------
# Stabs and bass rhythms
# --------------------------------------------------------------------------

def stab_spec(feel: str, density: float, syncopation: float = 0.8,
              note_length: float = 0.45) -> GrooveSpec:
    """A chord-stab rhythm: never on the downbeat, pulled to the upbeats.

    Density is per felt beat, so 1.0 in a straight feel is the full skank
    (four upbeats) and 1.0 in half-time is two stabs a bar.
    """
    bs = 8 if feel == "half_time" else 4
    upbeats = bs // 2
    hits_per_bar = density * (STEPS_PER_BAR / bs)
    return GrooveSpec(
        density=hits_per_bar / STEPS_PER_BAR, syncopation=syncopation,
        avoid=frozenset({0}), offbeat_bias=0.7, min_gap=2, beat_steps=bs,
        velocity=(72, 100), note_length=note_length,
    )


def bass_spec(feel: str, density: float, syncopation: float,
              anticipate: bool = False, anchor_root: bool = True) -> GrooveSpec:
    """A bassline's rhythm.

    `anticipate` puts a hit on the & of 4 - the push across the barline that
    makes a riddim lurch. `anchor_root` keeps a hit on the downbeat, which
    almost every bassline in this music has, one-drop being the exception.
    """
    bs = 8 if feel == "half_time" else 4
    anchors = set()
    if anchor_root:
        anchors.add(0)
    if anticipate:
        anchors.add(14)
    return GrooveSpec(
        density=density, syncopation=syncopation, anchors=frozenset(anchors),
        min_gap=2, beat_steps=bs, velocity=(88, 112),
    )


def describe(pattern: Pattern) -> str:
    """Render a pattern as a 16-step string, for display and tests."""
    grid = ["."] * pattern.length
    for h in pattern.hits:
        i = int(round(h.step))
        if 0 <= i < pattern.length:
            grid[i] = "X" if h.velocity > 100 else "x"
    return "".join(grid)


def intensify(spec: DrumSpec, amount: float = 1.0) -> DrumSpec:
    """The same feel, played harder: the kit a peak earns.

    More hats and ghost notes, a touch more kick, the rim joining in. The
    anchors and backbeat do not move - a drop is the riddim at full weight,
    not a different riddim.
    """
    half = spec.feel == "half_time"
    return replace(
        spec,
        hat_density=min(0.6 if half else 0.75, spec.hat_density + 0.22 * amount),
        snare_ghosts=min(0.25, spec.snare_ghosts + 0.07 * amount),
        kick_density=min(_kick_cap(spec), spec.kick_density + 0.04 * amount),
        rim_density=min(0.25, spec.rim_density + 0.05 * amount),
    )
