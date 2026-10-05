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

FEELS = {
    "one_drop": ONE_DROP, "steppers": STEPPERS, "rockers": ROCKERS,
    "half_time": HALF_TIME, "half_time_roll": HALF_TIME_ROLL,
}


def drums(spec: DrumSpec, rng: random.Random | None = None) -> dict[str, Pattern]:
    """Generate one bar for every voice of a kit."""
    rng = rng or random.Random()
    return {voice: generate(g, rng) for voice, g in spec.specs().items()}


def vary(spec: DrumSpec, rng: random.Random, amount: float = 0.15) -> DrumSpec:
    """Perturb a drum spec's continuous parameters.

    This is how a genre becomes a distribution: start at a named feel and
    move a little in every direction. The anchors stay - they are the
    identity - and everything else drifts.
    """
    def jitter(v, lo=0.0, hi=1.0):
        return max(lo, min(hi, v + rng.uniform(-amount, amount)))
    # Half-time lives on its emptiness; too many kicks and it is not
    # half-time any more.
    kick_cap = 0.22 if spec.feel == "half_time" else 0.4
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
