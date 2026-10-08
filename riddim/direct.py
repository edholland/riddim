"""Steer a song by direction, not by rerolling.

The engine already makes two kinds of change: `compose` builds a whole song, and
`regenerate` gives one part a new seed - an unrelated re-take. Neither of those
is what a maker is doing when a track has an identity and they want to move it:
make the drop denser, take the key down, open the filter, pull a section back.
That is direction, and it is what this module adds.

The trick is that `SongParams` records the whole song completely and keeps
every seed frozen. So a directed change is just: nudge a knob, keep the seeds,
recompose. The result is related to the original, a step along a line from it,
not a new point in the space. `mutate` is this; a fresh roll is the opposite.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .compose import compose
from .params import SongParams
from .profile import PROFILES, ALIASES
from .theory import pitch
from .theory.scale import parse_key


# --------------------------------------------------------------------------
# Knobs: a discoverable view over the knobs of a composed song
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Knob:
    """One adjustable quantity, by path and bound, for the UI and for ops."""
    path: str
    label: str
    group: str
    low: float
    high: float
    value: float
    step: float = 0.01
    kind: str = "float"

    def clamp(self, v: float) -> float:
        v = max(self.low, min(self.high, v))
        if self.kind == "int":
            v = int(round(v))
        return v


_BOUNDS = {
    "drums.kick_density": (0.0, 0.5),
    "drums.kick_syncopation": (0.0, 1.0),
    "drums.snare_ghosts": (0.0, 0.4),
    "drums.hat_density": (0.0, 0.8),
    "drums.hat_offbeat_bias": (-0.5, 1.0),
    "drums.rim_density": (0.0, 0.4),
    "bass.density": (0.05, 0.6),
    "bass.syncopation": (0.0, 1.0),
    "bass.approach": (0.0, 1.0),
    "bass.octave_jumps": (0.0, 0.5),
    "bass.fifth_weight": (0.0, 1.0),
    "bass.gate": (0.3, 1.0),
    "tension": (0.0, 1.0),
    "swing": (0.0, 0.33),
    "stab_density": (0.1, 1.0),
    "stab_syncopation": (0.0, 1.0),
    "bpm": (60.0, 180.0),
    "bars_per_chord": (1.0, 8.0),
}

_DUMPS = ("kick_density", "kick_syncopation", "snare_ghosts",
        "hat_density", "hat_offbeat_bias", "rim_density")
_BASES = ("density", "syncopation", "approach", "octave_jumps", "fifth_weight",
        "gate", "max_hold")


def _sub_defaults(group: str):
    if group == "drums":
        from .theory.groove import DrumSpec
        return DrumSpec()
    if group == "bass":
        from .theory.bassline import BassParams
        return BassParams()
    if group == "structure":
        from .arrange.structure import StructureParams
        return StructureParams()
    return None


def _parts(path: str):
    if "." in path:
        group, leaf = path.split(".", 1)
        return group, leaf
    if path in _DUMPS:
        return "drums", path
    if path in _BASES:
        return "bass", path
    return None, path


def _container(params, group):
    """The sub-spec object to read/write, a default if the song left it unset."""
    if group is None:
        return params
    obj = getattr(params, group, None)
    if obj is None:
        obj = _sub_defaults(group)
        setattr(params, group, obj)
    return obj


def read_knob(params: SongParams, path: str) -> float:
    """The current value of a knob. `key` resolves to the tonic pitch class."""
    if path == "key":
        return parse_key(params.key or "G minor").root % 12
    group, leaf = _parts(path)
    obj = _container(params, group)
    val = getattr(obj, leaf, 0.0)
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def set_knob(params: SongParams, path: str, value: float) -> None:
    """Write a knob. Sub-spec dataclasses are frozen, so they are replaced,
    not mutated; `key` transposes the tonic while keeping the mode.
    """
    if path == "key":
        _set_key_root(params, int(round(value)))
        return
    from dataclasses import replace
    group, leaf = _parts(path)
    if group is None:
        setattr(params, leaf, value)
        return
    obj = _container(params, group)
    setattr(params, group, replace(obj, **{leaf: value}))


def _set_key_root(params: SongParams, root_pc: int) -> None:
    """Move the tonic to a pitch class, keeping the mode and a sane octave."""
    scale = parse_key(params.key or "G minor")
    names = pitch.FLAT_NAMES if scale.prefers_flats else pitch.SHARP_NAMES
    within = root_pc % 12
    target = pitch.transpose_into(
        scale.root + (within - scale.root % 12),
        scale.root - 12, scale.root + 12,
    )
    params.key = f"{names[target % 12]} {scale.name}"


def _structure(params: SongParams):
    from .arrange.structure import StructureParams
    if params.sections is not None:
        return params.structure or StructureParams()
    if params.structure is None:
        params.structure = StructureParams()
    return params.structure


def knob(params: SongParams, path: str) -> Knob | None:
    """Describe one knob for this song, or None if the path is not a knob."""
    if path == "key":
        root = parse_key(params.key or "G minor").root % 12
        return Knob("key", "key", "global", 0, 11, root, 1, "key")
    lo, hi = _BOUNDS.get(path, (None, None))
    if lo is None:
        return None
    return Knob(path, path.split(".")[-1], path.split(".")[0], lo, hi,
                read_knob(params, path))


def knobs(params: SongParams) -> dict[str, Knob]:
    """Every discoverable knob, by path, with this song's current value.

    `key` is first: it is a knob like any other, and steering it by words
    ("down a third") must show up in the readback or the maker cannot see it.
    """
    out = {"key": knob(params, "key")}
    out.update({path: knob(params, path) for path in _BOUNDS})
    return out


# --------------------------------------------------------------------------
# A knob change: a directed step, applied to a copy of the params
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Op:
    """One directed change. `delta` is relative; `to` is an absolute target."""
    path: str
    delta: float = 0.0
    to: str | None = None

    def apply(self, params: SongParams) -> None:
        if self.to is not None:
            _apply_absolute(params, self)
            return
        cur = read_knob(params, self.path)
        set_knob(params, self.path, cur + self.delta)

    def describe(self) -> str:
        if self.to is not None:
            return f"{self.path} = {self.to}"
        sign = "+" if self.delta >= 0 else ""
        return f"{self.path} {sign}{self.delta:g}"


def _apply_absolute(params: SongParams, op: Op) -> None:
    """Absolute targets: key, profile, progression, bar count, peaks, energy."""
    from dataclasses import replace
    to = str(op.to)
    if op.path == "key":
        # `to` is either a numeric relative shift in semitones, or a key name
        # like 'C phrygian' for an absolute target.
        if str(op.to).lstrip("-+").isdigit():
            root = (read_knob(params, "key") + int(op.to)) % 12
        else:
            root = parse_key(str(op.to)).root % 12
        _set_key_root(params, root)
    elif op.path == "profile":
        params.profile = ALIASES.get(to.lower(), to)
    elif op.path == "progression":
        params.progression = to
        params.tension = None
    elif op.path == "mode":
        # Same tonic, new mode: 'G phrygian' in, 'G dorian' out. The b2, the
        # natural 6th, the raised 7th - the mode is where the menace lives.
        scale = parse_key(params.key or "G minor")
        names = pitch.FLAT_NAMES if scale.prefers_flats else pitch.SHARP_NAMES
        if to not in ("minor", "dorian", "phrygian", "harmonic_minor",
                      "phrygian_dominant"):
            raise ValueError(
                f"unknown mode {to!r}; known: minor, dorian, phrygian, "
                f"harmonic_minor, phrygian_dominant")
        params.key = f"{names[scale.root_pc]} {to}"
    elif op.path == "_total_bars":
        st = params.structure or _sub_defaults("structure")
        params.structure = replace(st, total_bars=int(to))
    elif op.path == "_peaks":
        st = params.structure or _sub_defaults("structure")
        params.structure = replace(st, peaks=int(to))
    elif op.path == "_sections_energy":
        from .params import SectionSpec
        new_sections = []
        for spec in (params.sections or []):
            new_sections.append(replace(spec, energy=max(0.05, min(1.0, spec.energy + float(to)))))
        params.sections = new_sections
    else:
        set_knob(params, op.path, float(to))


def apply_ops(params: SongParams, ops: list[Op]) -> SongParams:
    """Apply a list of ops to a fresh copy, leaving the caller's params intact."""
    next_params = params.copy()
    for op in ops:
        op.apply(next_params)
    return next_params


# --------------------------------------------------------------------------
# Intent: a direction in words, resolved to a list of ops
# --------------------------------------------------------------------------

# A small musical vocabulary: each direction is a directed step on one or more
# knobs. Deliberately few, but enough to steer a track by feeling.
INTENTS = {
    "brighter": [Op("stab_density", 0.12), Op("drums.hat_density", 0.08)],
    "brighten": [Op("stab_density", 0.12), Op("drums.hat_density", 0.08)],
    "darker": [Op("stab_density", -0.12), Op("drums.hat_density", -0.06)],
    "busier": [Op("drums.kick_density", 0.08), Op("bass.density", 0.06)],
    "denser": [Op("drums.kick_density", 0.08), Op("bass.density", 0.06)],
    "thinner": [Op("drums.kick_density", -0.06), Op("bass.density", -0.05)],
    "more_syncopated": [Op("bass.syncopation", 0.15), Op("drums.kick_syncopation", 0.1)],
    "straighter": [Op("bass.syncopation", -0.15), Op("drums.kick_syncopation", -0.1)],
    "more_approach": [Op("bass.approach", 0.2)],
    "more_octaves": [Op("bass.octave_jumps", 0.15)],
    "less_octaves": [Op("bass.octave_jumps", -0.15)],
    "fifthier": [Op("bass.fifth_weight", 0.15)],
    "shorter_notes": [Op("bass.gate", -0.15)],
    "longer_notes": [Op("bass.gate", 0.1)],
    "swingier": [Op("swing", 0.06)],
    "straight_grid": [Op("swing", -0.06)],
    "more_tension": [Op("tension", 0.15)],
    "less_tension": [Op("tension", -0.15)],
    "drop_energy": [Op("_sections_energy", to=-0.15)],
    "build_energy": [Op("_sections_energy", to=0.15)],
    "more_tense_progression": [Op("progression", to="i_bII")],
    "relax_progression": [Op("progression", to="tonic_drone")],
    "faster": [Op("bpm", 2)],
    "slower": [Op("bpm", -2)],
    # A mode is the harmonic personality; the tonic stays put. 'phrygian'
    # alone means 'make it phrygian', which moves the chords, bass colour and
    # lead onto the new mode - a big move, on purpose.
    "minor": [Op("mode", to="minor")],
    "dorian": [Op("mode", to="dorian")],
    "phrygian": [Op("mode", to="phrygian")],
    "phrygian_dominant": [Op("mode", to="phrygian_dominant")],
    "more_phrygian": [Op("mode", to="phrygian")],
}

INTENT_ALIASES = {
    "up": "brighter", "down": "darker",
    "busy": "busier", "dense": "denser", "thin": "thinner",
    "sync": "more_syncopated", "syncopate": "more_syncopated",
    "straighten": "straighter", "swing": "swingier",
    "tense": "more_tension", "relax": "less_tension",
    "bright": "brighter", "dark": "darker",
    "phrygian dominant": "phrygian_dominant",
    "more phrygian": "more_phrygian",
}


def _pitch_semitones(text: str) -> int | None:
    return _pitch_from_phrase(text)





def direct(text: str) -> list[Op]:
    """Resolve a direction in words to a list of ops.

    A single word or alias looks up `INTENTS`. A phrase like 'down a third',
    'up an octave', 'faster by 4', 'shorter by 16 bars' expands to a precise
    op. Everything else raises, so the UI can say 'did not understand' rather
    than silently do nothing.
    """
    raw = text.strip()
    t = " ".join(raw.lower().replace("-", " ").split())

    # "up a <pitch>" / "down a <pitch>": move the key by a named interval.
    for verb, d in (("up", 1), ("down", -1)):
        for pre in (verb + " a ", verb + " an "):
            if t.startswith(pre):
                semis = _pitch_from_phrase(t[len(pre):])
                if semis is not None:
                    return [Op("key", to=d * semis)]

    # "faster by 4", "slower by 4", "shorter by 16", "longer by 16".
    # 'by N' means N written bpm or N bars, in the direction of the verb;
    # a bare verb is one default step (+/-2 bpm, +/-16 bars).
    for head in ("faster", "slower", "shorter", "longer"):
        if not t.startswith(head):
            continue
        m = t.find(" by ")
        n = _number(t[m + 4:]) if m >= 0 else None
        sign = 1 if head in ("faster", "longer") else -1
        if head in ("faster", "slower"):
            return [Op("bpm", sign * (n if n is not None else 2))]
        return [Op("_total_bars", sign * (n if n is not None else 16))]

    key = t
    if key in INTENT_ALIASES:
        key = INTENT_ALIASES[key]
    if key in INTENTS:
        return list(INTENTS[key])
    raise ValueError(
        f"did not understand {raw!r}; try one of: "
        + ", ".join(sorted(set(INTENTS) | set(INTENT_ALIASES))))


def _pitch_from_phrase(rest: str) -> int | None:
    """The named pitch interval in a phrase, in semitones, or None."""
    words = " ".join(rest.lower().split())
    table = [
        ("octave", 7), ("octaves", 7),
        ("seventh", 6), ("7th", 6),
        ("sixth", 5), ("6th", 5),
        ("fifth", 4), ("5th", 4),
        ("fourth", 3), ("4th", 3),
        ("third", 2), ("3rd", 2),
        ("second", 1), ("2nd", 1), ("tone", 1),
    ]
    for name, n in table:
        if name in words:
            return n
    return None


def _number(text: str) -> int | None:
    """First run of digits in a string, else None. Used for 'faster by 4'."""
    digits = ""
    for c in text:
        if c.isdigit():
            digits += c
        elif digits:
            break
    return int(digits) if digits else None


# --------------------------------------------------------------------------
# Feedback: what moved between two states
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Delta:
    """One knob that changed between two param states, and by how much."""
    path: str
    from_value: float
    to_value: float


def diff(a: SongParams, b: SongParams) -> list[Delta]:
    """The knobs that differ between two param states, ordered by size of move.

    This is the readback that makes a change legible: 'I said busier, and here
    is exactly which knobs moved' - the difference between a change you can see
    and one you cannot.
    """
    out: list[Delta] = []
    for path in knobs(a):
        va, vb = read_knob(a, path), read_knob(b, path)
        if abs(va - vb) > 1e-6:
            out.append(Delta(path, round(va, 3), round(vb, 3)))
    out.sort(key=lambda d: -abs(d.to_value - d.from_value))
    return out


@dataclass(frozen=True)
class SongDiff:
    """The audible difference between two composed songs.

    `moved_parts` is roles whose notes differ in any field, so a pure key
    change - same note counts, every pitch shifted - still reads as moved.
    """
    deltas: list[Delta] = field(default_factory=list)
    total_notes: int = 0
    delta_notes: int = 0
    moved_parts: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if not self.deltas:
            return "only a seed moved; no knob changed"
        moved = ", ".join(self.moved_parts) if self.moved_parts else "nothing"
        head = "; ".join(f"{d.path} {d.from_value:.2f}->{d.to_value:.2f}"
                        for d in self.deltas[:5])
        return f"moved {moved} ({head})"


def _part_notes(song) -> dict[str, tuple]:
    return {p.role: tuple(p.sorted_notes()) for p in song.parts}


def song_diff(baseline: object, trial: object) -> SongDiff:
    """Compare two composed songs: knobs that moved and which parts changed.

    `moved_parts` lists the roles whose *notes* differ in any field - pitch,
    time, length or velocity - not just their count. Count alone would report
    'moved nothing' for a key change, which transposes every pitched note and
    leaves every part with the same number of them. The normal case of a nudge
    is still that only the intended part moves: 'the bass got busier, the
    chords did not touch'.
    """
    deltas = diff(baseline.params, trial.params)
    base = _part_notes(baseline)
    trial_parts = _part_notes(trial)
    moved = sorted(
        r for r in set(base) | set(trial_parts)
        if base.get(r, ()) != trial_parts.get(r, ())
    )
    total = sum(len(n) for n in trial_parts.values())
    d_notes = total - sum(len(n) for n in base.values())
    return SongDiff(deltas, total, d_notes, moved)


# --------------------------------------------------------------------------
# The trail: a stateful steer, with A/B readback and undo
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Step:
    """One point on the steer.

    `delta` is the move from the previous step; `delta_from_base` is the total
    move since the baseline. `song_diff` answers 'what actually changed between
    the two songs' - the part-by-part readback.
    """
    label: str
    ops: tuple[Op, ...]
    params: SongParams
    song: object
    delta: tuple[Delta, ...] = ()
    delta_from_base: tuple[Delta, ...] = ()
    song_diff: SongDiff | None = None


@dataclass
class Session:
    """A song being steered, with its trail.

    The trail is a line of `Step`s. Step 0 is the baseline. Each `step()` /
    `attempt()` nudges the current step's params, keeps every seed frozen, and
    recomposes - so each point is a small move *from the one before*, not a new
    roll. `undo()` walks back; `compare()` runs the readback that tells you the
    nudge did what you meant. Because the seeds never move, two nearby steps
    differ only where the knobs told them to.
    """
    rig: object = None
    steps: list[Step] = field(default_factory=list)

    def __init__(self, song, rig=None):
        self.rig = rig or _rig_of(song)
        base = _ensure_resolved(song)
        self.steps = [Step(
            "baseline", (), base.params, base,
            delta=(), delta_from_base=(), song_diff=None,
        )]

    # -- where we are ----------------------------------------------------

    @property
    def current(self) -> Step:
        return self.steps[-1]

    @property
    def song(self):
        return self.current.song

    @property
    def params(self) -> SongParams:
        return self.current.params

    @property
    def baseline(self) -> Step:
        return self.steps[0]

    # -- the walk --------------------------------------------------------

    def step(self, spec, label: str = "") -> Step:
        """Nudge the current step by `spec` (ops or an intent) and recompose."""
        ops = tuple(_as_ops(spec))
        parent = self.current
        new_params = apply_ops(parent.params, ops)
        trial = compose(new_params, rig=self.rig)
        diff_parent = diff(parent.params, new_params)
        step = Step(
            label or _label(ops),
            ops, new_params, trial,
            delta=tuple(diff_parent),
            delta_from_base=tuple(diff(self.baseline.params, new_params)),
            song_diff=song_diff(parent.song, trial),
        )
        self.steps.append(step)
        return step

    def attempt(self, text: str, label: str = "") -> Step:
        """Steer by a direction in words: 'busier', 'down a third', 'drop'..."""
        return self.step(direct(text), label=label or text)

    # -- revision control ------------------------------------------------

    def undo(self) -> Step:
        if len(self.steps) < 2:
            return self.current
        self.steps.pop()
        return self.current

    def revert_to(self, index: int) -> Step:
        if not 0 <= index < len(self.steps) - 1:
            return self.current
        del self.steps[index + 1:]
        return self.current

    # -- readback --------------------------------------------------------

    def compare(self, other=None) -> SongDiff:
        """How the current song differs from the baseline (or a given song)."""
        if other is None:
            other = self.baseline.song
        return song_diff(self.current.song, other)

    def report(self) -> str:
        """A human-readable transcript of the steer so far."""
        lines = [f"baseline: {self.baseline.song.summary().splitlines()[0]}"]
        for s in self.steps[1:]:
            lines.append(f"\n-- {s.label} --")
            for d in s.delta:
                lines.append(f"     {d.path:<18} {d.from_value:6.2f} -> {d.to_value:6.2f}")
            if s.song_diff and s.song_diff.moved_parts:
                lines.append("     notes moved: " + ", ".join(s.song_diff.moved_parts))
        return "\n".join(lines)


def _ensure_resolved(song):
    """Return a song whose params have the drum/bass/structure specs filled in.

    A freshly composed song already has this (compose writes the drawn values
    back). A hand-built song might not, so we resolve it once, which is the only
    time the engine's own generators run here.
    """
    from .song import Song
    if song.params is not None and song.params.drums is not None:
        return song
    params = song.params or _blank_params()
    return compose(params, rig=_rig_of(song))


def _blank_params() -> SongParams:
    return SongParams(profile="dubstep_140", seed=1, key="D phrygian")


def _rig_of(song):
    """Recover the rig a song was built with, defaulting to the rack rig."""
    from .song import Rig, ALL_DELUGE
    roles_with_output = [p for p in song.parts if p.role in
                        ("sub", "chords", "lead", "pad")]
    if roles_with_output and all(p.output == "synth" for p in roles_with_output):
        return ALL_DELUGE
    return Rig()


def _as_ops(spec):
    """Accept an Op, a list of Op, a (path, delta) pair, or an intent string."""
    if isinstance(spec, Op):
        return [spec]
    if isinstance(spec, str):
        return direct(spec)
    out = []
    for item in spec:
        if isinstance(item, Op):
            out.append(item)
        elif isinstance(item, str):
            out.extend(direct(item))
        else:
            out.append(Op(*item))
    return out


def _label(ops: tuple[Op, ...]) -> str:
    return ", ".join(o.describe() for o in ops)


def mutate(song, spec, rig=None) -> object:
    """One-shot: nudge a song by ops or an intent and return the new song.

    For a steer you keep walking, build a `Session` instead, so the trail and its
    readback are kept.
    """
    ops = _as_ops(spec)
    new_params = apply_ops(_ensure_resolved(song).params, ops)
    return compose(new_params, rig=(rig or _rig_of(song)))
