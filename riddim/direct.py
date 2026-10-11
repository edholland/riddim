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
    choices: tuple | None = None
    labels: tuple | None = None

    def clamp(self, v: float) -> float:
        v = max(self.low, min(self.high, v))
        if self.kind == "int":
            v = int(round(v))
        return v

    def describe(self) -> str:
        if self.kind == "choice":
            return self.label
        return self.path


# Choice knobs: a name from a fixed menu rather than a number. Read as the
# index of the current choice (so diffs and sliders still work), written by
# index or by name.
def _choice_table() -> dict[str, tuple[str, ...]]:
    from .theory import groove, hook as hook_mod, melody, rhythm as rhythm_mod
    return {
        # The drum feel by name: one-drop, steppers, half-time, 2-step...
        "drum_feel": tuple(groove.FEELS),
        "lead_style": ("hook", "call_response", "riff", "long_tones"),
        "hook.form": tuple(hook_mod.FORMS),
        "hook.phrasing": tuple(hook_mod.PHRASINGS),
        "bass.style": ("walk", "sustain", "pulse", "bounce", "sparse", "skip"),
        "hook.rhythm": tuple(hook_mod.RHYTHM_NAMES),
        "stab_pattern": ("generated",) + tuple(
            n for table in rhythm_mod.STAB_PATTERNS.values() for n in table),
        "hook.contour": tuple(hook_mod.HOOK_CONTOURS),
    }


_CHOICES = _choice_table()

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
    "stab_density": (0.1, 1.0),
    "stab_syncopation": (0.0, 1.0),
    "bpm": (60.0, 180.0),
    "bars_per_chord": (1.0, 8.0),
    "chord_size": (3.0, 5.0),
    # The lead's performance: how the hook is played.
    "lead.octave": (-1.0, 1.0),
    "lead.gate": (0.0, 1.0),
    "lead.slide": (0.0, 1.0),
    "lead.accent": (0.0, 1.0),
    "lead.grace": (0.0, 1.0),
    "lead.flick": (0.0, 1.0),
    "lead.echo": (0.0, 1.0),
    "lead.ornament": (0.0, 1.0),
    # The hook: the theme everything else answers.
    "hook.density": (0.1, 1.0),
    "hook.syncopation": (0.0, 1.0),
    "hook.motion": (0.0, 1.0),
    "hook.repeat": (0.0, 1.0),
    "hook.leap": (2.0, 6.0),
    "hook.span": (4.0, 12.0),
    "hook.climax": (0.2, 0.95),
    "hook.repetition": (0.0, 1.0),
    "hook.colour": (0.0, 1.0),
    "hook.breath": (0.1, 0.5),
    "hook.unit_bars": (1.0, 2.0),
    "hook.candidates": (1.0, 64.0),
    # How the band talks around it.
    "interplay.presence": (0.0, 1.0),
    "interplay.answer": (0.0, 1.0),
    "interplay.lock": (0.0, 1.0),
    "interplay.counter": (0.0, 1.0),
    "interplay.space": (0.0, 1.0),
    # Tension and release.
    "dynamics.roll": (0.0, 1.0),
    "dynamics.riser": (0.0, 1.0),
    "dynamics.drop_gap": (0.0, 4.0),
    "dynamics.impact": (0.0, 1.0),
    "dynamics.switch_up": (0.0, 1.0),
    "dynamics.fills": (0.0, 1.0),
    "dynamics.cadence": (0.0, 1.0),
    "dynamics.layering": (0.0, 1.0),
    "dynamics.lift": (0.0, 1.0),
    "hook.contrast": (0.0, 1.0),
}

# Whole-number knobs: counts and registers, not percentages.
_INTS = {"bpm", "bars_per_chord", "chord_size", "lead.octave", "hook.leap",
         "hook.span", "hook.unit_bars", "hook.candidates"}
_HALVES = {"dynamics.drop_gap"}

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
    if group == "lead":
        from .theory.melody import LeadParams
        return LeadParams()
    if group == "structure":
        from .arrange.structure import StructureParams
        return StructureParams()
    if group == "hook":
        from .theory.hook import HookParams
        return HookParams()
    if group == "interplay":
        from .arrange.interplay import InterplayParams
        return InterplayParams()
    if group == "dynamics":
        from .arrange.dynamics import DynamicsParams
        return DynamicsParams()
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


# Per-section fields, steerable as 'sections.<i>.<field>'. These are a
# variable-length family, so they sit outside the fixed knob surface and the
# readback, and are compared directly in diff(). Bars snap to the phrasing:
# a section that is not a multiple of 4 fights the fill and the phrase.
_SECTION_FIELDS = {"energy": (0.05, 1.0), "ramp": (0.0, 1.0), "bars": (4.0, 64.0)}


def _section_ref(path: str):
    """The (index, field) behind a 'sections.<i>.<field>' path, or None."""
    if not path.startswith("sections."):
        return None
    bits = path.split(".")
    if len(bits) != 3 or bits[2] not in _SECTION_FIELDS:
        return None
    try:
        return int(bits[1]), bits[2]
    except ValueError:
        return None


def read_knob(params: SongParams, path: str) -> float:
    """The current value of a knob. `key` resolves to the tonic pitch class."""
    if path == "key":
        return parse_key(params.key or "G minor").root % 12
    if path in _CHOICES:
        name = choice_value(params, path)
        choices = _CHOICES[path]
        return float(choices.index(name)) if name in choices else 0.0
    sec = _section_ref(path)
    if sec is not None:
        i, field = sec
        if params.sections is None or not 0 <= i < len(params.sections):
            return 0.0
        return float(getattr(params.sections[i], field))
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
    if path in _CHOICES:
        choices = _CHOICES[path]
        if isinstance(value, str):
            if value not in choices:
                raise ValueError(f"unknown {path} {value!r}; known: {', '.join(choices)}")
            name = value
        else:
            name = choices[int(round(value)) % len(choices)]
        if path == "drum_feel":
            # A new feel is a new riddim: its anchors and kit, played at the
            # song's present density, so 'two_step' changes what is played,
            # not how much.
            from .theory import groove
            base, cur = groove.FEELS[name], params.drums
            if cur is not None and cur.feel == base.feel:
                base = replace(base, kick_density=max(base.kick_density, min(cur.kick_density, 0.4)),
                               hat_density=cur.hat_density, snare_ghosts=cur.snare_ghosts)
            params.drums = base
            return
        group, leaf = _parts(path)
        if group is None:
            setattr(params, leaf, name)
        else:
            setattr(params, group, replace(_container(params, group), **{leaf: name}))
        return
    if path in _BOUNDS:
        lo, hi = _BOUNDS[path]
        value = max(lo, min(hi, float(value)))
        if path in _INTS:
            value = int(round(value))
        elif path in _HALVES:
            value = round(value * 2) / 2
    sec = _section_ref(path)
    if sec is not None:
        i, field = sec
        if params.sections is None or not 0 <= i < len(params.sections):
            return
        lo, hi = _SECTION_FIELDS[field]
        v = max(lo, min(hi, float(value)))
        if field == "bars":
            v = max(4, int(round(v)) // 4 * 4)
        params.sections[i] = replace(params.sections[i], **{field: v})
        return
    group, leaf = _parts(path)
    if group is None:
        setattr(params, leaf, value)
        return
    obj = _container(params, group)
    setattr(params, group, replace(obj, **{leaf: value}))


def _drum_feel(params: SongParams) -> str:
    """The named feel this song's drums are a variation of: the one sharing
    its anchors, backbeat and kit idiom."""
    from .theory import groove
    d = params.drums
    if d is None:
        return _CHOICES["drum_feel"][0]
    for name, f in groove.FEELS.items():
        if (f.feel, f.kick_anchors, f.backbeat, f.idiom) == \
                (d.feel, d.kick_anchors, d.backbeat, getattr(d, "idiom", "roots")):
            return name
    return _CHOICES["drum_feel"][0]


def choice_value(params: SongParams, path: str) -> str:
    """The current name of a choice knob."""
    if path == "drum_feel":
        return _drum_feel(params)
    group, leaf = _parts(path)
    obj = params if group is None else _container(params, group)
    return getattr(obj, leaf, None) or _CHOICES[path][0]


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
    if path in _CHOICES:
        choices = _CHOICES[path]
        cur = read_knob(params, path)
        group = path.split(".")[0] if "." in path else "lead"
        return Knob(path, path.split(".")[-1], group, 0.0, len(choices) - 1, cur,
                    step=1.0, kind="choice", choices=choices, labels=choices)
    lo, hi = _BOUNDS.get(path, (None, None))
    if lo is None:
        return None
    # Counts, not percentages: whole-step sliders for what is a whole number.
    if path in _INTS:
        step, kind = 1, "int"
    elif path in _HALVES:
        step, kind = 0.5, "float"
    else:
        step, kind = 0.01, "float"
    return Knob(path, path.split(".")[-1], path.split(".")[0], lo, hi,
                read_knob(params, path), step, kind)


def knobs(params: SongParams) -> dict[str, Knob]:
    """Every discoverable knob, by path, with this song's current value.

    `key` is first: it is a knob like any other, and steering it by words
    ("down a third") must show up in the readback or the maker cannot see it.
    """
    out = {"key": knob(params, "key")}
    out.update({path: knob(params, path) for path in _BOUNDS})
    out.update({path: knob(params, path) for path in _CHOICES})
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
        # A progression belongs to a mode: the sections redraw theirs (the
        # drop's development included) in the new one.
        params.progression = None
        for spec in (params.sections or []):
            spec.progression = None
    elif op.path == "_live":
        from .compose import _make_live
        live = _make_live(params)
        params.structure, params.sections = live.structure, live.sections
    elif op.path == "_garage":
        # Garage the song up, keeping its tempo and feel: the 2-step kit
        # (its half-time cousin under a half-time snare), the skipping
        # bassline, minor-ninth organ stabs and a chopped hook.
        half = params.drums is not None and params.drums.feel == "half_time"
        set_knob(params, "drum_feel", "half_time_skip" if half else "two_step")
        set_knob(params, "bass.style", "skip")
        set_knob(params, "chord_size", 5)
        set_knob(params, "stab_pattern", "organ")
        set_knob(params, "hook.phrasing", "chop")
    elif op.path == "_total_bars":
        st = params.structure or _sub_defaults("structure")
        params.structure = replace(st, total_bars=int(to))
    elif op.path == "_peaks":
        st = params.structure or _sub_defaults("structure")
        params.structure = replace(st, peaks=int(to))
    elif op.path in _CHOICES and not to.lstrip("-+").replace(".", "", 1).isdigit():
        set_knob(params, op.path, to)
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
    "busier": [Op("drums.kick_density", 0.08), Op("bass.density", 0.06),
               Op("drums.hat_density", 0.2)],
    "denser": [Op("drums.kick_density", 0.08), Op("bass.density", 0.06),
               Op("drums.hat_density", 0.2)],
    "thinner": [Op("drums.kick_density", -0.06), Op("bass.density", -0.05),
                Op("drums.hat_density", -0.2)],
    "more_syncopated": [Op("bass.syncopation", 0.15), Op("drums.kick_syncopation", 0.1)],
    "straighter": [Op("bass.syncopation", -0.15), Op("drums.kick_syncopation", -0.1)],
    "more_approach": [Op("bass.approach", 0.2)],
    "more_octaves": [Op("bass.octave_jumps", 0.15)],
    "less_octaves": [Op("bass.octave_jumps", -0.15)],
    "fifthier": [Op("bass.fifth_weight", 0.15)],
    "shorter_notes": [Op("bass.gate", -0.15)],
    "longer_notes": [Op("bass.gate", 0.1)],
    "more_tension": [Op("tension", 0.15)],
    "less_tension": [Op("tension", -0.15)],
    "drop_energy": [Op("_sections_energy", to=-0.15)],
    "build_energy": [Op("_sections_energy", to=0.15)],
    "more_tense_progression": [Op("progression", to="i_bII")],
    "relax_progression": [Op("progression", to="tonic_drone")],
    "sevenths": [Op("chord_size", to=4)],
    "triads": [Op("chord_size", to=3)],
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
    # Lead intents
    "lead_busier": [Op("hook.density", 0.2)],
    "lead_thinner": [Op("hook.density", -0.2)],
    "lead_higher": [Op("lead.octave", 1.0)],
    "lead_lower": [Op("lead.octave", -1.0)],
    "lead_syncopated": [Op("hook.syncopation", 0.25)],
    "lead_straighter": [Op("hook.syncopation", -0.25)],
    # How the lead is played.
    "punchier": [Op("lead.gate", -0.25), Op("lead.accent", 0.25)],
    "smoother": [Op("lead.gate", 0.25), Op("lead.slide", 0.25), Op("lead.accent", -0.2)],
    "staccato": [Op("lead.gate", to=0.15)],
    "legato": [Op("lead.gate", to=0.95)],
    "slidier": [Op("lead.slide", 0.3)],
    "more_echo": [Op("lead.echo", 0.3)],
    "less_echo": [Op("lead.echo", -0.3)],
    "flicks": [Op("lead.flick", 0.3)],
    "grace_notes": [Op("lead.grace", 0.3)],
    "trills": [Op("lead.ornament", 0.3)],
    "leapier": [Op("hook.motion", 0.25)],
    "steppier": [Op("hook.motion", -0.25)],
    "more_repeats": [Op("hook.repeat", 0.25)],
    "fewer_repeats": [Op("hook.repeat", -0.25)],
    "call_lead": [Op("hook.phrasing", to="call")],
    "riff_lead": [Op("hook.phrasing", to="riff")],
    "song_lead": [Op("hook.phrasing", to="song")],
    "stab_lead": [Op("hook.phrasing", to="stabs")],
    "drone_lead": [Op("hook.phrasing", to="drone")],
    "arp": [Op("hook.phrasing", to="arp")],
    "held_bass": [Op("bass.style", to="sustain")],
    "walking_bass": [Op("bass.style", to="walk")],
    "pulsing_bass": [Op("bass.style", to="pulse")],
    "bouncing_bass": [Op("bass.style", to="bounce")],
    "sparse_bass": [Op("bass.style", to="sparse")],
    "skip_bass": [Op("bass.style", to="skip")],
    # UK garage, dubstep's other parent.
    "garage": [Op("_garage", to="1")],
    # The same song as a live track: short loops, straight in at energy.
    "live": [Op("_live", to="1")],
    "two_step": [Op("drum_feel", to="two_step")],
    "skippy_drums": [Op("drum_feel", to="two_step_skip")],
    "speed_garage": [Op("drum_feel", to="garage_4x4")],
    "one_drop": [Op("drum_feel", to="one_drop")],
    "steppers": [Op("drum_feel", to="steppers")],
    "half_time": [Op("drum_feel", to="half_time")],
    "ninths": [Op("chord_size", to=5)],
    "organ_stabs": [Op("stab_pattern", to="organ")],
    "chop_lead": [Op("hook.phrasing", to="chop")],
    "dotted": [Op("hook.rhythm", to="dotted")],
    "stabs": [Op("hook.rhythm", to="stabs")],
    "pickup": [Op("hook.rhythm", to="pickup")],
    "triplet": [Op("hook.rhythm", to="triplet")],
    "offbeat": [Op("hook.rhythm", to="offbeat")],
    "flowing": [Op("hook.rhythm", to="flowing")],
    # The hook. "Catchier" is more literal repetition, more auditions and a
    # longer breath - the three things that make a phrase stick.
    "catchier": [Op("hook.repetition", 0.15), Op("hook.repeat", 0.1),
                 Op("hook.candidates", 16), Op("hook.breath", 0.04)],
    "wider_hook": [Op("hook.span", 2), Op("hook.motion", 0.15)],
    "smoother_hook": [Op("hook.motion", -0.25), Op("hook.span", -1)],
    "busier_hook": [Op("hook.density", 0.12)],
    "sparser_hook": [Op("hook.density", -0.12)],
    "syncopated_hook": [Op("hook.syncopation", 0.2)],
    "straighter_hook": [Op("hook.syncopation", -0.2)],
    "more_colour": [Op("hook.colour", 0.25)],
    "less_colour": [Op("hook.colour", -0.25)],
    "sentence": [Op("hook.form", to="sentence")],
    "period": [Op("hook.form", to="period")],
    "riff_form": [Op("hook.form", to="riff")],
    "call_answer": [Op("hook.form", to="call_answer")],
    "sing": [Op("lead_style", to="hook")],
    "riff": [Op("lead_style", to="riff")],
    "call_response": [Op("lead_style", to="call_response")],
    "long_tones": [Op("lead_style", to="long_tones")],
    # The conversation.
    "more_hook": [Op("interplay.presence", 0.2)],
    "less_hook": [Op("interplay.presence", -0.2)],
    "chattier": [Op("interplay.answer", 0.2)],
    "quieter_band": [Op("interplay.answer", -0.2)],
    "tighter": [Op("interplay.lock", 0.2)],
    "looser": [Op("interplay.lock", -0.2)],
    "more_counter": [Op("interplay.counter", 0.25)],
    "less_counter": [Op("interplay.counter", -0.25)],
    "more_space": [Op("interplay.space", 0.2)],
    # Tension and release.
    "bigger_drop": [Op("dynamics.drop_gap", 1.0), Op("dynamics.impact", 0.2),
                    Op("dynamics.roll", 0.2)],
    "smaller_drop": [Op("dynamics.drop_gap", -1.0), Op("dynamics.impact", -0.2)],
    "more_build": [Op("dynamics.roll", 0.2), Op("dynamics.riser", 0.25)],
    "less_build": [Op("dynamics.roll", -0.2), Op("dynamics.riser", -0.25)],
    "key_lift": [Op("dynamics.lift", to=1.0)],
    "no_lift": [Op("dynamics.lift", to=0.0)],
    "more_breaks": [Op("dynamics.switch_up", 0.25)],
    "fewer_breaks": [Op("dynamics.switch_up", -0.25)],
    "more_fills": [Op("dynamics.fills", 0.2)],
    "fewer_fills": [Op("dynamics.fills", -0.2)],
}

# Mutation: a random twist across many knobs at once, seeds still frozen. The
# song moves somewhere it has never been while staying itself - same hook
# seed, same bass seed, new settings. `amount` scales every move.
MUTATIONS = {
    "mutate": (("drums", "bass", "lead", "hook", "interplay", "dynamics", "chords"), 0.5),
    "mutate_hook": (("hook",), 0.6),
    "mutate_groove": (("drums", "bass", "chords"), 0.5),
    "mutate_arrangement": (("interplay", "dynamics"), 0.6),
    "mutate_lead": (("lead", "hook"), 0.5),
    "wild": (("drums", "bass", "lead", "hook", "interplay", "dynamics", "chords",
              "harmony"), 1.0),
}

_KNOB_GROUP = {"stab_density": "chords", "stab_syncopation": "chords",
               "tension": "harmony", "bars_per_chord": "harmony", "chord_size": "harmony",
               "lead_style": "lead", "stab_pattern": "chords", "drum_feel": "drums"}


def mutation(params: SongParams, groups, amount: float = 0.5, rng=None) -> list[Op]:
    """A random walk step over the knobs in `groups`, as ops.

    Each float knob moves with probability ~40% by a Gaussian step scaled to
    its range and `amount`; each choice knob flips with probability
    `amount * 0.3`. The result is a list of absolute ops, so it lands on the
    trail with a readback like any other twist and can be undone.
    """
    import random as _random
    rng = rng or _random.Random()
    ops: list[Op] = []
    for path, (lo, hi) in _BOUNDS.items():
        group = _KNOB_GROUP.get(path, path.split(".")[0])
        if group not in groups or path in ("bpm", "hook.candidates", "hook.unit_bars"):
            continue
        if rng.random() > 0.4:
            continue
        cur = read_knob(params, path)
        nxt = cur + rng.gauss(0.0, 0.22) * amount * (hi - lo)
        nxt = max(lo, min(hi, nxt))
        if path in _INTS:
            nxt = int(round(nxt))
        if abs(nxt - cur) > 1e-6:
            ops.append(Op(path, to=round(nxt, 3)))
    for path, choices in _CHOICES.items():
        group = _KNOB_GROUP.get(path, path.split(".")[0])
        if group not in groups or rng.random() > amount * 0.3:
            continue
        cur = choice_value(params, path)
        others = [c for c in choices if c != cur]
        if path == "drum_feel":
            # A mutation stays in the song's meter: half-time to half-time.
            from .theory import groove
            want = params.drums.feel if params.drums is not None else None
            others = [c for c in others if groove.FEELS[c].feel == want]
        if others:
            ops.append(Op(path, to=rng.choice(others)))
    return ops

INTENT_ALIASES = {
    "up": "brighter", "down": "darker",
    "busy": "busier", "dense": "denser", "thin": "thinner",
    "sync": "more_syncopated", "syncopate": "more_syncopated",
    "straighten": "straighter",
    "tense": "more_tension", "relax": "less_tension",
    "bright": "brighter", "dark": "darker",
    "phrygian dominant": "phrygian_dominant",
    "more phrygian": "more_phrygian",
    "hookier": "catchier", "catchy": "catchier", "hooky": "catchier",
    "bigger": "bigger_drop", "harder": "bigger_drop", "drop harder": "bigger_drop",
    "build up": "more_build", "more tension build": "more_build",
    "conversation": "chattier", "answer": "chattier", "talk": "chattier",
    "lock": "tighter", "unison": "tighter", "counter": "more_counter",
    "breaks": "more_breaks", "switch up": "more_breaks", "fills": "more_fills",
    "mutate everything": "mutate", "mutate hook": "mutate_hook",
    "mutate groove": "mutate_groove", "mutate arrangement": "mutate_arrangement",
    "mutate lead": "mutate_lead", "go wild": "wild",
    "singing": "song_lead", "sing it": "song_lead", "stabby": "stab_lead",
    "stabby lead": "stab_lead", "drone": "drone_lead", "arpeggio": "arp",
    "arpeggiate": "arp", "ostinato": "riff_lead",
    "live set": "live", "live track": "live", "for the stage": "live",
    "ukg": "garage", "uk garage": "garage", "garage it": "garage", "garagey": "garage",
    "more garage": "garage", "2 step": "two_step", "2step": "two_step",
    "skippy": "skippy_drums", "4x4": "speed_garage", "four to the floor": "speed_garage",
    "skippy bass": "skip_bass", "2 step bass": "skip_bass", "garage bass": "skip_bass",
    "ninth": "ninths", "9ths": "ninths", "minor nines": "ninths", "rhodes": "ninths",
    "organ": "organ_stabs", "vocal chop": "chop_lead", "chopped": "chop_lead",
    "chop": "chop_lead", "chops": "chop_lead", "one drop": "one_drop",
    "halftime": "half_time", "half time": "half_time",
}


def _pitch_semitones(text: str) -> int | None:
    return _pitch_from_phrase(text)





def direct(text: str, params: SongParams | None = None, rng=None) -> list[Op]:
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
    key = key.replace(" ", "_")
    if key in INTENT_ALIASES:
        key = INTENT_ALIASES[key]
    if key in INTENTS:
        return list(INTENTS[key])
    if key in MUTATIONS:
        if params is None:
            raise ValueError(f"{raw!r} needs a song to mutate")
        groups, amount = MUTATIONS[key]
        return mutation(params, groups, amount, rng)
    raise ValueError(
        f"did not understand {raw!r}; try one of: "
        + ", ".join(sorted(set(INTENTS) | set(INTENT_ALIASES) | set(MUTATIONS))))


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
    # Section fields are steerable too, but the section list is variable-length,
    # so they are compared directly rather than through the fixed knob surface.
    # (progression is a name, not a number, and the section editor shows it.)
    if a.sections is not None and b.sections is not None:
        for i, (sa, sb) in enumerate(zip(a.sections, b.sections)):
            for f in _SECTION_FIELDS:
                va, vb = float(getattr(sa, f)), float(getattr(sb, f))
                if abs(va - vb) > 1e-6:
                    out.append(Delta(f"sections.{i}.{f}", round(va, 3), round(vb, 3)))
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
        """Steer by a direction in words: 'busier', 'down a third', 'mutate'..."""
        return self.step(direct(text, self.params), label=label or text)

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
