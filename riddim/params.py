"""Every knob of a song, in one serialisable object.

This is the contract between the engine and anything that wants to steer it
- the CLI, the UI, a saved file. Two rules make it work:

1. **Everything optional resolves.** A field left `None` is drawn by the
   engine from the song seed, and the drawn value is written back, so after
   one `compose()` the params describe the song completely. Edit a field and
   recompose, and only that field has changed.

2. **Each part has its own seed.** Drums, bass, chords, pad, lead and perc
   are each generated from their own random stream, so "regenerate the bass"
   means "new bass seed, everything else identical". That is the whole basis
   of tweaking a song rather than rerolling it.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from typing import Any

from .arrange.structure import StructureParams
from .theory.bassline import BassParams
from .theory.groove import DrumSpec

PARTS = ("drums", "perc", "sub", "chords", "pad", "lead")
# Also seeded separately, so they can be redrawn alone.
EXTRA_SEEDS = ("structure", "harmony")


@dataclass
class SectionSpec:
    """One section as the UI edits it. Mirrors arrange.structure.Section."""

    name: str
    bars: int
    energy: float
    active: list[str]
    dub_throw: bool = False
    direction: str = ""
    # This section's harmonic personality: a progression name (a library
    # name, a Roman-numeral string, or a generated name like 'i-bIII-iv').
    # None = inherit the tune's core progression.
    progression: str | None = None
    # How much this section climbs through its own span, 0..1.
    ramp: float = 0.0


@dataclass
class SongParams:
    profile: str = "dubstep_140"
    seed: int = 1
    # Global musical choices. None = draw from the seed.
    key: str | None = None            # "G phrygian"
    bpm: int | None = None
    tension: float | None = None      # harmony tension target 0..1
    bars_per_chord: float | None = None
    progression: str | None = None    # a library name, or "generate"
    swing: float | None = None
    lead_style: str | None = None     # call_response | long_tones | riff
    stab_density: float | None = None # per felt beat, 0..1
    stab_syncopation: float | None = None
    # Structured sub-parameters. None = draw.
    structure: StructureParams | None = None
    sections: list[SectionSpec] | None = None   # explicit override of structure
    drums: DrumSpec | None = None
    bass: BassParams | None = None
    # Per-part seeds. Missing entries are derived from `seed`.
    seeds: dict[str, int] = field(default_factory=dict)

    # -- seeds ---------------------------------------------------------

    def seed_for(self, part: str) -> int:
        if part not in self.seeds:
            # Derived deterministically from the song seed so an unedited
            # song is reproducible from `seed` alone.
            self.seeds[part] = random.Random(f"{self.seed}:{part}").randrange(1, 10**9)
        return self.seeds[part]

    def reseed(self, part: str, rng: random.Random | None = None) -> int:
        """Give one part a fresh seed. Everything else is untouched."""
        rng = rng or random.Random()
        self.seeds[part] = rng.randrange(1, 10**9)
        return self.seeds[part]

    # -- serialisation -------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return _encode(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SongParams":
        return _decode(cls, data)

    @classmethod
    def from_json(cls, text: str) -> "SongParams":
        return cls.from_dict(json.loads(text))

    def copy(self) -> "SongParams":
        return SongParams.from_dict(self.to_dict())


# --------------------------------------------------------------------------
# (de)serialisation that survives frozensets and nested dataclasses
# --------------------------------------------------------------------------

def _encode(value):
    if is_dataclass(value):
        return {f.name: _encode(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, (frozenset, set, tuple)):
        return [_encode(v) for v in sorted(value) if True] if isinstance(value, (frozenset, set)) else [_encode(v) for v in value]
    if isinstance(value, list):
        return [_encode(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _encode(v) for k, v in value.items()}
    return value


_FROZENSET_FIELDS = {"anchors", "avoid", "kick_anchors", "backbeat_override"}
_TUPLE_FIELDS = {"answer_bars", "velocity", "omit"}


def _decode(cls, data):
    if data is None:
        return None
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        v = data[f.name]
        if f.name == "structure":
            v = _decode(StructureParams, v)
        elif f.name == "drums":
            v = _decode(DrumSpec, v)
        elif f.name == "bass":
            v = _decode(BassParams, v)
        elif f.name == "sections":
            v = [SectionSpec(**s) for s in v] if v is not None else None
        elif f.name in _FROZENSET_FIELDS:
            v = frozenset(v) if v is not None else None
        elif f.name in _TUPLE_FIELDS:
            v = tuple(v)
        elif f.name == "seeds":
            v = {k: int(s) for k, s in v.items()}
        kwargs[f.name] = v
    return cls(**kwargs)
