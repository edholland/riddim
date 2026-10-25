"""Steering: a few big, plainly named moves over many knobs at once.

The knobs are honest but there are sixty of them; a maker wants to say
"more energy", "heavier", "dubbier" and hear it. A macro is a set of knobs
each pushed one way (w=+1 toward its musical top, -1 toward its bottom).

Macros are relative to the take: 0 is the song as it was rolled, +1 pushes
every knob it owns all the way to its musical bound in its direction, -1 the
other way. Nothing is stored - a macro's position is read back from where its
knobs sit against the take's start, so undo, reset and hand twists all show
up on it. Moving a macro adds the difference to each knob, so a hand twist on
a shared knob is kept rather than overwritten.
"""

from __future__ import annotations

from dataclasses import dataclass

from .direct import Op, knobs as all_knobs, read_knob
from .params import SongParams


@dataclass(frozen=True)
class Macro:
    name: str
    low: str        # what -1 sounds like
    high: str       # what +1 sounds like
    help: str
    knobs: tuple    # (path, weight, musical low, musical high) - bounds None = the knob's own


MACROS: tuple[Macro, ...] = (
    Macro("energy", "laid back", "driving",
          "How hard the band pushes: busier hats and kicks, more ghost snares, "
          "more stabs, fills, rolls and crashes.",
          (("drums.hat_density", 1, 0.15, 0.8), ("drums.kick_density", 1, 0.0, 0.35),
           ("drums.snare_ghosts", 1, 0.0, 0.25), ("stab_density", 1, 0.3, 1.0),
           ("bass.density", 1, 0.1, 0.45), ("dynamics.fills", 1, 0.0, 0.8),
           ("dynamics.roll", 1, 0.0, 1.0), ("dynamics.impact", 1, 0.1, 0.9))),
    Macro("weight", "light", "heavy",
          "Sub pressure: long held sub notes on the root, fewer jumps, the bass "
          "pulled and slammed back, silence before the drops.",
          (("bass.gate", 1, 0.4, 1.0), ("bass.octave_jumps", -1, 0.0, 0.4),
           ("bass.approach", -1, 0.0, 0.8), ("bass.fifth_weight", -1, 0.0, 0.6),
           ("dynamics.bass_drops", 1, 0.0, 2.5), ("dynamics.drop_gap", 1, 0.0, 3.0))),
    Macro("space", "packed", "dubby",
          "Room in the mix: the band steps aside, the hook breathes, echoes "
          "fall into the gaps, mid-drop stops.",
          (("lead.echo", 1, 0.0, 0.8), ("interplay.space", 1, 0.2, 1.0),
           ("hook.breath", 1, 0.15, 0.5), ("stab_density", -1, 0.3, 1.0),
           ("drums.hat_density", -1, 0.15, 0.8), ("hook.density", -1, 0.2, 0.9),
           ("interplay.answer", -1, 0.2, 1.0), ("dynamics.switch_up", 1, 0.0, 0.6))),
    Macro("hook", "subtle", "catchy",
          "How memorable the tune is: fewer notes, a narrower range, steps not "
          "leaps, phrases coming back, more of the drop given to it.",
          (("hook.repetition", 1, 0.3, 1.0), ("hook.repeat", 1, 0.0, 0.7),
           ("hook.density", -1, 0.2, 0.9), ("hook.span", -1, 4, 10),
           ("hook.motion", -1, 0.0, 0.7), ("hook.outside", -1, 0.0, 0.5),
           ("interplay.presence", 1, 0.3, 1.0))),
    Macro("groove", "on the grid", "pushed",
          "Where the notes sit: on the beat, or pushed off it - kicks, bass, "
          "stabs, hook and the hat accents all together.",
          (("drums.kick_syncopation", 1, 0.0, 1.0), ("bass.syncopation", 1, 0.0, 1.0),
           ("stab_syncopation", 1, 0.0, 1.0), ("hook.syncopation", 1, 0.0, 0.9),
           ("drums.hat_offbeat_bias", 1, -0.3, 1.0))),
    Macro("mood", "dark", "bright",
          "Colour: calm-to-tense harmony, plain triads to garage ninths, the "
          "lead's register, and a key lift into the last drop.",
          (("tension", -1, 0.0, 0.8), ("chord_size", 1, 3, 5),
           ("lead.octave", 1, -1, 1), ("dynamics.lift", 1, 0.0, 0.9))),
    Macro("flair", "plain", "ornate",
          "How the lead is played: straight, or with slides, grace notes, "
          "octave flicks, trills and hard accents.",
          (("lead.slide", 1, 0.0, 0.6), ("lead.grace", 1, 0.0, 0.5),
           ("lead.flick", 1, 0.0, 0.4), ("lead.ornament", 1, 0.0, 0.4),
           ("lead.accent", 1, 0.2, 0.9))),
)
BY_NAME = {m.name: m for m in MACROS}


def _bounds(entry, ks) -> tuple[float, float] | None:
    path, _w, lo, hi = entry
    k = ks.get(path)
    if k is None or k.kind not in ("float", "int"):
        return None
    lo = k.low if lo is None else max(k.low, lo)
    hi = k.high if hi is None else min(k.high, hi)
    return (lo, hi) if hi > lo else None


def _offset(m: float, w: int, base: float, lo: float, hi: float) -> float:
    """Where position m puts a knob relative to its take value."""
    s = m * w
    base = min(max(base, lo), hi)
    return s * (hi - base) if s >= 0 else s * (base - lo)


def _position(macro: Macro, base: SongParams, now: SongParams, ks) -> float:
    """Read a macro back from its knobs: the mean of each knob's own position."""
    pos = []
    for entry in macro.knobs:
        b = _bounds(entry, ks)
        if b is None:
            continue
        path, w, *_ = entry
        lo, hi = b
        b0 = min(max(read_knob(base, path), lo), hi)
        d = read_knob(now, path) - b0
        room = (hi - b0) if d >= 0 else (b0 - lo)
        if room <= 1e-9:
            continue
        s_ = d / room
        pos.append(max(-1.0, min(1.0, s_ * w)))
    return sum(pos) / len(pos) if pos else 0.0


def positions(base: SongParams, now: SongParams) -> list[dict]:
    ks = all_knobs(now)
    return [{"name": m.name, "low": m.low, "high": m.high, "help": m.help,
             "value": round(_position(m, base, now, ks), 3),
             "knobs": [e[0] for e in m.knobs if e[0] in ks]}
            for m in MACROS]


def steer(name: str, to: float, base: SongParams, now: SongParams) -> list[Op]:
    """The ops that move macro `name` from where it reads now to `to`."""
    macro = BY_NAME[name]
    ks = all_knobs(now)
    to = max(-1.0, min(1.0, float(to)))
    frm = _position(macro, base, now, ks)
    ops = []
    for entry in macro.knobs:
        b = _bounds(entry, ks)
        if b is None:
            continue
        path, w, *_ = entry
        lo, hi = b
        b0 = read_knob(base, path)
        cur = read_knob(now, path)
        target = cur + _offset(to, w, b0, lo, hi) - _offset(frm, w, b0, lo, hi)
        k = ks[path]
        target = k.clamp(max(min(target, max(hi, cur)), min(lo, cur)))
        if abs(target - cur) > 1e-6:
            ops.append(Op(path, to=target))
    return ops
