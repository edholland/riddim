"""A self-critic: the engine's own knowledge applied to a finished song.

Generation is one direction of the engine; this is the other. The same
knowledge that keeps the generators musical - registers, the mute map, the
energy curve, the rule that the downbeat says what the chord is - runs in
reverse over a completed song and says, plainly, where it is not doing its
job.

It reports, and it *proposes*: every finding carries the change that would
cure it. Whether that change is a twist (a knob move - a small step that
preserves the song's identity) or a roll (a new take of the part the
generator got wrong) follows the same rule as the steering model. `revise`
walks the loop - worst finding, apply its fix, recompose, re-critique -
until the song is clean or a step stops improving the score. The CLI
surfaces it with `--check` (the report) and `--fix` (the loop).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from . import profile as profile_mod
from .compose import compose, regenerate
from .direct import Op, _ensure_resolved, _rig_of, apply_ops
from .song import Song
from .units import TICKS_PER_16TH, TICKS_PER_BAR


@dataclass(frozen=True)
class Fix:
    """How a finding can be cured.

    `kind` is the steering vocabulary: "twist" is a list of knob ops applied
    with the seeds frozen - the result is a small move from where the song
    was. "roll" re-draws one part - a new take, the honest answer to a
    generator bug that no knob can cure.
    """

    kind: str
    ops: tuple[Op, ...] = ()
    part: str | None = None
    why: str = ""

    def describe(self) -> str:
        if self.kind == "roll":
            return f"re-roll {self.part}"
        return ", ".join(o.describe() for o in self.ops)


@dataclass(frozen=True)
class Finding:
    """One thing that is not doing its job.

    `where` is a part role, a section name, or 'song'. `severity` is 'note'
    (worth knowing) or 'warning' (likely to sound wrong). `fixes` are the
    changes that would cure it, in order of preference.
    """

    where: str
    message: str
    severity: str = "note"
    fixes: tuple[Fix, ...] = ()


# --------------------------------------------------------------------------
# The checks
# --------------------------------------------------------------------------

def critique(song: Song) -> list[Finding]:
    """Run every check the engine knows over a composed song."""
    findings: list[Finding] = []
    params = song.params
    prof = profile_mod.get(params.profile) if params is not None else None

    findings += _mute_map(song)
    if prof is not None:
        findings += _registers(song, prof)
        findings += _key(song, prof)
    findings += _sub_confirms(song)
    findings += _lead_chord_tones(song)
    findings += _energy_curve(song)
    findings += _monophony(song)
    findings += _hook(song)
    if prof is not None:
        findings += _lead_register(song, prof)
    findings += _drops(song)
    return findings


# Below this the hook is a line, not a hook: a new take of the theme.
HOOK_FLOOR = 0.72


def _hook(song: Song) -> list[Finding]:
    """The hook must be strong, and it must be heard where it counts.

    A weak hook (by its own score) is a roll of the hook seed - a new tune
    over the same band. A peak where neither the lead sings nor the sub
    carries the hook's rhythm is a twist: more presence, or a tighter lock.
    """
    from .arrange.structure import kind_of
    out = []
    params = song.params
    if params is None or params.hook is None:
        return out
    concept = song.concept or ""
    if "score=" in concept:
        try:
            s = float(concept.split("score=")[1].split()[0])
        except ValueError:
            s = 1.0
        if s < HOOK_FLOOR:
            out.append(Finding(
                "lead", f"hook scores {s:.2f} - more a line than a hook", "note",
                (Fix("roll", part="hook", why="a new hook over the same band"),),
            ))
    lead = song.part("lead")
    sub = song.part("sub")
    for i, sec in enumerate(song.arrangement.sections):
        if kind_of(sec) != "peak":
            continue
        sung = lead is not None and len(lead.in_bars(sec.start_bar, sec.end_bar)) >= sec.bars // 2
        carried = sub is not None and sec.plays("sub")
        if not sung and not carried:
            out.append(Finding(
                sec.name, "a peak with no hook in it", "warning",
                (Fix("twist", (Op("interplay.presence", to=1.0),
                               Op("interplay.lock", to=0.8)),
                     why="the lead sings every pass, the sub takes the rest"),),
            ))
    return out


def _lead_register(song: Song, prof) -> list[Finding]:
    """The tune sits above the stabs. A lead living inside the chord window
    is a melody nobody can pick out: lift it an octave."""
    lead, chords = song.part("lead"), song.part("chords")
    if lead is None or chords is None or len(lead.notes) < 8 or not chords.notes:
        return []
    top = sorted(n.midi for n in chords.notes)[int(len(chords.notes) * 0.9)]
    inside = sum(1 for n in lead.notes if n.midi <= top) / len(lead.notes)
    if inside > 0.5 and song.params is not None and song.params.lead is not None \
            and song.params.lead.octave < 1:
        return [Finding(
            "lead", f"{inside:.0%} of the lead sits inside the stabs' register", "note",
            (Fix("twist", (Op("lead.octave", 1.0),), why="lift the tune clear of the chords"),),
        )]
    return []


def _drops(song: Song) -> list[Finding]:
    """A drop is an event: a roll into it, a beat of silence, a crash on it.
    A peak that arrives with none of that is a level change, not a drop."""
    from .arrange import dynamics
    params = song.params
    if params is None or params.dynamics is None:
        return []
    dyn = params.dynamics
    secs = song.arrangement.sections
    drops = [s for p, s in zip(secs, secs[1:]) if dynamics.drop_entry(p, s)]
    if not drops:
        return []
    if dyn.roll < 0.15 and dyn.drop_gap <= 0 and dyn.impact < 0.15:
        return [Finding(
            drops[0].name, "the drop arrives with no roll, gap or impact", "note",
            (Fix("twist", (Op("dynamics.roll", to=0.7), Op("dynamics.drop_gap", to=1.0),
                           Op("dynamics.impact", to=0.8)),
                 why="roll in, stop for a beat, land on a crash"),),
        )]
    return []


def _mute_map(song: Song) -> list[Finding]:
    """A role listed in a section must be heard in it.

    The mute map is the entire arrangement: a part that is listed as playing
    but contributes no notes there is a section that is not doing what the
    plan says it is. No knob can guarantee emission, so the fix is a take.
    """
    out = []
    for section in song.arrangement.sections:
        for part in song.parts:
            if not section.plays(part.role):
                continue
            if not part.in_bars(section.start_bar, section.end_bar):
                out.append(Finding(
                    part.role,
                    f"listed in {section.name} but silent there",
                    "warning",
                    (Fix("roll", part=part.role,
                          why="a new take of that part"),),
                ))
    return out


def _registers(song: Song, prof) -> list[Finding]:
    """Each part sits in its register window - the mix's division of labour.

    A note that escaped the window is a generator bug; the window is not a
    knob, so the fix is a take of the part.
    """
    out = []
    for part in song.parts:
        if part.output == "kit":
            continue
        try:
            window = profile_mod.register_for(prof, part.role)
        except ValueError:
            continue
        if part.role == "lead" and song.params is not None and song.params.lead is not None:
            # The lead's window moves with its register dial (and a flick
            # may poke five semitones over the top).
            from .compose import lead_window
            lo, hi = lead_window(prof, song.params.lead)
            window = profile_mod.Register(lo, hi + 5)
        bad = [n for n in part.notes if not window.low <= n.midi <= window.high]
        if bad:
            lo = min(n.midi for n in bad)
            hi = max(n.midi for n in bad)
            out.append(Finding(
                part.role,
                f"{len(bad)} notes outside {window} "
                f"({profile_mod.pitch.note_name(lo)}-{profile_mod.pitch.note_name(hi)})",
                "warning",
                (Fix("roll", part=part.role,
                      why="a new take stays in its window"),),
            ))
    return out


def _key(song: Song, prof) -> list[Finding]:
    """A pitched note is a scale tone, or a tone of a chord that borrows one.

    Anything else is outside the key entirely - the one thing the degree
    system is supposed to make impossible. A take of the part is the fix.
    """
    if song.scale is None:
        return []
    allowed = set(song.scale.pitch_classes)
    if song.harmony is not None:
        for ev in song.harmony.events:
            allowed |= ev.chord.pitch_classes()
            if ev.meta.get("lift"):
                # after a key lift the new key's scale is the key
                allowed |= {(pc + ev.meta["lift"]) % 12 for pc in song.scale.pitch_classes}
    out = []
    for part in song.parts:
        if part.output == "kit":
            continue
        bad = [n for n in part.notes if n.midi % 12 not in allowed]
        if bad:
            out.append(Finding(
                part.role, f"{len(bad)} notes outside the key",
                "note",
                (Fix("roll", part=part.role,
                      why="a new take is built from degrees"),),
            ))
    return out


def _sub_confirms(song: Song) -> list[Finding]:
    """The downbeat says what the chord is.

    Bar by bar, does the bass put a root under the current chord within the
    first half of the bar? A bass that mostly skips that is counterpoint, not
    a foundation - fine in one hole, a problem song-wide, and a real problem
    when the harmony is changing fast and there is no time to catch up.
    """
    if song.harmony is None:
        return []
    sub = song.part("sub")
    if sub is None:
        return []
    bars = range(song.arrangement.bars)
    checked = confirmed = 0
    for bar in bars:
        notes = sub.in_bars(bar, bar + 1)
        if not notes:
            continue
        ev = song.harmony.at(bar * TICKS_PER_BAR)
        if ev is None:
            continue
        checked += 1
        for n in notes:
            if (n.start % TICKS_PER_BAR) <= TICKS_PER_16TH * 8 and \
                    n.midi % 12 == ev.chord.root_pc:
                confirmed += 1
                break
    if checked < 8:
        return []
    share = confirmed / checked
    fast = song.params is not None and (song.params.bars_per_chord or 2) < 2
    if share < (0.25 if fast else 0.35):
        if fast:
            # The chord is moving faster than the bass can follow: slow the
            # harmonic rhythm so each root is held long enough to be under.
            fixes = (Fix("twist", (Op("bars_per_chord", to=2.0),),
                         why="two bars per chord, so the root can be under it"),)
        else:
            # The rhythm is fine; the figure is wandering. Only a new take
            # of the bass answers that.
            fixes = (Fix("roll", part="sub",
                         why="a new figure that stays on the root"),)
        return [Finding(
            "sub",
            f"confirms the chord root in only {share:.0%} of its bars",
            "warning" if fast else "note",
            fixes,
        )]
    return []


def _lead_chord_tones(song: Song) -> list[Finding]:
    """A lead that never agrees with the harmony reads as a competing tune.

    The melody is developed by contour, which can drift off the chord; a new
    take re-snaps it.
    """
    if song.harmony is None:
        return []
    lead = song.part("lead")
    if lead is None or not lead.notes:
        return []
    hits = on_tone = 0
    for n in lead.notes:
        ev = song.harmony.at(n.start)
        if ev is None:
            continue
        hits += 1
        if n.midi % 12 in ev.chord.pitch_classes() | {ev.chord.root_pc}:
            on_tone += 1
    if hits >= 8 and on_tone / hits < 0.25:
        return [Finding(
            "lead", f"agrees with the harmony in {on_tone / hits:.0%} of its notes",
            "note",
            (Fix("roll", part="lead", why="a new line snaps to the chord tones"),),
        )]
    return []


def _energy_curve(song: Song) -> list[Finding]:
    """The energy curve is the arrangement's argument; check the argument.

    - a peak that is not louder than everything before it is not a peak;
    - a build that never climbs is a plateau wearing a build's name;
    - a hole that is loud is not a hole.

    All three are plain section numbers, so all three are twists.
    """
    out = []
    seen_max = 0.0
    for i, section in enumerate(song.arrangement.sections):
        if section.energy >= 0.85 and section.energy < seen_max:
            out.append(Finding(
                section.name,
                f"peak at {section.energy:.2f} arrives after {seen_max:.2f}",
                "note",
                (Fix("twist",
                     (Op(f"sections.{i}.energy",
                         to=round(min(1.0, seen_max + 0.05), 2)),),
                     why="a peak must clear everything before it"),),
            ))
        seen_max = max(seen_max, section.energy)
        # (Not in a song whose holes take out the low end - the 140 records:
        # there the approach to a drop is a level, held, and the drop is the
        # low end arriving. An EDM ramp into it is the machine's habit.)
        flat_ok = song.params is not None and song.params.structure is not None \
            and song.params.structure.low_end_out
        if section.name.startswith("build") and section.ramp <= 0 and section.bars >= 8 \
                and not flat_ok:
            out.append(Finding(
                section.name, "is a build that never climbs (ramp is 0)",
                "note",
                (Fix("twist", (Op(f"sections.{i}.ramp", to=0.6),),
                     why="a build earns its name by climbing"),),
            ))
        hole = section.name in ("dub_out", "breakdown") or \
            section.active in ({"drums", "sub"}, {"pad"}, set())
        if hole and section.energy > 0.5:
            out.append(Finding(
                section.name, f"hole at {section.energy:.2f} energy is loud",
                "note",
                (Fix("twist", (Op(f"sections.{i}.energy", to=0.3),),
                     why="a hole is a place to go, not a destination"),),
            ))
    return out


def _monophony(song: Song) -> list[Finding]:
    """One oscillator on the sub voice: no two notes may sound at once.

    Overlaps are a generator bug - the fix is a take of the voice.
    """
    out = []
    for part in song.parts:
        if not part.monophonic:
            continue
        notes = part.sorted_notes()
        overlaps = sum(1 for a, b in zip(notes, notes[1:]) if b.start < a.end)
        if overlaps:
            out.append(Finding(
                part.role, f"{overlaps} overlapping notes on a monophonic voice",
                "warning",
                (Fix("roll", part=part.role, why="a new take never overlaps"),),
            ))
    return out


# --------------------------------------------------------------------------
# The score, and the revision loop
# --------------------------------------------------------------------------

# What a finding costs in song-credibility. A warning is something likely to
# sound wrong; a note is something worth knowing. The score is what makes a
# batch of songs comparable, and what tells the loop when a fix helped.
WARN_WEIGHT = 0.2
NOTE_WEIGHT = 0.1


def score(song: Song) -> float:
    """How well the song does its job, 1.0 when the critic finds nothing."""
    total = 1.0
    for f in critique(song):
        total -= WARN_WEIGHT if f.severity == "warning" else NOTE_WEIGHT
    return max(0.0, total)


@dataclass(frozen=True)
class ReviseStep:
    """One move in the revision loop, with the score on either side."""

    finding: Finding
    fix: Fix
    score_before: float
    score_after: float
    applied: bool = True


def next_fix(song: Song) -> tuple[Finding, Fix] | None:
    """The next move of the revision loop: the worst actionable finding and
    the fix to apply - twists before rolls, warnings before notes. None when
    there is nothing to fix."""
    findings = critique(song)
    for want in ("warning", "note"):
        for f in findings:
            if f.severity == want and f.fixes:
                fix = next((fx for fx in f.fixes if fx.kind == "twist"),
                           f.fixes[0])
                return f, fix
    return None


def revise(
    song: Song,
    max_steps: int = 8,
    min_gain: float = 0.01,
    rig=None,
) -> tuple[Song, list[ReviseStep]]:
    """The engine improving its own song.

    Worst finding, apply its fix, recompose, re-critique. A twist keeps the
    seeds frozen and moves the song a little; a roll is a new take of the part
    the generator got wrong. A step that does not improve the score is
    reported and not kept - the score is quantised in the weight grid, and a
    strictly-rising walk on a finite state space always ends.

    Rolls here are seeded from the song's own seed (`{seed}:fix:{part}`), so
    a given song revises the same way every time: `--fix` and `batch` promise
    reproducible output. A maker's manual re-roll in the UI stays random -
    that one is a creative act, not a rule being applied.
    """
    if song.params is None:
        # A hand-built song has no steering surface - nothing to twist, and a
        # roll would crash on the missing params. Report it as-is.
        return song, []
    song = _ensure_resolved(song)
    rig = rig or _rig_of(song)

    current = song
    before = score(current)
    steps: list[ReviseStep] = []
    for _ in range(max_steps):
        chosen = next_fix(current)
        if chosen is None:
            break
        f, fix = chosen
        if fix.kind == "twist":
            nxt = compose(apply_ops(current.params, list(fix.ops)), rig=rig)
        else:
            nxt = regenerate(current, fix.part, rig=rig,
                             rng=random.Random(f"{current.seed}:fix:{fix.part}"))
        after = score(nxt)
        applied = after - before >= min_gain
        steps.append(ReviseStep(f, fix, before, after, applied))
        if not applied:
            break
        current, before = nxt, after
    return current, steps
