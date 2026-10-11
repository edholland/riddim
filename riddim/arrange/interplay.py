"""Parts that listen: the theme plan, and the conversation around it.

A band is not six loops that happen to be in the same key. Somebody has the
tune, and everybody else is *placed around it*: the bass locks to its rhythm
or answers it, the chords stab into its gaps instead of on top of its notes,
the pad holds a slow line that moves against it. This module decides who has
the tune in each section and where everyone else may speak.

The plan for a section is a `Line`: the lead's notes (in diatonic space),
plus the **answer windows** - the hook's breaths and rested units, where the
band is invited to reply - and the **bass line**, the stretches where the
sub takes the hook's rhythm itself. The part builders in `compose` read the
plan; nothing here emits MIDI.

What the hook does is decided by what the section *is*:

    intro    the cell, once, at the very end - a tease
    build    the cell's head, sparse then every unit, climbing a step every
             two units, and in the last unit diminished to double speed
    peak     the whole hook, repeated; passes the lead sits out are handed
             to the bass, which plays the hook's rhythm under the riddim
    body     the whole hook (dub versions), passes rested by `presence`
    release  the first half of the hook at half speed - the long-tones
             memory of it over the hole
    outro    the cell, then its head, then nothing
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace

from ..theory import melody
from ..theory.hook import Hook, Unit
from ..theory.melody import Motif, Tone


@dataclass(frozen=True)
class InterplayParams:
    presence: float = 0.75   # share of hook passes the lead sings in a peak/body
    answer: float = 0.6      # how readily chords and bass reply in the lead's gaps
    lock: float = 0.35       # how often the sub takes the hook's rhythm
    counter: float = 0.5     # the pad's top voice as a moving counter-line
    space: float = 0.6       # how far the band ducks out of the lead's onsets


def params_for(family: str, rng: random.Random) -> InterplayParams:
    return InterplayParams(
        presence=round(rng.uniform(0.55, 0.95), 2),
        answer=round(rng.uniform(0.7, 1.0), 2),
        lock=round(rng.uniform(0.25, 0.7) if family in ("dubstep", "garage")
                   else rng.uniform(0.1, 0.45), 2),
        counter=round(rng.uniform(0.3, 0.85), 2),
        space=round(rng.uniform(0.75, 1.0), 2),
    )


@dataclass
class Line:
    """The lead's plan for one section, and the room it leaves."""

    tones: list[Tone] = field(default_factory=list)   # steps from section start
    treatment: str = ""
    answers: list[tuple[float, float]] = field(default_factory=list)
    # Whole passes the lead sits out: the bass has the hook there, and the
    # chords go back to plain stabs rather than answering a call nobody made.
    rested: list[tuple[float, float]] = field(default_factory=list)
    bass_line: list[Tone] = field(default_factory=list)
    # The replies: (start, end, who, figure) - in the space after a call,
    # `who` ("sub" or "chords") plays the call's figure back: a list of
    # (step from `start`, degree relative to the figure's first note).
    responses: list[tuple] = field(default_factory=list)

    def response_in(self, start: float, end: float, who: str | None = None) -> list[tuple]:
        return [r for r in self.responses if r[0] < end and r[1] > start
                and (who is None or r[2] == who)]

    def onsets(self) -> set[int]:
        return {int(round(t.step)) for t in self.tones}

    def sounding(self) -> set[int]:
        """Every step the lead is holding a note on."""
        out = set()
        for t in self.tones:
            out.update(range(int(t.step), int(math.ceil(t.step + t.length))))
        return out

    def answers_in(self, start: float, end: float) -> list[tuple[float, float]]:
        return [(max(a, start), min(b, end)) for a, b in self.answers
                if a < end and b > start]


# --------------------------------------------------------------------------
# Hook variants for the lead styles
# --------------------------------------------------------------------------

def styled(hook: Hook, style: str) -> Hook:
    """The hook as a lead style delivers it.

    hook           as written
    riff           the cell three times and the closing unit - a loop with a
                   turnaround, for when the hook should behave like a riff
    call_response  units two and four rested, so the band answers every call
    long_tones     the first half at half speed - the melodica version
    """
    u = hook.units
    if style == "riff":
        units = [u[0], u[0], u[0], u[-1] if u[-1].tones else u[0]]
    elif style == "call_response":
        # The rested unit is the whole answer, so the call holds its last
        # note through its own breath instead of leaving a second gap.
        def held(unit: Unit) -> Unit:
            tones = sorted(unit.tones, key=lambda t: t.step)
            if tones:
                last = tones[-1]
                tones[-1] = replace(last, length=max(last.length, hook.unit_steps - last.step - 1))
            return Unit(unit.role, tones, unit.steps)
        call2 = u[2] if u[2].tones else u[0]
        units = [held(u[0]), Unit("rest", [], hook.unit_steps), held(call2),
                 Unit("rest", [], hook.unit_steps)]
    elif style == "long_tones":
        units = _augmented(hook)
    else:
        return hook
    return Hook(units, hook.params, hook.unit_steps, hook.contour, hook.score, hook.report)


def _augmented(hook: Hook) -> list[Unit]:
    """Units 0-1 stretched to fill all four."""
    steps = hook.unit_steps
    out: list[Unit] = []
    for src in hook.units[:2]:
        first = [replace(t, step=t.step * 2, length=t.length * 2)
                 for t in src.tones if t.step * 2 < steps]
        second = [replace(t, step=t.step * 2 - steps, length=t.length * 2)
                  for t in src.tones if t.step * 2 >= steps]
        out.append(Unit(src.role, first, steps))
        out.append(Unit(src.role + "~", second, steps))
    return out


def breaths(hook: Hook, offset: float) -> list[tuple[float, float]]:
    """Where a pass of `hook` leaves room: each unit's tail, and rested units."""
    out = []
    U = hook.unit_steps
    for i, u in enumerate(hook.units):
        start = offset + i * U
        if not u.tones:
            out.append((start, start + U))
            continue
        tail = max(t.step + t.length for t in u.tones)
        if U - tail >= 2:
            out.append((start + math.ceil(tail), start + U))
    return out


# --------------------------------------------------------------------------
# The plan
# --------------------------------------------------------------------------

def figure_of(unit: Unit) -> list[tuple[float, int]]:
    """The call's figure: the notes before its first landing, as (step from
    the first note, degree relative to it). This is what a reply plays."""
    tones = sorted(unit.tones, key=lambda t: t.step)
    if not tones:
        return []
    out = []
    for t in tones:
        out.append((t.step - tones[0].step, t.degree - tones[0].degree))
        if t.length >= 3.0 and len(out) >= 2:
            break          # the landing ends the figure
    return out[:5]


def _respond(line: Line, variant: Hook, offset: float, ip: InterplayParams,
             roles, rng: random.Random, unit_index: int) -> None:
    """Hand each answer space after a sung unit to a responder - the bass and
    the chords take turns - with the call's own figure to play back."""
    order = ["sub", "chords"] if unit_index % 2 == 0 else ["chords", "sub"]
    who_list = [r for r in order if r in roles]
    U = variant.unit_steps
    last_call = None
    for i, u in enumerate(variant.units):
        start = offset + i * U
        if not u.tones:
            # A rested unit is a whole bar or two of answer to the last call.
            a, b, fig = start, start + U, figure_of(last_call) if last_call else []
        else:
            last_call = u
            tail = max(t.step + t.length for t in u.tones)
            a, b, fig = start + math.ceil(tail), start + U, figure_of(u)
        if b - a < 3 or not fig or not who_list or rng.random() > ip.answer:
            continue
        who = who_list[(unit_index + i) % len(who_list)]
        line.responses.append((a, b, who, fig))


def plan(
    kind: str,
    bars: int,
    hook: Hook,
    realise_at,
    style: str,
    ip: InterplayParams,
    peak_index: int,
    rng: random.Random,
    lead_plays: bool = True,
    fits_lift=None,
    roles: tuple = ("sub", "chords"),
    hook_b: Hook | None = None,
) -> Line:
    """Plan the hook's treatment for one section.

    `realise_at(hook, offset)` fits a hook variant to the chords starting at
    `offset` steps into the section and returns its tones there. `peak_index`
    counts peaks so far (0 for the first drop): later peaks vary more.
    `fits_lift(tones)` says whether those tones can go up an octave and stay
    in the lead's window.
    """
    total = bars * 16
    U = hook.unit_steps
    P = hook.length
    line = Line(treatment=kind)

    if kind in ("peak", "body"):
        variant = styled(hook, style)
        passes = max(1, math.ceil(total / P))
        n_sing = max(1, int(round(ip.presence * passes))) if lead_plays else 0
        if lead_plays and kind == "peak" and peak_index == 0:
            # The first drop establishes the hook: it sings every pass (all
            # but one in a long drop). Later drops have earned the rests.
            n_sing = max(n_sing, passes - 1 if passes >= 4 else passes)
        sing = {int(round(i * passes / n_sing)) for i in range(n_sing)} if n_sing else set()
        # A later drop brings the B-theme for one pass - first, when there
        # are only two (new material, then the hook as the payoff), else
        # second-to-last (the hook frames it and has the last word).
        b_pass = None
        if hook_b is not None and kind == "peak" and peak_index >= 1 and passes >= 2 \
                and rng.random() < hook.params.contrast:
            b_pass = 0 if passes == 2 else passes - 2
            sing.add(b_pass)
        variant_b = styled(hook_b, style) if hook_b is not None else None
        for k in range(passes):
            off = k * P
            if k in sing:
                this = variant_b if k == b_pass else variant
                # Later drops develop the hook: one pass in four is
                # *sequenced* - the restatement climbs a third, refitted to
                # the chords - the composer's oldest way to make a figure go
                # somewhere without becoming a different figure.
                sequenced = peak_index >= 1 and k % 4 == 3 and k != b_pass and len(this.units) >= 2
                if sequenced:
                    this = _sequenced(this)
                tones = realise_at(this, off)
                # The first drop *establishes* the hook: literal, bar for bar,
                # decorated only on a fourth pass. Later drops have earned the
                # right to play with it - every second pass is ornamented.
                if not sequenced and ((peak_index >= 1 and k % 2 == 1) or k % 4 == 3):
                    tones = _ornament(tones, rng)
                last_sung = k == max(sing)
                if peak_index >= 1 and last_sung and k > 0 and fits_lift is not None \
                        and fits_lift(tones):
                    tones = [replace(t, degree=t.degree + 7) for t in tones]
                line.tones += tones
                line.answers += breaths(this, off)
                _respond(line, this, off, ip, roles, rng, k)
                if kind == "peak" and rng.random() < ip.lock * 0.5:
                    line.bass_line += realise_at(hook if k != b_pass else hook_b, off)
            else:
                line.rested.append((off, off + P))
                if kind == "peak" and (not lead_plays or rng.random() < ip.lock):
                    line.bass_line += realise_at(hook, off)       # the bass answers
        line.treatment = f"{kind}:{style}"

    elif kind == "build" and lead_plays:
        head = [t for t in hook.cell.tones if t.step < U / 2] or hook.cell.tones[:2]
        n_units = max(1, total // U)
        for j in range(n_units):
            off = j * U
            dense = j >= n_units / 2
            if not dense and j % 2 == 1:
                line.answers.append((off, off + U))
                continue
            # Climb a step at a time across the whole build, so the top of
            # the climb is the bar before the drop - not halfway through.
            # ...and stop a step short of the hook's own high point: the
            # build approaches the climax, it does not spend it.
            hook_top = max((t.degree for u in hook.units for t in u.tones), default=0)
            head_top = max((t.degree for t in head), default=0)
            # Where the figure already sits near the top, the climb starts
            # lower than written, so the build still rises - to the step below.
            cap = hook_top - 1 - head_top
            rise = min(cap, 3) - 3 + (j * 4) // n_units
            if j == n_units - 1 and n_units > 1:
                # Diminution: the head at double speed, twice - the lead
                # running at the drop.
                for rep in range(2):
                    line.tones += [replace(t, step=off + rep * U / 4 + t.step / 2,
                                           length=max(0.5, t.length / 2),
                                           degree=t.degree + rise)
                                   for t in head]
            else:
                line.tones += [replace(t, step=off + t.step, degree=t.degree + rise)
                               for t in head]
                if dense and U >= 16:
                    line.tones += [replace(t, step=off + U / 2 + t.step,
                                           degree=t.degree + rise)
                                   for t in head if t.step + U / 2 < U]
                line.answers.append((off + U / 2, off + U) if not dense else (off + U - 2, off + U))
        line.treatment = "build:fragment"

    elif kind == "release" and lead_plays:
        aug = Hook(_augmented(hook), hook.params, U, hook.contour)
        line.tones = [replace(t, velocity=max(40, t.velocity - 18))
                      for t in realise_at(aug, 0)]
        line.answers = breaths(aug, 0)
        _respond(line, aug, 0, ip, [r for r in roles if r == "chords"] or roles, rng, 0)
        line.treatment = "release:augmented"

    elif kind == "intro" and lead_plays and total >= 2 * U:
        off = total - U
        line.tones = [replace(t, step=off + t.step, velocity=max(40, t.velocity - 20))
                      for t in hook.cell.tones]
        line.treatment = "intro:tease"

    elif kind == "outro" and lead_plays:
        line.tones = [replace(t, velocity=max(40, t.velocity - 10)) for t in hook.cell.tones]
        if total >= 3 * U:
            head = [t for t in hook.cell.tones if t.step < U / 2] or hook.cell.tones[:2]
            line.tones += [replace(t, step=2 * U + t.step, velocity=max(36, t.velocity - 25))
                           for t in head]
        line.treatment = "outro:dissolve"

    # Nothing may ring past the section.
    line.tones = [replace(t, length=min(t.length, total - t.step))
                  for t in line.tones if t.step < total]
    line.bass_line = [t for t in line.bass_line if t.step < total]
    line.answers = [(a, min(b, total)) for a, b in line.answers if a < total and b - a >= 2]
    line.rested = [(a, min(b, total)) for a, b in line.rested if a < total]
    line.responses = [(a, min(b, total), w, f) for a, b, w, f in line.responses if a < total]
    return line


def _sequenced(hook: Hook) -> Hook:
    """The hook with its restatement (the second unit) sequenced up a third
    and the third unit carried with it, so the phrase climbs and the close
    comes home from higher up."""
    units = list(hook.units)
    for i in (1, 2):
        if i < len(units) and units[i].tones:
            units[i] = Unit(units[i].role, [replace(t, degree=t.degree + 2) for t in units[i].tones],
                            units[i].steps)
    return Hook(units, hook.params, hook.unit_steps, hook.contour, hook.score, hook.report)


def _ornament(tones: list[Tone], rng: random.Random) -> list[Tone]:
    """Passing notes through the thirds - the second time round, decorated."""
    if not tones:
        return tones
    m = Motif(list(tones), int(max(t.end for t in tones)) + 1)
    return melody.ornament(m, 0.6, rng).sorted_tones()
