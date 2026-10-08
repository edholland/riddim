"""A self-critic: the engine's own knowledge applied to a finished song.

Generation is one direction of the engine; this is the other. The same
knowledge that keeps the generators musical - registers, the mute map, the
energy curve, the rule that the downbeat says what the chord is - runs in
reverse over a completed song and says, plainly, where it is not doing its
job. It reports, it does not rewrite: the maker decides what to fix. The CLI
surfaces it with `--check`.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import profile as profile_mod
from .song import Song
from .units import TICKS_PER_16TH, TICKS_PER_BAR


@dataclass(frozen=True)
class Finding:
    """One thing that is not doing its job.

    `where` is a part role, a section name, or 'song'. `severity` is 'note'
    (worth knowing) or 'warning' (likely to sound wrong).
    """

    where: str
    message: str
    severity: str = "note"


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
    return findings


def _mute_map(song: Song) -> list[Finding]:
    """A role listed in a section must be heard in it.

    The mute map is the entire arrangement: a part that is listed as playing
    but contributes no notes there is a section that is not doing what the
    plan says it is.
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
                ))
    return out


def _registers(song: Song, prof) -> list[Finding]:
    """Each part sits in its register window - the mix's division of labour."""
    out = []
    for part in song.parts:
        if part.output == "kit":
            continue
        try:
            window = profile_mod.register_for(prof, part.role)
        except ValueError:
            continue
        bad = [n for n in part.notes if not window.low <= n.midi <= window.high]
        if bad:
            lo = min(n.midi for n in bad)
            hi = max(n.midi for n in bad)
            out.append(Finding(
                part.role,
                f"{len(bad)} notes outside {window} "
                f"({profile_mod.pitch.note_name(lo)}-{profile_mod.pitch.note_name(hi)})",
                "warning",
            ))
    return out


def _key(song: Song, prof) -> list[Finding]:
    """A pitched note is a scale tone, or a tone of a chord that borrows one.

    Anything else is outside the key entirely - the one thing the degree
    system is supposed to make impossible.
    """
    if song.scale is None:
        return []
    allowed = set(song.scale.pitch_classes)
    if song.harmony is not None:
        for ev in song.harmony.events:
            allowed |= ev.chord.pitch_classes()
    out = []
    for part in song.parts:
        if part.output == "kit":
            continue
        bad = [n for n in part.notes if n.midi % 12 not in allowed]
        if bad:
            out.append(Finding(part.role, f"{len(bad)} notes outside the key"))
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
        out = Finding(
            "sub",
            f"confirms the chord root in only {share:.0%} of its bars",
            "warning" if fast else "note",
        )
        return [out]
    return []


def _lead_chord_tones(song: Song) -> list[Finding]:
    """A lead that never agrees with the harmony reads as a competing tune."""
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
        if n.midi % 12 in ev.chord.pitch_classes():
            on_tone += 1
    if hits >= 8 and on_tone / hits < 0.25:
        return [Finding(
            "lead", f"agrees with the harmony in {on_tone / hits:.0%} of its notes",
        )]
    return []


def _energy_curve(song: Song) -> list[Finding]:
    """The energy curve is the arrangement's argument; check the argument.

    - a peak that is not louder than everything before it is not a peak;
    - a build that never climbs is a plateau wearing a build's name;
    - a hole that is loud is not a hole.
    """
    out = []
    seen_max = 0.0
    for section in song.arrangement.sections:
        if section.energy >= 0.85 and section.energy < seen_max:
            out.append(Finding(
                section.name,
                f"peak at {section.energy:.2f} arrives after {seen_max:.2f}",
            ))
        seen_max = max(seen_max, section.energy)
        if section.name.startswith("build") and section.ramp <= 0 and section.bars >= 8:
            out.append(Finding(
                section.name, "is a build that never climbs (ramp is 0)",
            ))
        hole = section.name in ("dub_out", "breakdown") or \
            section.active in ({"drums", "sub"}, {"pad"}, set())
        if hole and section.energy > 0.5:
            out.append(Finding(
                section.name, f"hole at {section.energy:.2f} energy is loud",
            ))
    return out


def _monophony(song: Song) -> list[Finding]:
    """One oscillator on the sub voice: no two notes may sound at once."""
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
            ))
    return out
