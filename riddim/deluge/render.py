"""Send a Song to the Deluge as a grid of session clips.

Layout: one clip per (role, section). Each Deluge section holds every role
that plays in that part of the arrangement, so launching a section fires the
whole arrangement for those bars, and `set_section_repeat` chains the sections
into a song that plays start to finish without being driven.

Roles get one Deluge output each, not one per section. The first clip for a
role is *created*; every later clip for that role is *cloned* from it, which
is what makes them share an instrument instead of spawning eight synths.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..song import Song, Part
from ..units import TICKS_PER_BAR
from . import sysex
from .client import Deluge, DelugeError
from .sysex import OutputType

# If this many notes in a row are rejected, something systemic is wrong - the
# wrong clip is selected, or the clip length never took, or the firmware has
# stopped answering. Grinding through the remaining thousand notes with three
# retries and a sleep each would take minutes and tell us nothing new.
MAX_CONSECUTIVE_FAILURES = 8

OUTPUT_TYPES = {
    "synth": OutputType.SYNTH,
    "kit": OutputType.KIT,
    "midi": OutputType.MIDI,
    "cv": OutputType.CV,
    "audio": OutputType.AUDIO,
}

# Deluge session sections are colour groups. `kMaxNumSections` in
# definitions_cxx.hpp.
MAX_SECTIONS = 24


@dataclass
class RenderReport:
    clips: int = 0
    notes: int = 0
    automation: int = 0
    trimmed: int = 0
    failures: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        line = (
            f"{self.clips} clips, {self.notes} notes, "
            f"{self.automation} automation moves, {self.trimmed} clips trimmed"
        )
        if self.failures:
            line += f"\n{len(self.failures)} failures:"
            for f in self.failures[:10]:
                line += f"\n  {f}"
            if len(self.failures) > 10:
                line += f"\n  ... and {len(self.failures) - 10} more"
        return line


def send(
    song: Song,
    deluge: Deluge,
    start_index: int | None = None,
    mode: str = "jam",
    template: dict[str, int] | None = None,
    template_rows: dict[str, int] | None = None,
    progress: bool = True,
) -> RenderReport:
    """Write `song` into the Deluge session.

    `mode` is "jam" (sections loop until you launch the next) or "song"
    (sections auto-advance through the arrangement). See `_chain`.

    `template` maps a role to the index of an existing clip to clone from.
    Cloning shares the source's output, so a role pointed at a template clip
    inherits that instrument entirely - its MIDI channel, its patch, its
    sends. That is the only way to control a MIDI clip's output channel,
    since the firmware exposes no command to set one, and it means the
    role-to-channel map in `Rig` describes something real rather than
    something hoped for.

    `start_index` defaults to the end of the current song, so this appends
    rather than assuming it is writing into an empty one.
    """
    if song.arrangement is None:
        raise ValueError("song has no arrangement; nothing to lay out")
    sections = song.arrangement.sections

    if template and template_rows:
        raise ValueError("pass template or template_rows, not both")

    existing = deluge.count_clips()
    if template_rows:
        template = rows_to_indices(template_rows, existing)

    if start_index is None:
        start_index = existing
    elif start_index > existing:
        # The firmware clamps an out-of-range insert index to the end of the
        # list, so asking for a gap does not leave one - it silently puts the
        # clip somewhere else while our counter believes otherwise, and every
        # index after it is wrong.
        raise ValueError(
            f"start_index {start_index} is past the end of the song, which "
            f"has {existing} clips. The Deluge would clamp the insert to the "
            f"end and every index after it would be wrong. Use "
            f"{existing} or less."
        )

    template = dict(template or {})
    if template:
        _check_template(template, start_index, song, deluge)
    if len(sections) > MAX_SECTIONS:
        raise ValueError(
            f"arrangement has {len(sections)} sections but the Deluge has "
            f"{MAX_SECTIONS} colour groups; shorten the template or merge sections"
        )

    report = RenderReport()
    next_index = start_index
    # Seeding this from the template makes every clip for that role take the
    # clone path, so they all share the template's instrument. The template
    # clip itself is only ever a clone source - never selected, cleared or
    # written to.
    first_clip_for_role: dict[str, int] = dict(template)
    # index -> the length the clip is supposed to be, for the trim pass.
    intended_length: dict[int, int] = {}

    for section_number, section in enumerate(sections):
        length = section.length_ticks
        for part in song.parts:
            if not section.plays(part.role):
                continue
            notes = part.in_bars(section.start_bar, section.end_bar)
            if not notes:
                continue

            index = next_index
            next_index += 1

            if part.role in first_clip_for_role:
                ok = deluge.clone(
                    first_clip_for_role[part.role], index, section_number, length
                )
                action = "clone"
            else:
                ok = deluge.create(
                    index, _output_type(part), section_number, length
                )
                first_clip_for_role[part.role] = index
                action = "create"

            if not ok:
                report.failures.append(
                    f"{action} {part.role}/{section.name} at index {index}"
                )
                continue

            if not deluge.select_index(index):
                report.failures.append(f"select index {index} ({part.role})")
                continue

            deluge.set_length(length)
            deluge.clear(0, min(length, sysex.MAX_14))
            deluge.name(f"{part.role[:4]}{section_number}")

            if action == "create" and part.preset and part.output == "synth":
                deluge.preset(part.preset)

            written = _write_notes(
                deluge, notes, section.start_tick, length, report,
                label=f"{part.role}/{section.name}",
            )
            report.notes += written
            report.automation += _write_automation(deluge, part, section, report)
            report.clips += 1
            intended_length[index] = length

            if progress:
                print(
                    f"  [{index:>3}] {section.name:<12} {part.role:<7} "
                    f"{written:>4} notes  ({action})"
                )

    _trim(deluge, intended_length, report, progress)

    if mode != "none":
        _chain(deluge, sections, report, mode)

    return report


def _trim(deluge: Deluge, intended: dict[int, int], report: RenderReport,
          progress: bool) -> None:
    """Force every written clip back to its intended length, and verify it.

    This has to happen at the end, after the notes are in. Writing notes can
    leave a note row carrying its own `loopLengthIfIndependent`, which makes
    the row loop at a length of its own and the clip behave as though it were
    enormously longer than it was created - clips set to 8 bars were turning
    up as roughly 1024. `forceClipLength` (which is what
    CLAMP_SESSION_INDEX_LENGTH calls) zeroes every row's independent length as
    well as setting the clip's, so this pass fixes both.

    It addresses clips by absolute index rather than by selection, and the
    firmware reports the resulting lengths back, so this both repairs and
    checks. Anything that still disagrees is recorded as a failure rather than
    assumed fine.
    """
    for index in sorted(intended):
        want = intended[index]
        detail = deluge.clamp_length(index, want)
        if detail is None:
            report.failures.append(f"trim clip {index}: no reply")
            continue
        if detail["loop_length"] != want:
            report.failures.append(
                f"trim clip {index}: length is {detail['loop_length']}, "
                f"wanted {want}"
            )
            continue
        if detail["max_row_length"]:
            report.failures.append(
                f"trim clip {index}: a note row still loops independently at "
                f"{detail['max_row_length']}"
            )
            continue
        report.trimmed += 1
    if progress:
        print(f"  trimmed {report.trimmed}/{len(intended)} clips to length")


def _check_template(
    template: dict[str, int], start_index: int, song: Song, deluge: Deluge
) -> None:
    """Fail loudly on a template that cannot work, before writing anything."""
    from ..arrange.structure import ROLES

    roles = {p.role for p in song.parts}
    for role, index in list(template.items()):
        if role not in ROLES:
            raise ValueError(
                f"template names unknown role {role!r}; known: "
                f"{', '.join(ROLES)}"
            )
        if role not in roles:
            # The template describes the rig, and a rig can have more
            # instruments than one song uses - a song with no lead is still
            # sent into a template that has a lead clip. Ignore the entry.
            del template[role]
            continue
        if index < 0:
            raise ValueError(
                f"template clip index for {role!r} is negative ({index})"
            )
        if index >= start_index:
            raise ValueError(
                f"template clip for {role!r} is at index {index}, which is "
                f"inside the range being written (from {start_index}). "
                f"Template clips must come before the generated ones."
            )
        if not deluge.clip_exists(index):
            raise ValueError(
                f"template clip for {role!r} at index {index} does not exist "
                f"on the Deluge - is the right song loaded?"
            )


# The rig's layout: which template row (1 = top of screen) holds each role's
# clip - the same numbers as the Coral's MIDI channels in `song.Rig`.
DEFAULT_TEMPLATE_ROWS = "sub=1,chords=2,lead=3,pad=4,drums=5,perc=6"


def rows_to_indices(rows: dict[str, int], num_clips: int) -> dict[str, int]:
    """Convert visible song rows (1 = top) to absolute session clip indices.

    The Deluge displays clips in the reverse of their internal order -
    `visibleClipIndexToInternal` in the firmware is literally
    `numClips - 1 - visibleIndex`. So the clip you see at the top of the
    screen is the *last* one in the list. Counting rows off the screen and
    passing them as indices silently points every role at the wrong
    instrument, which is exactly the kind of mistake that is invisible until
    the bassline comes out of the chord patch.
    """
    out = {}
    for role, row in rows.items():
        if not 1 <= row <= num_clips:
            raise ValueError(
                f"row {row} for {role!r} is outside the song's {num_clips} "
                f"clips (rows are 1-based, counted from the top of the screen)"
            )
        out[role] = num_clips - row
    return out


def parse_template(text: str) -> dict[str, int]:
    """Parse 'chords=0,sub=1,lead=2' into a role -> clip index map."""
    out: dict[str, int] = {}
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(
                f"bad template entry {chunk!r}; expected 'role=index', e.g. "
                f"'chords=0'"
            )
        role, _, index = chunk.partition("=")
        role = role.strip()
        try:
            out[role] = int(index)
        except ValueError as exc:
            raise ValueError(
                f"bad clip index {index.strip()!r} for role {role!r}"
            ) from exc
    if not out:
        raise ValueError("template is empty")
    return out


def _output_type(part: Part) -> OutputType:
    if part.output not in OUTPUT_TYPES:
        raise ValueError(
            f"part {part.role!r} has unknown output {part.output!r}; known: "
            f"{', '.join(sorted(OUTPUT_TYPES))}"
        )
    return OUTPUT_TYPES[part.output]


def _write_notes(
    deluge: Deluge, notes, section_start: int, length: int, report, label: str = ""
) -> int:
    written = 0
    consecutive = 0
    for note in notes:
        position = note.start - section_start
        if position < 0 or position >= length:
            continue
        # The firmware clamps a note that runs past the loop end, but clamping
        # to zero would be rejected, so drop those here.
        note_length = min(note.length, length - position)
        if note_length <= 0:
            continue
        if deluge.note(note.midi, position, note_length, note.velocity):
            written += 1
            consecutive = 0
        else:
            report.failures.append(f"note {note.midi}@{position}")
            consecutive += 1
            if consecutive >= MAX_CONSECUTIVE_FAILURES:
                raise DelugeError(
                    f"aborted while writing {label or 'a clip'}: "
                    f"{consecutive} notes rejected in a row after "
                    f"{report.notes + written} written. The clip length or "
                    f"selection is probably wrong - check the Deluge is on "
                    f"the session screen and not mid-recording."
                )
    return written


def _write_automation(deluge: Deluge, part: Part, section, report) -> int:
    written = 0
    for auto in part.automation:
        # Only the automation overlapping this section, rebased to clip time.
        if auto.start >= section.start_tick + section.length_ticks:
            continue
        if auto.start + auto.length <= section.start_tick:
            continue
        start = max(0, auto.start - section.start_tick)
        end = min(section.length_ticks, auto.start + auto.length - section.start_tick)
        if end - start <= 0:
            continue
        if deluge.automation(
            auto.param, start, end - start, auto.start_value, auto.end_value
        ):
            written += 1
        else:
            report.failures.append(f"automation {auto.param} @{start}")
    return written


def _chain(deluge: Deluge, sections, report: RenderReport, mode: str) -> None:
    """Set section repeat counts, which decides how the song behaves live.

    The firmware auto-advances to the next section only when a section's
    repeat count is >= 1 (`session.cpp`, `armNextSection`). So the repeat
    count is the difference between the two things you might want:

    - ``jam``  - every section set to infinite. Launching a section loops it
      until you launch another. You drive the arrangement from the grid.
    - ``song`` - every section plays once and hands over to the next, and the
      AI section loop wraps the end back to the start. The song plays itself.
    """
    if mode not in ("jam", "song"):
        raise ValueError(f"mode must be 'jam' or 'song', got {mode!r}")

    repeats = 0 if mode == "jam" else 1
    for number, _section in enumerate(sections):
        if not deluge.section_repeat(number, repeats=repeats):
            report.failures.append(f"section repeat {number}")

    if mode == "song":
        # Only meaningful when sections auto-advance; with infinite repeats
        # nothing ever reaches the end of a section to trigger the wrap.
        last = len(sections) - 1
        if not deluge.section_repeat(0, repeats=1, loop=True, start=0, end=last):
            report.failures.append("enable section loop")


def plan(
    song: Song, start_index: int = 0, template: dict[str, int] | None = None
) -> str:
    """Describe what `send` would do, without touching the hardware."""
    if song.arrangement is None:
        return "(no arrangement)"
    lines = []
    index = start_index
    template = template or {}
    emitted: set[str] = set()
    for number, section in enumerate(song.arrangement.sections):
        for part in song.parts:
            if not section.plays(part.role):
                continue
            count = len(part.in_bars(section.start_bar, section.end_bar))
            if not count:
                continue
            if part.role in emitted:
                action = "clone"
            elif part.role in template:
                action = f"clone from template [{template[part.role]}]"
            else:
                action = "create"
            emitted.add(part.role)
            lines.append(
                f"  [{index:>3}] section {number:<2} {section.name:<12} "
                f"{part.role:<7} {part.output:<5} {count:>4} notes  {action}"
            )
            index += 1
    total_bars = song.arrangement.bars
    lines.append(f"  {index - start_index} clips, {total_bars} bars total")
    return "\n".join(lines)
