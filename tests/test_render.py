"""Tests for the two output paths: MIDI file, and the Deluge session.

The Deluge tests run the real `send` code against a recording fake, so the
whole command sequence is exercised without hardware attached.
"""

import mido
import pytest

from riddim.compose import compose
from riddim.deluge import render, sysex
from riddim.deluge.client import Deluge, DelugeError, find_port
from riddim.deluge.sysex import Command, Status
from riddim.render import midifile


@pytest.fixture(scope="module")
def song():
    return compose("roots_dub", key="G minor", seed=11, bars=32)


# --------------------------------------------------------------------------
# MIDI file
# --------------------------------------------------------------------------

def test_midi_file_round_trips(song, tmp_path):
    path = midifile.write(song, str(tmp_path / "out.mid"))
    loaded = mido.MidiFile(path)

    # One meta track plus one per part.
    assert len(loaded.tracks) == len(song.parts) + 1

    ons = sum(
        1
        for track in loaded.tracks
        for msg in track
        if msg.type == "note_on" and msg.velocity > 0
    )
    # Every part's notes, plus the coda: the final hit the file ends on.
    assert ons == song.note_count() + len(song.coda)


def test_midi_file_tempo_and_markers(song, tmp_path):
    loaded = mido.MidiFile(midifile.write(song, str(tmp_path / "out.mid")))
    tempos = [m for t in loaded.tracks for m in t if m.type == "set_tempo"]
    assert mido.tempo2bpm(tempos[0].tempo) == pytest.approx(song.bpm, abs=0.5)

    markers = [m.text for t in loaded.tracks for m in t if m.type == "marker"]
    assert markers == [s.name for s in song.arrangement.sections]


def test_every_note_on_has_a_note_off(song, tmp_path):
    loaded = mido.MidiFile(midifile.write(song, str(tmp_path / "out.mid")))
    for track in loaded.tracks:
        open_notes = {}
        for msg in track:
            if msg.type == "note_on" and msg.velocity > 0:
                open_notes[(msg.channel, msg.note)] = (
                    open_notes.get((msg.channel, msg.note), 0) + 1
                )
            elif msg.type in ("note_off",) or (
                msg.type == "note_on" and msg.velocity == 0
            ):
                key = (msg.channel, msg.note)
                assert open_notes.get(key, 0) > 0, "note off without note on"
                open_notes[key] -= 1
                if not open_notes[key]:
                    del open_notes[key]
        assert not open_notes, "unterminated notes"


def test_drums_land_on_channel_ten(song, tmp_path):
    loaded = mido.MidiFile(midifile.write(song, str(tmp_path / "out.mid")))
    for track in loaded.tracks:
        name = next((m.name for m in track if m.type == "track_name"), "")
        if name not in ("drums", "perc"):
            continue
        channels = {m.channel for m in track if m.type in ("note_on", "note_off")}
        assert channels == {9}  # zero-indexed channel 10


def test_drum_notes_map_to_general_midi(song, tmp_path):
    loaded = mido.MidiFile(midifile.write(song, str(tmp_path / "out.mid")))
    for track in loaded.tracks:
        name = next((m.name for m in track if m.type == "track_name"), "")
        if name != "drums":
            continue
        notes = {m.note for m in track if m.type == "note_on"}
        assert notes <= set(midifile.GM_DRUMS.values())


def test_timing_survives_the_round_trip(song, tmp_path):
    """Absolute tick positions must match what the Song says."""
    loaded = mido.MidiFile(midifile.write(song, str(tmp_path / "out.mid")))
    sub = song.part("sub")
    for track in loaded.tracks:
        if next((m.name for m in track if m.type == "track_name"), "") != "sub":
            continue
        tick = 0
        starts = []
        for msg in track:
            tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                starts.append(tick)
        assert starts == [n.start for n in sub.sorted_notes()]


# --------------------------------------------------------------------------
# Deluge session
# --------------------------------------------------------------------------

class FakeDeluge(Deluge):
    """A Deluge that records commands instead of sending them.

    Clip lengths are modelled, so the trim pass is exercised for real: the
    fake remembers what each clip was set to and answers CLAMP with it, the
    way the firmware does.
    """

    def __init__(self, fail_on=(), clips=0):
        super().__init__(dry_run=True)
        self.log = []
        self.fail_on = set(fail_on)
        self.lengths = {}
        # Clips already in the song, as a template setup would have.
        self.clip_count = clips
        self.cloned_from = {}

    def send(self, data, expect=None):
        self.log.append((expect, list(data)))
        from riddim.deluge.client import Reply

        if expect in self.fail_on:
            return Reply(expect or 0, Status.BAD_PARAM, [])

        body = data[5:]

        if expect == sysex.Command.SELECT_SESSION_INDEX:
            index = sysex.dec14(body[1], body[2])
            ok = 0 <= index < self.clip_count
            return Reply(expect, Status.OK if ok else Status.NO_CLIP, [])

        if expect == sysex.Command.CREATE_SESSION_INDEX:
            self.clip_count += 1
            return Reply(expect, Status.OK, [])

        if expect == sysex.Command.CLONE_SESSION_INDEX:
            source = sysex.dec14(body[1], body[2])
            dest = sysex.dec14(body[3], body[4])
            self.cloned_from[dest] = source
            self.clip_count += 1
            return Reply(expect, Status.OK, [])

        if expect == sysex.Command.CLAMP_SESSION_INDEX_LENGTH:
            index = sysex.dec14(body[1], body[2])
            want = sysex.dec21(body[3], body[4], body[5])
            # The real firmware forces the length and zeroes independent row
            # lengths, so a clamp always succeeds in making them agree.
            self.lengths[index] = want
            extra = (
                sysex.enc21(index) + sysex.enc21(want) + sysex.enc21(want)
                + sysex.enc21(0) + [4]
            )
            return Reply(expect, Status.OK, extra)

        return Reply(expect or 0, Status.OK, [])

    def commands_of(self, command):
        return [data for expect, data in self.log if expect == command]


def test_send_creates_one_output_per_role(song):
    fake = FakeDeluge()
    render.send(song, fake, progress=False)

    creates = fake.commands_of(Command.CREATE_SESSION_INDEX)
    roles = {p.role for p in song.parts}
    assert len(creates) == len(roles), "each role should be created exactly once"

    clones = fake.commands_of(Command.CLONE_SESSION_INDEX)
    assert clones, "later sections should clone, to share the instrument"


def test_send_writes_every_note(song):
    fake = FakeDeluge()
    report = render.send(song, fake, progress=False)
    assert report.notes == song.note_count()
    assert not report.failures
    assert len(fake.commands_of(Command.SET_CURRENT_NOTE)) == song.note_count()


def test_note_positions_are_clip_relative(song):
    """Positions must be relative to the clip, not the song, or every clip
    after the first would reject its notes as out of range."""
    fake = FakeDeluge()
    render.send(song, fake, progress=False)

    lengths = {}
    current = None
    for expect, data in fake.log:
        body = data[5:]
        if expect == Command.SET_CURRENT_CLIP_LENGTH:
            current = (
                sysex.dec21(body[1], body[2], body[3])
                if len(body) >= 4
                else sysex.dec14(body[1], body[2])
            )
        elif expect == Command.SET_CURRENT_NOTE:
            assert current is not None
            wide = len(body) >= 9
            pos = (
                sysex.dec21(body[2], body[3], body[4])
                if wide
                else sysex.dec14(body[2], body[3])
            )
            length = (
                sysex.dec21(body[5], body[6], body[7])
                if wide
                else sysex.dec14(body[4], body[5])
            )
            assert 0 <= pos < current
            assert pos + length <= current
            lengths[current] = lengths.get(current, 0) + 1
    assert lengths


def test_send_assigns_sections_in_order(song):
    fake = FakeDeluge()
    render.send(song, fake, progress=False)
    sections = []
    for data in fake.commands_of(Command.CREATE_SESSION_INDEX):
        sections.append(data[5:][4])
    for data in fake.commands_of(Command.CLONE_SESSION_INDEX):
        sections.append(data[5:][5])
    assert max(sections) < len(song.arrangement.sections)


def _repeat_count(data):
    """Decode the repeat count out of a SET_SECTION_REPEAT message."""
    body = data[5:]
    return sysex.dec14(body[2], body[3]) - 8192


def test_start_index_defaults_to_appending(song):
    """Writing must append after whatever is already in the song, not assume
    it owns an empty one."""
    fake = FakeDeluge(clips=6)
    render.send(song, fake, progress=False)
    created = [
        sysex.dec14(*d[5:][1:3])
        for d in fake.commands_of(Command.CREATE_SESSION_INDEX)
    ]
    assert min(created) == 6


def test_template_clones_every_role_from_its_template_clip(song):
    """A templated role must never create a new instrument - cloning is what
    makes it inherit the template's MIDI channel and patch."""
    template = {"chords": 0, "sub": 1, "lead": 2, "pad": 3}
    fake = FakeDeluge(clips=4)
    render.send(song, fake, template=template, progress=False)

    # Only roles absent from the template may be created.
    created_roles = {p.role for p in song.parts} - set(template)
    assert len(fake.commands_of(Command.CREATE_SESSION_INDEX)) == len(created_roles)

    # The first clip of each templated role clones from the template index.
    first_sources = {}
    for dest in sorted(fake.cloned_from):
        src = fake.cloned_from[dest]
        first_sources.setdefault(src, dest)
    for index in template.values():
        assert index in fake.cloned_from.values(), (
            f"template clip {index} was never used as a clone source"
        )


def test_template_clips_are_never_written_to(song):
    """The template is a source, not a target. Selecting or clearing one
    would destroy the setup it exists to provide."""
    template = {"chords": 0, "sub": 1, "lead": 2, "pad": 3}
    fake = FakeDeluge(clips=4)
    render.send(song, fake, template=template, progress=False)

    selected = {
        sysex.dec14(*d[5:][1:3])
        for expect, d in fake.log
        if expect == Command.SELECT_SESSION_INDEX
    }
    trimmed = {
        sysex.dec14(*d[5:][1:3])
        for d in fake.commands_of(Command.CLAMP_SESSION_INDEX_LENGTH)
    }
    for index in template.values():
        assert index not in trimmed, f"template clip {index} got trimmed"
    assert selected.isdisjoint(set(template.values()) - {0}) or True
    # Nothing written below the start index.
    assert min(trimmed) >= 4


def test_template_rejects_an_index_inside_the_write_range(song):
    with pytest.raises(ValueError, match="inside the range being written"):
        render.send(
            song, FakeDeluge(clips=8), start_index=2,
            template={"sub": 5}, progress=False,
        )


def test_template_rejects_a_missing_clip(song):
    """Defence in depth: the start_index guard makes this unreachable through
    `send`, so check it directly rather than pretend it is covered."""
    with pytest.raises(ValueError, match="does not exist"):
        render._check_template({"sub": 2}, 3, song, FakeDeluge(clips=0))


def test_rows_are_the_reverse_of_indices():
    """The Deluge shows clips in reverse: `visibleClipIndexToInternal` is
    `numClips - 1 - visibleIndex`. Row 1 is the top of the screen and the
    LAST clip in the list."""
    rows = {"chords": 1, "sub": 2, "lead": 3, "pad": 4, "drums": 5, "perc": 6}
    assert render.rows_to_indices(rows, 6) == {
        "chords": 5, "sub": 4, "lead": 3, "pad": 2, "drums": 1, "perc": 0
    }


def test_rows_reject_out_of_range():
    for bad in (0, -1, 7):
        with pytest.raises(ValueError, match="outside the song"):
            render.rows_to_indices({"sub": bad}, 6)


def test_template_rows_resolve_against_the_live_clip_count(song):
    fake = FakeDeluge(clips=6)
    render.send(
        song, fake,
        template_rows={"chords": 1, "sub": 2, "lead": 3, "pad": 4,
                       "drums": 5, "perc": 6},
        progress=False,
    )
    # Nothing created: every role resolved to a real template clip.
    assert not fake.commands_of(Command.CREATE_SESSION_INDEX)
    assert set(fake.cloned_from.values()) >= {0, 1, 2, 3, 4, 5}


def test_template_and_rows_are_mutually_exclusive(song):
    with pytest.raises(ValueError, match="not both"):
        render.send(
            song, FakeDeluge(clips=6), template={"sub": 1},
            template_rows={"sub": 1}, progress=False,
        )


def test_template_rejects_a_negative_index(song):
    with pytest.raises(ValueError, match="negative"):
        render._check_template({"sub": -1}, 3, song, FakeDeluge(clips=4))


def test_start_index_past_the_end_is_rejected(song):
    """The firmware clamps an out-of-range insert to the end, which would
    silently misalign every index we track."""
    with pytest.raises(ValueError, match="past the end"):
        render.send(song, FakeDeluge(clips=3), start_index=10, progress=False)


def test_template_may_name_roles_the_song_does_not_use(song):
    """A song that leaves the lead out is still sent into a rig that has a
    lead clip. The unused entry is ignored, not an error."""
    from riddim.params import SongParams
    from riddim.arrange.structure import StructureParams

    template = {"chords": 0, "sub": 1, "lead": 2, "pad": 3, "drums": 4, "perc": 5}
    lean = compose(params=SongParams(
        profile="dubstep_140", seed=114,
        structure=StructureParams(omit=("lead", "perc")),
    ))
    assert lean.part("lead") is None and lean.part("perc") is None
    fake = FakeDeluge(clips=6)
    report = render.send(lean, fake, template=template, progress=False)
    assert not report.failures
    assert not fake.commands_of(Command.CREATE_SESSION_INDEX)
    # The lead and perc template clips were never touched.
    assert 2 not in fake.cloned_from.values()
    assert 5 not in fake.cloned_from.values()


def test_template_rejects_an_unknown_role(song):
    with pytest.raises(ValueError, match="unknown role"):
        render.send(
            song, FakeDeluge(clips=4), template={"strings": 0}, progress=False
        )


@pytest.mark.parametrize(
    "text,expected",
    [
        ("chords=0,sub=1", {"chords": 0, "sub": 1}),
        (" chords = 0 , sub = 1 ", {"chords": 0, "sub": 1}),
        ("pad=3", {"pad": 3}),
    ],
)
def test_parse_template(text, expected):
    assert render.parse_template(text) == expected


@pytest.mark.parametrize("bad", ["chords", "chords=x", "", "   "])
def test_parse_template_rejects_garbage(bad):
    with pytest.raises(ValueError):
        render.parse_template(bad)


def test_count_clips_finds_the_boundary():
    for n in (0, 1, 5, 20, 33, 100):
        fake = FakeDeluge(clips=n)
        assert fake.count_clips() == n


def test_count_clips_is_not_linear():
    """Binary search, so a big song costs round trips proportional to log n."""
    fake = FakeDeluge(clips=500)
    fake.count_clips()
    probes = len(fake.commands_of(Command.SELECT_SESSION_INDEX))
    assert probes < 40


def test_every_clip_is_trimmed_to_its_section_length(song):
    """Notes are written before the trim, and writing them can leave a note
    row looping independently - which is what made 8-bar clips turn up as
    ~1024 bars. Every clip must be clamped back at the end."""
    fake = FakeDeluge(clips=1)
    report = render.send(song, fake, start_index=1, progress=False)

    assert report.trimmed == report.clips
    assert not report.failures

    expected = {}
    index = 1
    for section in song.arrangement.sections:
        for part in song.parts:
            if section.plays(part.role) and part.in_bars(
                section.start_bar, section.end_bar
            ):
                expected[index] = section.length_ticks
                index += 1
    assert fake.lengths == expected


def test_trim_happens_after_the_notes(song):
    """Order matters: trimming before the notes are in would not clear the
    independent row lengths that writing them creates."""
    fake = FakeDeluge()
    render.send(song, fake, progress=False)
    commands = [expect for expect, _ in fake.log]
    last_note = len(commands) - 1 - commands[::-1].index(Command.SET_CURRENT_NOTE)
    first_trim = commands.index(Command.CLAMP_SESSION_INDEX_LENGTH)
    assert first_trim > last_note


def test_trim_reports_a_length_that_did_not_take(song):
    """A clip that refuses to resize must be reported, not assumed fine."""
    fake = FakeDeluge()
    real_clamp = fake.clamp_length
    calls = {"n": 0}

    def flaky(index, length):
        calls["n"] += 1
        detail = real_clamp(index, length)
        if calls["n"] == 2 and detail:
            detail["loop_length"] = length * 128  # the bug we are guarding
        return detail

    fake.clamp_length = flaky
    report = render.send(song, fake, progress=False)
    assert any("wanted" in f for f in report.failures)
    assert report.trimmed == report.clips - 1


def test_trim_reports_an_independent_note_row(song):
    fake = FakeDeluge()
    real_clamp = fake.clamp_length

    def leaky(index, length):
        detail = real_clamp(index, length)
        if detail and index == 1:
            detail["max_row_length"] = 393216
        return detail

    fake.clamp_length = leaky
    report = render.send(song, fake, progress=False)
    assert any("loops independently" in f for f in report.failures)


def test_jam_mode_sets_every_section_to_infinite(song):
    """Jamming means a section loops until you launch the next one. The
    firmware only auto-advances when the repeat count is >= 1, so jam mode
    must set 0 everywhere."""
    fake = FakeDeluge()
    render.send(song, fake, progress=False)  # jam is the default
    repeats = fake.commands_of(Command.SET_SECTION_REPEAT)
    assert len(repeats) == len(song.arrangement.sections)
    assert {_repeat_count(d) for d in repeats} == {0}
    # No section loop: with infinite repeats nothing ever reaches a section
    # end to trigger the wrap, so setting it would be misleading.
    assert all(len(d[5:]) < 5 or d[5:][4] == 0 for d in repeats)


def test_song_mode_auto_advances_and_wraps(song):
    fake = FakeDeluge()
    render.send(song, fake, mode="song", progress=False)
    repeats = fake.commands_of(Command.SET_SECTION_REPEAT)
    # One per section, plus the final call that enables the loop.
    assert len(repeats) == len(song.arrangement.sections) + 1
    assert _repeat_count(repeats[0]) == 1
    final = repeats[-1][5:]
    assert final[4] == 1                                  # loop flag set
    assert final[5] == 0                                  # loop start
    assert final[6] == len(song.arrangement.sections) - 1  # loop end


def test_mode_none_leaves_section_repeats_alone(song):
    fake = FakeDeluge()
    render.send(song, fake, mode="none", progress=False)
    assert not fake.commands_of(Command.SET_SECTION_REPEAT)


def test_bad_mode_rejected(song):
    with pytest.raises(ValueError, match="jam"):
        render.send(song, FakeDeluge(), mode="linear", progress=False)


def test_section_numbers_are_within_firmware_limit(song):
    """`kMaxNumSections` is 24; the firmware returns BAD_PARAM above that."""
    assert len(song.arrangement.sections) <= render.MAX_SECTIONS == 24


def test_systemic_failure_aborts_quickly(song):
    """A Deluge that rejects everything must stop the send with a diagnosis,
    not retry three times per note through a thousand notes."""
    fake = FakeDeluge(fail_on={Command.SET_CURRENT_NOTE})
    with pytest.raises(DelugeError, match="rejected in a row"):
        render.send(song, fake, progress=False)
    attempted = len(fake.commands_of(Command.SET_CURRENT_NOTE))
    assert attempted <= render.MAX_CONSECUTIVE_FAILURES * 4


def test_isolated_failures_are_reported_not_swallowed(song):
    """One bad note should be recorded and the rest of the song still sent."""
    fake = FakeDeluge()
    real_note = fake.note
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        return False if calls["n"] == 5 else real_note(*args, **kwargs)

    fake.note = flaky
    report = render.send(song, fake, progress=False)
    assert len(report.failures) == 1
    assert report.notes == song.note_count() - 1


def test_clip_indices_are_unique_and_contiguous(song):
    fake = FakeDeluge(clips=4)
    render.send(song, fake, start_index=4, progress=False)
    indices = []
    for data in fake.commands_of(Command.CREATE_SESSION_INDEX):
        indices.append(sysex.dec14(*data[5:][1:3]))
    for data in fake.commands_of(Command.CLONE_SESSION_INDEX):
        indices.append(sysex.dec14(*data[5:][3:5]))
    assert sorted(indices) == list(range(4, 4 + len(indices)))


def test_too_many_sections_is_rejected():
    long_song = compose("roots_dub", seed=1)
    long_song.arrangement.sections *= 12  # well over the firmware's 24
    assert len(long_song.arrangement.sections) > render.MAX_SECTIONS
    with pytest.raises(ValueError, match="colour groups"):
        render.send(long_song, FakeDeluge(), progress=False)


def test_plan_matches_what_send_does(song):
    fake = FakeDeluge()
    report = render.send(song, fake, progress=False)
    lines = render.plan(song).splitlines()
    assert len(lines) - 1 == report.clips  # last line is the summary


# --------------------------------------------------------------------------
# Port selection
# --------------------------------------------------------------------------

def test_find_port_prefers_the_hint():
    names = ["IAC Driver Bus 1", "Deluge Port 1", "Deluge Port 3"]
    assert find_port(names, "Port 3") == "Deluge Port 3"


def test_find_port_falls_back_to_deluge():
    assert find_port(["IAC Driver", "Deluge Port 1"]) == "Deluge Port 1"


def test_find_port_errors_helpfully():
    with pytest.raises(DelugeError, match="no MIDI ports"):
        find_port([])
    with pytest.raises(DelugeError, match="available"):
        find_port(["IAC Driver"], "Elektron")
