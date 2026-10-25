"""Writing to the Deluge fast: notes pipelined, refusals not retried, a
missing kit row reported once, progress for the UI's bar."""

import pytest

from riddim.compose import compose
from riddim.deluge import render, sysex
from riddim.deluge.client import Deluge
from riddim.deluge.sysex import Command, Status

from test_render import FakeDeluge


class _Msg:
    def __init__(self, data):
        self.type = "sysex"
        self.data = data


class FakePorts:
    """A Deluge on the other end of a MIDI pair: each SysEx it reads gets its
    reply, in order. `refuse` maps a note to a status; `lose` drops the reply
    to the nth note message; `window_seen` records the most messages ever in
    flight."""

    def __init__(self, refuse=None, lose=(), present=None):
        self.pending = []
        self.refuse = refuse or {}
        self.lose = set(lose)
        self.present = present            # notes already in the clip
        self.notes_seen = 0
        self.in_flight = 0
        self.window_seen = 0
        self.written = []

    # output port
    def send(self, msg):
        body = list(msg.data)[5:]
        cmd = body[0]
        status = Status.OK
        if cmd == Command.SET_CURRENT_NOTE:
            self.notes_seen += 1
            note = body[1]
            key = (note, sysex.dec14(body[2], body[3]))
            if note in self.refuse:
                status = self.refuse[note]
            elif self.present is not None and key in self.present:
                status = Status.ADD_FAILED
            else:
                self.written.append(key)
                if self.present is not None:
                    self.present.add(key)
            if self.notes_seen in self.lose:
                return
        self.in_flight += 1
        self.window_seen = max(self.window_seen, self.in_flight)
        self.pending.append(_Msg(sysex.HEADER + [cmd, int(status)]))

    # input port
    def iter_pending(self):
        while self.pending:
            self.in_flight -= 1
            yield self.pending.pop(0)

    def close(self):
        pass


def _deluge(ports, window=16):
    d = Deluge(window=window, timeout=0.05)
    d._out = d._in = ports
    return d


def _notes(n, midi=36):
    return [(midi, i * 24, 12, 100) for i in range(n)]


def test_pipelined_notes_land_in_order_with_their_statuses():
    ports = FakePorts(refuse={40: Status.NO_KIT_ROW})
    notes = _notes(50) + _notes(5, midi=40) + _notes(30, midi=37)
    out = _deluge(ports).write_notes(notes)
    assert out == [Status.OK] * 50 + [Status.NO_KIT_ROW] * 5 + [Status.OK] * 30
    assert ports.window_seen <= 16


def test_a_lost_reply_falls_back_to_one_at_a_time():
    """The reply to the 20th note never comes: the rest are confirmed one by
    one, and a note that already landed (ADD_FAILED on the resend) counts."""
    ports = FakePorts(lose={20}, present=set())
    notes = _notes(40)
    out = _deluge(ports).write_notes(notes)
    assert all(s == Status.OK for s in out)
    assert sorted(set(ports.written)) == sorted((m, p) for m, p, _, _ in notes)


def test_refusals_are_not_retried():
    ports = FakePorts(refuse={44: Status.NO_KIT_ROW})
    d = _deluge(ports, window=1)
    assert not d.note(44, 0, 12, 100)
    assert ports.notes_seen == 1          # was three tries and two sleeps


def test_missing_kit_rows_are_one_line_each():
    song = compose("dubstep_140", seed=3)
    fake = FakeDeluge()
    hands = set(range(44, 51))

    def send(data, expect=None):
        body = data[5:]
        if expect == Command.SET_CURRENT_NOTE and body[1] in hands:
            from riddim.deluge.client import Reply
            fake.log.append((expect, list(data)))
            return Reply(expect, Status.NO_KIT_ROW, [])
        return FakeDeluge.send(fake, data, expect)

    fake.send = send
    report = render.send(song, fake, progress=False)
    lines = [f for f in report.failures if "no row for note" in f]
    perc_clips = sum(1 for s in song.arrangement.sections
                     if s.plays("perc") and any(n.midi in hands for n in
                                                song.part("perc").in_bars(s.start_bar, s.end_bar)))
    assert lines and len(lines) <= 7 * perc_clips
    assert not [f for f in report.failures if f.startswith("note ")]


def test_progress_rises_to_the_total():
    song = compose("roots_dub", seed=3)
    seen = []
    render.send(song, FakeDeluge(), progress=False,
                on_progress=lambda d, t, label: seen.append((d, t)))
    assert seen
    total = seen[0][1]
    assert total == song.note_count()
    assert all(b[0] >= a[0] for a, b in zip(seen, seen[1:]))
    assert seen[-1][0] == total
