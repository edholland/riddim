"""The model lead, arranged by riddim - pinned with a stand-in for the
model, so none of this needs torch.

The stand-in answers like the worker does: seconds in, seconds out. What is
pinned is riddim's side of the bargain: only the drops' lead changes, the
hook comes back, an answer stays an answer, a restated hook sits on its
chords, and riddim's rhythm is kept when the model only picks pitches.
"""

import random

import pytest

from riddim import model_lead as ml
from riddim.compose import compose
from riddim.song import Note
from riddim.theory import harmony
from riddim.units import TICKS_PER_16TH, TICKS_PER_BAR, TICKS_PER_BEAT


class FakeWriter:
    """Returns scale tones in the requested register: on the given rhythm,
    or on the eighths, seeded so drafts differ."""

    model = "fake"

    def __init__(self):
        self.requests = []

    def request(self, req):
        self.requests.append(req)
        rng = random.Random(req["seed"])
        pitches = [p for p in range(req["lo"], req["hi"] + 1) if p % 12 in req["scale"]]
        if req.get("rhythm") is not None:
            times = req["rhythm"]
        else:
            step = req["min_gap"] or 0.2
            times, t = [], req["start"]
            while t < req["end"]:
                if rng.random() < 0.6:
                    times.append([t, step])
                t += step
        return {"notes": [[s, d, rng.choice(pitches)] for s, d in times]}


def _drop_lead(song, sec):
    return [n for n in song.part("lead").notes
            if sec.start_tick <= n.start < sec.start_tick + sec.length_ticks]


@pytest.fixture
def song():
    return compose(profile="dubstep_140", seed=11, live=True)


@pytest.mark.parametrize("style", ml.STYLES)
def test_only_the_drops_lead_changes(song, style):
    out = ml.write(song, FakeWriter(), style=style)
    for part in song.parts:
        if part.role != "lead":
            assert out.part(part.role).notes == part.notes
    drops = ml.drops(song)
    assert drops

    def outside(s):
        return [n for n in s.part("lead").notes
                if not any(d.start_tick <= n.start < d.start_tick + d.length_ticks for d in drops)]
    assert outside(out) == outside(song)
    # the input song is untouched
    assert song.part("lead").notes == compose(params=song.params, live=True).part("lead").notes


@pytest.mark.parametrize("style", ml.STYLES)
def test_the_hook_comes_back(song, style):
    out = ml.write(song, FakeWriter(), style=style)
    d1, d2 = ml.drops(out)[:2]
    phrase = ml.PHRASE_BARS * TICKS_PER_BAR

    def bars(sec, k):
        a = sec.start_tick + k * phrase
        return [(n.start - a, n.midi) for n in out.part("lead").notes if a <= n.start < a + phrase]
    hook = bars(d1, 0)
    assert hook
    # AABA: phrase 2 and 4 restate it (bent onto its chords at most a step or two)
    for sec, k in ((d1, 1), (d1, 3), (d2, 0), (d2, 1)):
        again = bars(sec, k)
        assert [t for t, _ in again] == [t for t, _ in hook]
        assert all(abs(a - b) <= 4 for (_, a), (_, b) in zip(again, hook))


def test_rhythm_style_keeps_riddims_onsets(song):
    out = ml.write(song, FakeWriter(), style="rhythm")
    sec = ml.drops(song)[0]
    phrase = ml.PHRASE_BARS * TICKS_PER_BAR
    first = [n.start for n in _drop_lead(song, sec) if n.start < sec.start_tick + phrase]
    mine = [n.start for n in _drop_lead(out, sec) if n.start < sec.start_tick + phrase]
    assert mine == sorted(set(first))


def test_lead_stays_in_key_register_and_on_grid(song):
    out = ml.write(song, FakeWriter(), style="free")
    scale = set(song.scale.pitch_classes)
    lo, hi = song.part("lead").range()
    for sec in ml.drops(out):
        for n in _drop_lead(out, sec):
            assert n.midi % 12 in scale
            assert lo - 2 <= n.midi <= hi + 2
            assert n.start % TICKS_PER_16TH == 0
            assert n.end <= sec.start_tick + sec.length_ticks


def test_seconds_round_trip_through_the_requests(song):
    w = FakeWriter()
    ml.write(song, w, style="free")
    for req in w.requests:
        assert 0 < req["start"] < req["end"] <= ml.MAX_SECONDS
        # the band is heard before and during the window, never past its end
        assert max(e[2] for e in req["events"]) < req["end"]
        assert all(e[0] != ml.LEAD_INSTR or e[2] < req["start"] for e in req["events"])


def test_takes_differ_and_are_repeatable(song):
    a = ml.write(song, FakeWriter(), take=0).part("lead").notes
    b = ml.write(song, FakeWriter(), take=1).part("lead").notes
    assert a != b
    assert ml.write(song, FakeWriter(), take=0).part("lead").notes == a


def test_phrase_score_prefers_a_tune_to_a_pedal(song):
    t0 = ml.drops(song)[0].start_tick
    ev = song.harmony.at(t0)
    root = min(p for p in range(70, 90) if p % 12 == ev.chord.root_pc)
    pedal = [Note(root, t0 + i * TICKS_PER_16TH * 2, 24) for i in range(32)]
    tones = sorted(p for p in range(root, root + 13) if p % 12 in ev.chord.pitch_classes())
    tune = [Note(tones[i % len(tones)], t0 + bar * TICKS_PER_BAR + beat * TICKS_PER_BEAT, 48)
            for bar in range(4) for i, beat in enumerate((0, 1, 2) if bar % 2 == 0 else (0, 2))]
    assert ml.phrase_score(tune, song, t0, 4) > ml.phrase_score(pedal, song, t0, 4)
    assert ml.phrase_score([], song, t0, 4) == 0.0


def test_bend_moves_only_strong_notes_that_clash(song):
    t0 = ml.drops(song)[0].start_tick
    chord = song.harmony.at(t0).chord
    clash = next(p for p in range(72, 96) if not harmony.sits_on(p, chord)
                 and p % 12 in set(song.scale.pitch_classes))
    notes = [Note(clash, t0, 48), Note(clash, t0 + TICKS_PER_16TH, 24)]
    bent = ml.bend(notes, song, 70, 98)
    assert harmony.sits_on(bent[0].midi, chord) and abs(bent[0].midi - clash) <= 4
    assert bent[1] == notes[1]                      # off the beat: a passing note, kept


def test_form_shapes():
    assert ml._form(1) == "A"
    assert ml._form(2) == "AB"
    assert ml._form(4) == "AABA"
    assert ml._form(8) == "AABAAABA"


def test_unavailable_names_what_is_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("RIDDIM_ML_PYTHON", str(tmp_path / "nope"))
    assert "RIDDIM_ML_PYTHON" in ml.unavailable()


def test_ui_model_lead_is_a_step_undo_brings_riddim_back(monkeypatch):
    from riddim.ui.server import State

    fake = FakeWriter()
    monkeypatch.setattr(ml, "unavailable", lambda: None)
    monkeypatch.setattr(ml, "shared", lambda: fake)
    st = State(compose(profile="dubstep_140", seed=11, live=True).params, live=True)
    riddim_lead = list(st.song.part("lead").notes)
    st.model_lead("rhythm")
    assert st.song.part("lead").notes != riddim_lead
    assert fake.requests
    st.undo()
    assert st.song.part("lead").notes == riddim_lead
