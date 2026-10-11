"""Tests for the UI server's state and serialisation - no sockets involved."""

import json

from riddim.compose import compose
from riddim.params import SongParams
from conftest import clean_seed
from riddim.direct import Op
from riddim.ui.server import State, _changed, options, song_to_dict


def test_song_to_dict_is_json_and_complete():
    song = compose("dubstep_140", seed=3)
    d = song_to_dict(song)
    json.dumps(d)  # must serialise
    assert d["bpm"] == song.bpm and d["bars"] == song.bars
    assert set(d["parts"]) == {p.role for p in song.parts}
    for role, part in d["parts"].items():
        assert len(part["notes"]) == len(song.part(role).notes)
        assert all(len(n) == 4 for n in part["notes"])
    assert [s["name"] for s in d["sections"]] == [s.name for s in song.arrangement.sections]
    assert SongParams.from_dict(d["params"]).seed == song.seed


def test_state_recompose_from_edited_params():
    state = State(SongParams(profile="dubstep_140", seed=5))
    params = state.params.copy()
    # Edit the structure the way the page does: drop the last section.
    params.sections = params.sections[:-1]
    song = state.recompose(params)
    assert len(song.arrangement.sections) == len(params.sections)
    assert state.song is song


def test_state_regenerate_is_local():
    state = State(SongParams(profile="dubstep_140", seed=5))
    before = {p.role: p.sorted_notes() for p in state.song.parts}
    song = state.regenerate("chords")
    for p in song.parts:
        if p.role == "chords":
            assert p.sorted_notes() != before[p.role]
        else:
            assert p.sorted_notes() == before[p.role]


def test_state_regenerate_structure_and_harmony():
    state = State(SongParams(profile="roots_dub", seed=8))
    prog = state.song.harmony.progression.name
    secs = [s.name for s in state.song.arrangement.sections]
    changed_prog = any(
        state.regenerate("harmony").harmony.progression.name != prog for _ in range(4)
    )
    assert changed_prog
    changed_struct = any(
        [s.name for s in state.regenerate("structure").arrangement.sections] != secs
        for _ in range(6)
    )
    assert changed_struct


def test_reroll_keeps_what_it_is_told_to():
    state = State(SongParams(profile="dubstep_140", seed=5, key="G phrygian"))
    song = state.reroll({"key": "G phrygian"})
    assert song.scale.name == "phrygian" and song.scale.root % 12 == 7


def test_options_cover_every_profile():
    from riddim import profile as pm

    assert set(options()["profiles"]) == set(pm.PROFILES)


# -- the steering model -----------------------------------------------------
#
# A twist is a step on a trail; a roll (regenerate/reroll) starts a new take.
# Undo/redo walk the trail. The readback says what moved.



def _state():
    return State(SongParams(profile="dubstep_140", seed=17))


def test_mutate_is_a_step_on_the_trail():
    state = _state()
    before = state.params.copy()
    state.mutate([Op("bass.density", delta=0.1)])
    assert state.history and not state.future
    assert state.params is not before
    assert state.song is not None


def test_undo_redo_round_trip():
    state = _state()
    base = state.params.copy()
    state.mutate([Op("tension", 0.25), Op("bpm", 4)])
    state.undo()
    assert state.params.to_dict() == base.to_dict()
    state.redo()
    assert state.history and not state.future
    assert state.params.tension - base.tension > 0.2


def test_empty_ops_is_a_no_op():
    state = _state()
    song = state.mutate([])
    assert song is state.song
    assert not state.history


def test_regenerate_starts_a_new_take():
    state = _state()
    state.mutate([Op("tension", 0.3)])
    baseline_song = state.baseline
    state.regenerate("drums")
    # The take now starts at the regenerated song: the readback against it
    # is empty, because the baseline moved with it.
    assert state.baseline is state.song
    assert baseline_song is not state.song
    assert _changed(state, state.song)["knobs"] == []


def test_reroll_starts_a_new_take_with_new_seed():
    state = _state()
    old_seed = state.song.seed
    state.reroll({"key": "G phrygian"})
    assert state.baseline is state.song
    assert state.song.seed != old_seed
    assert state.song.scale.name == "phrygian"


def test_changed_readback_reports_moved_knobs():
    state = _state()
    state.mutate([Op("tension", 0.3)])
    changed = _changed(state, state.song)
    assert changed["knobs"] and changed["knobs"][0]["path"] == "tension"
    assert changed["notes"]


def test_key_steering_is_in_the_readback():
    state = _state()
    state.mutate([Op("key", to=-2)])
    changed = _changed(state, state.song)
    assert any(k["path"] == "key" for k in changed["knobs"])


def test_changed_reports_note_level_changes():
    """Counts alone cannot show a key change (same number of notes, every
    pitch moved), so the readback also counts notes added and removed per
    part - the picture the piano-roll shading is drawn from."""
    state = _state()
    state.mutate([Op("key", to=-2)])
    changed = _changed(state, state.song)
    for counts in changed["notes"].values():
        assert set(counts) == {"base", "now"}
    assert changed["note_changes"]["sub"]
    assert changed["note_changes"]["sub"]["added"] + \
        changed["note_changes"]["sub"]["removed"] > 0
    assert "drums" not in changed["note_changes"]


def test_song_payload_carries_checks_and_score():
    """The check panel redraws on every response, so every response carries
    the critic's view: one row per finding, and the score."""
    from riddim import critique

    song = compose("dubstep_140", seed=clean_seed())  # clean
    d = song_to_dict(song)
    assert d["checks"] == [] and d["score"] == 1.0

    broken = compose(params=SongParams(profile="roots_dub", seed=50, bpm=72))
    broken.arrangement.sections[3].ramp = 0.0   # a build that never climbs
    broken.arrangement.sections[2].energy = 0.9  # a loud hole
    d = song_to_dict(broken)
    assert d["score"] < 1.0
    assert any("never climbs" in c["message"] for c in d["checks"])
    assert any("loud" in c["message"] for c in d["checks"])
    # Every finding the panel draws has a fix to offer, and the serialised
    # shape the page's buttons read.
    for c in d["checks"]:
        assert {"where", "message", "severity", "fixes"} <= set(c)
        assert c["fixes"], c
        assert c["fixes"][0]["kind"] in ("twist", "roll")


def test_revise_is_a_no_op_when_clean():
    state = State(SongParams(profile="roots_dub", seed=5))  # clean
    song = state.song
    assert state.revise() is song
    assert not state.history


def test_revise_fixes_a_broken_take_and_stays_on_the_trail():
    """'fix all' runs the critic's loop through the take model: each applied
    twist is a step the maker can undo, and a step that would not improve the
    take is never pushed at all."""
    from riddim import critique

    state = State(SongParams(profile="roots_dub", seed=50, bpm=72))
    state.params.sections[3].ramp = 0.0     # build_2
    state.params.sections[2].energy = 0.9   # dub_out
    state.recompose(state.params)           # a step: the take is broken
    before = critique.score(state.song)
    assert before < 1.0

    state.revise()
    assert critique.score(state.song) == 1.0
    # Recompose (1) plus one step per applied fix (3) - bounded by max_steps.
    assert 1 < len(state.history) <= 1 + 8
    # Undoing the last fix brings a finding back.
    state.undo()
    assert critique.score(state.song) < 1.0


def test_song_payload_carries_the_readback_shape():
    state = _state()
    state.mutate([Op("bpm", 4)])
    d = song_to_dict(state.song, state)
    assert all(set(c) == {"base", "now"} for c in d["changed"]["notes"].values())
    assert "note_changes" in d["changed"]
    # The section editor's harmony and ramp fields ride along.
    for s in d["sections"]:
        assert "progression" in s and "ramp" in s


def test_history_is_bounded():
    state = _state()
    for _ in range(70):
        state.mutate([Op("tension", 0.01)])
    assert len(state.history) <= 60
