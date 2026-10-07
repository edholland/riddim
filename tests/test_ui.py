"""Tests for the UI server's state and serialisation - no sockets involved."""

import json

from riddim.compose import compose
from riddim.params import SongParams
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


def test_history_is_bounded():
    state = _state()
    for _ in range(70):
        state.mutate([Op("tension", 0.01)])
    assert len(state.history) <= 60
