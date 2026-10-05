"""Tests for the UI server's state and serialisation - no sockets involved."""

import json

from riddim.compose import compose
from riddim.params import SongParams
from riddim.ui.server import State, options, song_to_dict


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
