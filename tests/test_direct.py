"""Steering: directed change on an already-composed song.

These pin the contract the UI rests on: a directed change moves knobs and
keeps every seed frozen, so the result is a small step from the original
- not a new roll - and the readback says exactly which knobs moved and
which parts changed.
"""

import pytest

from riddim.compose import compose
from riddim.direct import (
    INTENTS,
    Op,
    Session,
    apply_ops,
    direct,
    diff,
    knobs,
    mutate,
    read_knob,
    set_knob,
    song_diff,
)
from riddim.params import SongParams


def _song():
    return compose(params=SongParams(profile="dubstep_140", seed=11))


# -- ops and apply_ops ----------------------------------------------------

def test_ops_apply_to_a_copy():
    s0 = _song()
    p1 = apply_ops(s0.params, [Op("tension", 0.3)])
    assert p1 is not s0.params
    assert p1.tension - s0.params.tension == pytest.approx(0.3)
    # The caller's params are untouched.
    assert s0.params.tension != p1.tension


def test_op_absolute_target():
    p1 = apply_ops(_song().params, [Op("tension", to=0.9)])
    assert p1.tension == 0.9


def test_key_op_shifts_tonic_keeps_mode():
    p = SongParams(profile="dubstep_140", seed=1, key="G phrygian")
    p1 = apply_ops(p, [Op("key", to=-2)])
    assert p1.key == "F phrygian"
    p2 = apply_ops(p, [Op("key", to="C phrygian")])
    assert p2.key == "C phrygian"


def test_mutate_keeps_every_seed():
    s0 = _song()
    s1 = mutate(s0, "denser")
    assert s1.seed == s0.seed
    assert s1.params.seeds == s0.params.seeds
    # Denser touches drums and bass only - the chords are byte-identical.
    assert s1.part("chords").notes == s0.part("chords").notes


# -- word intents ----------------------------------------------------------

def test_intent_lookup_and_aliases():
    assert direct("busier") == INTENTS["busier"]
    assert direct("sync") == INTENTS["more_syncopated"]
    assert direct("dark") == INTENTS["darker"]


def test_unknown_intent_raises_with_options():
    with pytest.raises(ValueError) as exc:
        direct("zzz")
    assert "did not understand" in str(exc.value)
    assert "busier" in str(exc.value)


def test_phrase_parsing():
    assert direct("up a third") == [Op("key", to=2)]
    assert direct("down a fifth") == [Op("key", to=-4)]
    assert direct("faster") == [Op("bpm", delta=2)]
    assert direct("slower") == [Op("bpm", delta=-2)]
    assert direct("longer") == [Op("_total_bars", delta=16)]
    assert direct("shorter") == [Op("_total_bars", delta=-16)]


def test_phrase_parsing_with_numbers():
    """'by N' says how many written bpm or bars; the sign comes from the verb.
    A double of the number, or a flipped sign, is a change nobody asked for."""
    assert direct("faster by 4") == [Op("bpm", delta=4)]
    assert direct("slower by 4") == [Op("bpm", delta=-4)]
    assert direct("shorter by 16") == [Op("_total_bars", delta=-16)]
    assert direct("longer by 16") == [Op("_total_bars", delta=16)]
    assert direct("shorter by 8 bars") == [Op("_total_bars", delta=-8)]


def test_mode_intent_keeps_the_tonic():
    """The mode is a word like any other; it moves the mode, not the key."""
    assert direct("phrygian") == [Op("mode", to="phrygian")]
    assert direct("dorian") == [Op("mode", to="dorian")]
    assert direct("phrygian dominant") == [Op("mode", to="phrygian_dominant")]
    p = SongParams(profile="dubstep_140", seed=1, key="G dorian")
    assert apply_ops(p, [Op("mode", to="phrygian")]).key == "G phrygian"
    assert apply_ops(p, INTENTS["minor"]).key == "G minor"
    with pytest.raises(ValueError):
        apply_ops(p, [Op("mode", to="locrian")])


# -- knobs: the discoverable view ------------------------------------------

def test_knobs_cover_the_song_and_round_trip():
    s0 = _song()
    ks = knobs(s0.params)
    assert "bass.density" in ks and "key" in ks and "bpm" in ks
    for k in ks.values():
        assert k.low <= k.value <= k.high
    # Write a knob on a copy and read it back.
    p = s0.params.copy()
    set_knob(p, "bass.density", 0.5)
    assert read_knob(p, "bass.density") == 0.5
    assert read_knob(s0.params, "bass.density") != 0.5


# -- readback --------------------------------------------------------------

def test_diff_orders_by_magnitude_and_sees_nothing_when_unchanged():
    p = _song().params
    assert diff(p, p.copy()) == []
    q = apply_ops(p, [Op("tension", 0.3), Op("swing", 0.05)])
    deltas = diff(p, q)
    assert [d.path for d in deltas] == ["tension", "swing"]


def test_song_diff_reports_which_parts_moved():
    s0 = _song()
    s1 = compose(apply_ops(s0.params, INTENTS["busier"]))
    d = song_diff(s0, s1)
    # Busier is drums + bass; nothing else moved.
    assert d.moved_parts == ["drums", "sub"]
    assert d.total_notes > 0
    assert d.delta_notes > 0
    assert "drums" in d.summary()


def test_song_diff_sees_a_key_change():
    """A key change transposes every pitched note and keeps every count the
    same, so a count-based diff would report 'moved nothing' - a readback
    lying to the maker. The diff compares notes, so it sees the move."""
    s0 = _song()
    s1 = compose(apply_ops(s0.params, [Op("key", to=-2)]))
    d = song_diff(s0, s1)
    assert "sub" in d.moved_parts
    assert "drums" not in d.moved_parts


# -- the trail -------------------------------------------------------------

def test_session_step_undo_and_revert():
    s0 = _song()
    sess = Session(s0)
    assert sess.current.label == "baseline"

    st = sess.step([Op("tension", 0.25)])
    assert len(sess.steps) == 2
    assert [d.path for d in st.delta] == ["tension"]

    sess.step([Op("bass.density", 0.1)])
    sess.step("busier")
    assert len(sess.steps) == 4

    sess.undo()
    assert len(sess.steps) == 3
    sess.revert_to(1)
    assert len(sess.steps) == 2
    # Back at the first step: nothing differs from the baseline except that
    # one knob.
    assert [d.path for d in diff(sess.baseline.params, sess.params)] == ["tension"]


def test_session_undo_is_exact():
    s0 = _song()
    sess = Session(s0)
    sess.step([Op("tension", 0.25)])
    sess.undo()
    assert diff(sess.baseline.params, sess.params) == []
    # Seeds frozen means the recomposed song is byte-identical to the
    # baseline: same notes in every part.
    for pa, pb in zip(s0.parts, sess.current.song.parts):
        assert pa.notes == pb.notes


def test_session_attempt_and_report():
    sess = Session(_song())
    sess.attempt("busier")
    sess.attempt("down a third")
    report = sess.report()
    assert "baseline" in report
    assert "key" in report and "drums.kick_density" in report
    # Compare against the baseline covers the whole walk.
    d = sess.compare()
    assert d.moved_parts


def test_session_key_move_is_legible():
    """The key is a first-class knob: steering it reports one knob delta and
    the part-level readback shows the pitched parts moving, the kit not."""
    sess = Session(_song())
    st = sess.attempt("down a third")
    assert [d.path for d in st.delta] == ["key"]
    assert "sub" in st.song_diff.moved_parts
    assert "drums" not in st.song_diff.moved_parts


def test_session_compare_baseline_untouched_by_later_steps():
    sess = Session(_song())
    base_key = sess.baseline.params.key
    sess.attempt("down a third")
    sess.attempt("busier")
    assert sess.baseline.params.key == base_key
    assert sess.current.params.key != base_key