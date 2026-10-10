"""The self-critic: the engine's knowledge run in reverse over a finished song.

The generated songs must pass the critic's invariants (mute map, registers,
monophony), and a song with one of those broken by hand must be called out -
that is what `--check` promises a maker. Each finding also carries the change
that would cure it: a twist for what a knob can fix, a re-roll for what only
a new take can, and `revise` walks the loop until the song is clean.
"""

import pytest

from riddim import critique
from riddim.compose import compose
from riddim.params import SongParams
from riddim.units import TICKS_PER_16TH, TICKS_PER_BAR


def _song(profile="roots_dub", seed=12):
    return compose(params=SongParams(profile=profile, seed=seed))


def test_generated_songs_keep_their_invariants():
    """Mute map, registers and monophony are things the generators guarantee
    by construction; the critic must not find them broken in a composed song.
    (Musical judgments like the sub confirming the root are not invariants -
    some seeds get caught by those, and that is the score working, not a bug.)"""
    for seed in (1, 12, 40):
        song = _song(seed=seed)
        for f in critique.critique(song):
            assert "silent there" not in f.message, (seed, f)
            assert "outside" not in f.message, (seed, f)
            assert "overlapping" not in f.message, (seed, f)


def test_critic_sees_a_silent_section():
    song = _song()
    sub = song.part("sub")
    sec = max((s for s in song.arrangement.sections if s.plays("sub")),
              key=lambda s: s.energy)
    sub.notes = [
        n for n in sub.notes
        if not sec.start_bar * TICKS_PER_BAR <= n.start < sec.end_bar * TICKS_PER_BAR
    ]
    assert not sub.in_bars(sec.start_bar, sec.end_bar)
    fs = critique.critique(song)
    f = next(f for f in fs if f.where == "sub" and "silent" in f.message)
    assert f.severity == "warning"
    # No knob can guarantee emission, so the fix is a take of the part.
    assert f.fixes and f.fixes[0].kind == "roll" and f.fixes[0].part == "sub"


def test_critic_sees_a_note_out_of_register():
    from riddim.song import Note

    song = _song()
    sub = song.part("sub")
    n = sub.notes[0]
    sub.notes[0] = Note(n.midi + 24, n.start, n.length, n.velocity)
    fs = critique.critique(song)
    f = next(f for f in fs if f.where == "sub" and "outside" in f.message
             and f.severity == "warning")
    # A register window is not a knob; a new take stays in its window.
    assert f.fixes and f.fixes[0].kind == "roll" and f.fixes[0].part == "sub"


def test_critic_sees_a_build_that_never_climbs():
    song = _song(seed=5)
    builds = [s for s in song.arrangement.sections if s.name.startswith("build")]
    assert builds
    b = builds[0]
    b.ramp = 0.0
    b.bars = max(8, b.bars)
    fs = critique.critique(song)
    f = next(f for f in fs if f.where == b.name and "never climbs" in f.message)
    i = song.arrangement.sections.index(b)
    fx = f.fixes[0]
    assert fx.kind == "twist"
    assert fx.ops[0].path == f"sections.{i}.ramp" and float(fx.ops[0].to) == 0.6


def test_critic_sees_a_loud_hole():
    song = _song()
    holes = [s for s in song.arrangement.sections
             if s.name in ("dub_out", "breakdown")]
    assert holes
    h = holes[0]
    h.energy = 0.9
    fs = critique.critique(song)
    f = next(f for f in fs if f.where == h.name and "loud" in f.message)
    i = song.arrangement.sections.index(h)
    fx = f.fixes[0]
    assert fx.kind == "twist"
    assert fx.ops[0].path == f"sections.{i}.energy" and float(fx.ops[0].to) == 0.3


def test_critic_sees_a_sub_that_never_confirms():
    """The downbeat says what the chord is. A fast-changing harmony with a
    bass that never puts a root under it is a warning - and the twist that
    cures it is the harmonic rhythm itself: slow the chords so the root can
    get under them."""
    from riddim import profile as profile_mod
    from riddim.song import Note

    song = _song(profile="dubstep_140", seed=5)
    sub = song.part("sub")
    bars = song.arrangement.bars
    roots = {
        song.harmony.at(b * TICKS_PER_BAR).chord.root_pc
        for b in range(bars) if song.harmony.at(b * TICKS_PER_BAR) is not None
    }
    avoid = next(pc for pc in range(12) if pc not in roots)
    window = profile_mod.register_for(
        profile_mod.get(song.params.profile), "sub")
    midi = next(
        m for k in range(-3, 5)
        for m in (avoid + 12 * k,) if window.low <= m <= window.high
    )
    playing = {b for b in range(bars) if sub.in_bars(b, b + 1)}
    assert len(playing) >= 8
    sub.notes = [
        Note(midi, b * TICKS_PER_BAR, TICKS_PER_16TH * 4, 100)
        for b in sorted(playing)
    ]
    song.params.bars_per_chord = 1
    fs = critique.critique(song)
    f = next(f for f in fs if f.where == "sub" and "confirms" in f.message)
    assert f.severity == "warning"
    fx = f.fixes[0]
    assert fx.kind == "twist"
    assert fx.ops[0].path == "bars_per_chord" and float(fx.ops[0].to) == 2.0
    # A slow harmony with a wandering figure is a note, and only a take cures it.
    song.params.bars_per_chord = 2
    f = next(f for f in critique.critique(song)
             if f.where == "sub" and "confirms" in f.message)
    assert f.severity == "note"
    assert f.fixes[0].kind == "roll" and f.fixes[0].part == "sub"


# --------------------------------------------------------------------------
# The score
# --------------------------------------------------------------------------

def test_score_is_one_when_clean_and_penalties_findings():
    """A warning is something likely to sound wrong; a note is something worth
    knowing. The score is the batch's ranking and the loop's progress bar."""
    song = _song()  # roots seed 12: the critic finds nothing
    assert critique.score(song) == 1.0

    holes = [s for s in song.arrangement.sections
             if s.name in ("dub_out", "breakdown")]
    holes[0].energy = 0.6  # a loud hole: a note (below 0.85, so not also a "peak")
    assert critique.score(song) == pytest.approx(1.0 - critique.NOTE_WEIGHT)

    from riddim.song import Note

    n = song.part("sub").notes[0]
    song.part("sub").notes[0] = Note(n.midi + 24, n.start, n.length, n.velocity)
    # one note outside the register: a warning, worth twice a note
    assert critique.score(song) == pytest.approx(
        1.0 - critique.NOTE_WEIGHT - critique.WARN_WEIGHT)


def test_score_stays_in_range():
    for profile in ("roots_dub", "steppers_dub", "dubstep_140", "dubby_halftime"):
        for seed in (1, 5, 12, 23, 40):
            s = critique.score(_song(profile=profile, seed=seed))
            assert 0.0 <= s <= 1.0, (profile, seed, s)


# --------------------------------------------------------------------------
# The revision loop
# --------------------------------------------------------------------------

def test_revise_is_a_no_op_on_a_clean_song():
    song = _song(seed=5)  # clean
    out, steps = critique.revise(song)
    assert out is song and steps == []


def test_revise_fixes_a_song_broken_through_params():
    """The break must be in the *params* - what a maker would actually twist -
    so that the recompose inside revise sees it, cures it, and keeps it cured."""
    song = _song(seed=23)
    p = song.params
    p.sections[3].ramp = 0.0    # build_2: a plateau wearing a build's name
    p.sections[2].energy = 0.9  # dub_out: a loud hole
    broken = compose(params=p)
    before = critique.score(broken)
    assert 0.0 < before < 1.0

    fixed, steps = critique.revise(broken)
    assert critique.score(fixed) == 1.0
    assert critique.critique(fixed) == []
    assert len(steps) <= 8
    assert all(st.applied for st in steps)
    assert all(st.fix.kind == "twist" for st in steps)  # plain numbers: twists
    assert all(st.score_after > st.score_before for st in steps)
    # The broken values are actually gone from the fixed song's params.
    assert fixed.params.sections[2].energy < 0.5
    assert fixed.params.sections[3].ramp > 0.0


def test_revise_respects_max_steps():
    song = _song(seed=23)
    p = song.params
    p.sections[3].ramp = 0.0
    p.sections[2].energy = 0.9
    broken = compose(params=p)
    fixed, steps = critique.revise(broken, max_steps=2)
    assert len(steps) == 2
    assert critique.score(fixed) < 1.0  # one finding still to fix


def test_revise_stops_at_a_wall_and_reports_it():
    """A fresh compose whose bass figure wanders under a fast harmony: the
    harmonic-rhythm twist helps, the figure itself needs a take, the take does
    not, and the loop says so and stops - score strictly up, never in a loop."""
    song = _song(seed=3)
    before = critique.score(song)
    fixed, steps = critique.revise(song)
    assert critique.score(fixed) > before
    assert steps[0].fix.kind == "twist"
    assert steps[-1].applied is False
    assert critique.score(fixed) == pytest.approx(before + 0.1)


def test_next_fix_prefers_warnings_and_twists():
    song = _song(seed=3)  # a sub warning and an energy-curve note
    f, fix = critique.next_fix(song)
    assert f.severity == "warning" and fix.kind == "twist"
    # On a clean song there is nothing to fix.
    assert critique.next_fix(_song(seed=5)) is None


def test_revise_leaves_a_hand_built_song_alone():
    """A song with no params has no steering surface: report it, do not touch it."""
    song = _song()
    song.params = None
    out, steps = critique.revise(song)
    assert out is song and steps == []
