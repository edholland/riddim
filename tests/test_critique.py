"""The self-critic: the engine's knowledge run in reverse over a finished song.

The generated songs must pass the critic's invariants (mute map, registers,
monophony), and a song with one of those broken by hand must be called out
- that is what `--check` promises a maker.
"""

from riddim import critique
from riddim.compose import compose
from riddim.params import SongParams
from riddim.units import TICKS_PER_16TH, TICKS_PER_BAR


def _song(profile="roots_dub", seed=12):
    return compose(params=SongParams(profile=profile, seed=seed))


def test_generated_songs_keep_their_invariants():
    """Mute map, registers and the key are things the generators guarantee by
    construction; the critic must not find them broken in a composed song."""
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
    assert any(f.where == "sub" and "silent" in f.message
               and f.severity == "warning" for f in fs)


def test_critic_sees_a_note_out_of_register():
    from riddim.song import Note

    song = _song()
    sub = song.part("sub")
    n = sub.notes[0]
    sub.notes[0] = Note(n.midi + 24, n.start, n.length, n.velocity)
    fs = critique.critique(song)
    assert any(f.where == "sub" and "outside" in f.message and
               f.severity == "warning" for f in fs)


def test_critic_sees_a_build_that_never_climbs():
    song = _song(seed=5)
    builds = [s for s in song.arrangement.sections if s.name.startswith("build")]
    assert builds
    b = builds[0]
    b.ramp = 0.0
    b.bars = max(8, b.bars)
    fs = critique.critique(song)
    assert any(f.where == b.name and "never climbs" in f.message for f in fs)


def test_critic_sees_a_loud_hole():
    song = _song()
    holes = [s for s in song.arrangement.sections
             if s.name in ("dub_out", "breakdown")]
    assert holes
    h = holes[0]
    h.energy = 0.9
    fs = critique.critique(song)
    assert any(f.where == h.name and "loud" in f.message for f in fs)


def test_critic_sees_a_sub_that_never_confirms():
    """The downbeat says what the chord is. A fast-changing harmony with a
    bass that never puts a root under it is a warning, not a note."""
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
    assert any(f.where == "sub" and "confirms" in f.message
               and f.severity == "warning" for f in fs)


def test_cli_check_flag(capsys):
    from riddim.cli import main

    rc = main(["compose", "--profile", "roots_dub", "--seed", "3", "--check"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "check:" in out
