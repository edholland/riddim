"""Writing like a person: the tells, and the economy that answers them.

The tells (riddim/tells.py) are where a producer who plays this music would
hear a machine. These tests hold the engine to the economy that keeps them
quiet - a pentatonic hook, a hook played the same way every time it comes
round, a 140 drop on one or two chords with a sub that sits on few notes -
and hold a batch of each family to a ceiling, so the giveaways cannot creep
back unnoticed.
"""

import random

import pytest

from riddim import tells
from riddim.compose import compose
from riddim.theory import hook
from riddim.theory.scale import Scale
from riddim.units import TICKS_PER_BAR

FAMILIES = ("roots_dub", "steppers_dub", "dubby_halftime", "uk_garage",
            "dark_garage", "dubstep_140")


def test_tells_report_every_measure():
    song = compose(profile="dubstep_140", seed=101)
    names = {t.name for t in tells.tells(song)}
    assert {"lead_vocab", "lead_range", "lead_density", "lead_restated",
            "drop_chords", "sub_extra", "sub_on_root"} <= names
    assert tells.count(song) == sum(not t.ok for t in tells.tells(song))
    assert all(str(t) for t in tells.tells(song))


def test_the_octave_flick_is_not_range():
    """A note popped up an octave from its neighbour is a gesture, not more
    range - the 140 lead's favourite move must not read as a wide tune."""
    from riddim.song import Note
    flat = [Note(60, 0, 24), Note(63, 24, 24), Note(67, 48, 24)]
    flicked = flat + [Note(79, 72, 24), Note(67, 96, 24)]
    assert tells._core_range(flicked) == tells._core_range(flat) == 7


@pytest.mark.parametrize("mode", ["minor", "phrygian", "dorian", "harmonic_minor"])
def test_an_inside_hook_keeps_to_its_pentatonic(mode):
    """With `outside` at nothing, every note of the hook is in the mode's
    pentatonic - except the colour tone, placed once on purpose."""
    scale = Scale(48, mode)
    pal = hook.palette(scale)
    colour = hook.MODE_COLOUR[mode][0]
    for seed in range(20):
        params = hook.params_for("half_time", random.Random(seed), "dubstep")
        params = hook.replace(params, outside=0.0, phrasing="call")
        h = hook.generate(params, scale, random.Random(seed))
        for u in h.units:
            for t in u.tones:
                assert t.alter or t.degree % 7 in pal or t.degree % 7 == colour, (mode, seed, t)


def test_economy_prefers_fewer_pitches():
    """Two hooks alike in every way but their vocabulary: the leaner one
    scores higher on economy."""
    p = hook.HookParams(phrasing="riff", span=6, outside=0.0)
    assert hook.vocabulary(p) == 4
    assert hook.vocabulary(hook.replace(p, outside=1.0)) == 6
    assert hook.vocabulary(hook.replace(p, span=11)) == 6


def test_the_hook_is_played_the_same_every_time_round():
    """Flicks, graces, trills and accents belong to a place in the hook, so
    the hook played twice is played the same twice - in this drop and the
    next. Across a batch, most of a first drop's sung bars restate a bar."""
    rates = []
    for profile in FAMILIES:
        for seed in range(100, 106):
            song = compose(profile=profile, seed=seed)
            got = {t.name: t.value for t in tells.tells(song)}
            if "lead_restated" in got:
                rates.append(got["lead_restated"])
    assert sum(rates) / len(rates) >= 0.45, rates


@pytest.mark.parametrize("profile", ["dubstep_140", "dark_garage"])
def test_a_140_drop_is_one_or_two_chords(profile):
    for seed in range(1, 21):
        song = compose(profile=profile, seed=seed)
        sec = tells.first_drop(song)
        a, b = sec.start_tick, sec.start_tick + sec.length_ticks
        roots = {e.chord.root_pc for e in song.harmony.events if e.start < b and e.end > a}
        assert len(roots) <= 2, (profile, seed, roots)


@pytest.mark.parametrize("profile", ["dubstep_140", "dark_garage"])
def test_the_140_drop_sub_sits_on_few_notes(profile):
    """A held, pulsing or skipping drop sub plays the chord's root and fifth
    (and phrygian's b2); the only other note is the approach into the next
    phrase, on the last eighth of its eighth bar."""
    checked = 0
    for seed in range(1, 25):
        song = compose(profile=profile, seed=seed)
        if song.params.bass.style == "walk":
            continue
        sec = tells.first_drop(song)
        flat2 = (song.scale.root_pc + 1) % 12
        for n in song.part("sub").notes:
            if not sec.start_tick <= n.start < sec.start_tick + sec.length_ticks:
                continue
            ev = song.harmony.at(n.start)
            frame = {ev.chord.root_pc, (ev.chord.root_pc + 7) % 12, flat2}
            bar = (n.start - sec.start_tick) // TICKS_PER_BAR
            approach = bar % 8 == 7 and n.start % TICKS_PER_BAR >= 14 * TICKS_PER_BAR // 16
            assert n.midi % 12 in frame or approach, (profile, seed, n)
            checked += 1
    assert checked > 100


def test_a_batch_gives_little_away():
    """The ceiling: across each family, few tells a song. (Before the
    economy work every family averaged three to four.)"""
    for profile in FAMILIES:
        songs = [compose(profile=profile, seed=s) for s in range(100, 108)]
        per_song = sum(tells.count(s) for s in songs) / len(songs)
        assert per_song <= 2.75, (profile, per_song)
