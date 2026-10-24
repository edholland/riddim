"""Hand percussion: the hands sit on the kit's grid, keep off the backbeat,
repeat their part, and call into the next phrase."""

import random

from riddim.compose import HAND_ROWS, KIT_ROWS, compose
from riddim.theory import percussion
from riddim.units import TICKS_PER_16TH, TICKS_PER_BAR


def test_params_are_drawn_in_range():
    for seed in range(30):
        p = percussion.params_for("dubstep", random.Random(seed))
        assert set(p.voices) <= set(percussion.VOICES)
        assert {"tom_lo", "tom_hi"} <= set(p.voices)
        assert 0.0 <= p.density <= 1.0 and p.clave in percussion.CLAVES


def test_the_low_conga_keeps_off_the_backbeat():
    p = percussion.PercParams(voices=("conga_lo", "conga_hi"), density=1.0)
    for beat_steps in (4, 8):
        back = percussion._backbeat(beat_steps)
        hits = percussion.pattern(p, beat_steps, 1.0, random.Random(1))
        assert not {int(s) for s, _ in hits["conga_lo"]} & back


def test_hands_play_on_the_sixteenth_grid_and_repeat():
    """On the grid (the Deluge's swing moves them with the kit), and the
    same two bars each time round outside the phrase's call."""
    song = compose(profile="dubstep_140", seed=205)
    perc = song.part("perc")
    hands = [n for n in perc.notes if n.midi in HAND_ROWS]
    assert hands
    assert all(n.start % TICKS_PER_16TH == 0 for n in hands)
    sec = next(s for s in song.arrangement.sections if s.bars >= 16 and s.plays("perc"))

    def bars(b):
        lo = (sec.start_bar + b) * TICKS_PER_BAR
        return sorted((n.start - lo, n.midi) for n in hands
                      if lo <= n.start < lo + 2 * TICKS_PER_BAR)
    assert bars(0) == bars(2) == bars(4)


def test_phrases_end_in_a_call():
    song = compose(profile="dubstep_140", seed=205)
    toms = {KIT_ROWS["tom_lo"], KIT_ROWS["tom_hi"]}
    perc = song.part("perc")
    called = [n for n in perc.notes if n.midi in toms or n.midi == KIT_ROWS["conga_hi"]
              and (n.start // TICKS_PER_BAR) % 8 == 7]
    assert called
    for n in perc.notes:
        if n.midi in toms:
            assert (n.start // TICKS_PER_BAR) % 8 == 7      # only in the call bar
