"""Reading stems back onto the grid, and mining rhythm pools from them."""

import os
import random
import sys

import numpy as np

from riddim import transcribe as T

CELL = [(0, 3, 40), (4, 2, 43), (6, 2, 45), (10, 3, 40), (14, 2, 38)]


def _stems(bpm=76.0, bars=6, lead=0.37, rotate=0, seed=1):
    """A kick on every beat (harder on 1 and 3) and a sine bass playing CELL,
    the bar starting `rotate` beats after the first kick."""
    sr, step = T.SR, 60.0 / bpm / 4
    n = int((lead + (bars + 1) * 16 * step + 1) * sr)
    drums, bass = np.zeros(n), np.zeros(n)
    rng = random.Random(seed)
    for k in range((bars + 1) * 4):
        t = lead + k * 4 * step + rng.gauss(0, 0.004)
        i, length = int(t * sr), int(0.15 * sr)
        tt = np.arange(length) / sr
        hard = (k - rotate) % 2 == 0
        drums[i:i + length] += (1.0 if hard else 0.5) * np.sin(
            2 * np.pi * (50 + 80 * np.exp(-tt * 40)) * tt) * np.exp(-tt * 20)
    for b in range(bars):
        for s, d, m in CELL:
            t = lead + (rotate * 4 + b * 16 + s) * step
            length = int(d * step * 0.9 * sr)
            tt = np.arange(length) / sr
            env = np.minimum(1, tt / 0.01) * np.minimum(1, (length / sr - tt) / 0.02)
            i = int(t * sr)
            bass[i:i + length] += 0.5 * np.sin(2 * np.pi * 440 * 2 ** ((m - 69) / 12) * tt) * env
    return bass, drums


def test_bass_notes_pitches():
    bass, _ = _stems(bars=2)
    notes = T.bass_notes(bass)
    assert [round(m) for _, _, m in notes[:5]] == [m for _, _, m in CELL]


def test_beat_grid_tempo():
    _, drums = _stems()
    beats = T.beat_grid(drums, 75)
    assert abs(60 / np.median(np.diff(beats)) - 76) < 0.5


def test_bars_recover_cell():
    bass, drums = _stems()
    notes = T.bass_notes(bass)
    beats = T.beat_grid(drums, 75)
    bars = T.bars(notes, beats, T.bar_phase(beats, notes, drums))
    assert len(bars) >= 4
    for bar in bars[:4]:
        assert T.mask_string(T.mask(bar)) == "x...x.x...x...x."
        assert [round(m) for _, _, m, _ in bar] == [m for _, _, m in CELL]


def test_bar_phase_finds_the_one():
    bass, drums = _stems(rotate=1)
    notes = T.bass_notes(bass)
    beats = T.beat_grid(drums, 75)
    bars = [b for b in T.bars(notes, beats, T.bar_phase(beats, notes, drums)) if b]
    assert T.mask_string(T.mask(bars[1])) == "x...x.x...x...x."


def test_mask_string():
    assert T.mask(([0, 1], [14, 1])) == (1 | 1 << 14)
    assert T.mask_string(1 | 1 << 14) == "x.............x."


def _mine():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "ml"))
    import mine_cells
    return mine_cells


def test_pool_counts_records_not_bars():
    mc = _mine()
    recs = [{"bars": [[(0,), (4,)]] * 30}, {"bars": [[(0,), (4,)]]},
            {"bars": [[(0,), (8,)]] * 3}, {"bars": [[(0,), (8,)]]}, {"bars": [[(0,), (12,)]] * 50}]
    got = mc.pool(recs, lambda r: [T.mask(x) for x in r["bars"]])
    masks = [m for m, _ in got]
    assert (1 | 1 << 12) not in masks           # one record's loop is not a pool
    assert masks[0] == (1 | 1 << 4) and got[0][1] == 2


def test_build_by_profile():
    mc = _mine()
    bar = [(0, 3.0, 40.0, 0.0), (14, 2.0, 38.0, 0.05)]
    recs = [{"family": "steppers_dub", "bars": [bar] * 4, "stabs": [0b0100010001000100] * 4,
             "swing": 0.02},
            {"family": "digital_dub", "bars": [bar] * 4, "stabs": [0b0100010001000100] * 4,
             "swing": 0.0}]
    cells = mc.build(recs)
    c = cells["steppers_dub"]
    assert c["families"] == ["digital_dub", "steppers_dub"]
    assert c["bass"] == [[1 | 1 << 14, 2]]
    assert c["stats"]["and_of_4"] == 1.0 and c["stats"]["one"] == 1.0
    assert "steppers_dub" in mc.report(cells)


def _fake_cells(monkeypatch, data):
    from riddim.theory import cells
    monkeypatch.setattr(cells, "_data", lambda path=cells.PATH: data)
    return cells


def _fit(on, count):
    """A fit whose records strike only the steps in `on`, `count` notes a bar."""
    return {"stats": {"steps": [0.9 if k in on else 0.0 for k in range(16)],
                      "notes": [1.0 if k == count else 0.0 for k in range(17)]}}


def test_bass_pair_draws_from_the_fit(monkeypatch):
    cells = _fake_cells(monkeypatch, {"roots_dub": _fit({0, 6, 10, 14}, 3)})
    for seed in range(20):
        verse, peak = cells.bass_pair("roots_dub", random.Random(seed))
        v, p = ({int(h.step) for h in x.hits} for x in (verse, peak))
        assert len(v) == 3 and v <= {0, 6, 10, 14}
        assert v < p <= {0, 6, 10, 14}           # the peak keeps the verse, adds to it
    assert cells.bass_pair("uk_garage", random.Random(0)) is None


def test_cells_off(monkeypatch):
    from riddim.theory import cells
    monkeypatch.setenv("RIDDIM_CELLS", "off")
    assert cells.stats("roots_dub") == {} and cells.bass_pair("roots_dub", random.Random(0)) is None


def test_compose_plays_mined_bass(monkeypatch):
    from collections import Counter
    from dataclasses import replace
    from riddim import profile
    from riddim.compose import compose
    from riddim.units import STEPS_PER_BAR, TICKS_PER_16TH
    # The sub stays on eighths (compose._on_eighths).
    _fake_cells(monkeypatch, {"steppers_dub": _fit({0, 6, 10}, 3)})

    def steps(mined):
        prof = replace(profile.get("steppers_dub"), mined_bass=mined, sub_sixteenths=False)
        song = compose(prof, seed=5)
        part = next(p for p in song.parts if p.role in ("sub", "bass"))
        return Counter(round(n.start / TICKS_PER_16TH) % STEPS_PER_BAR for n in part.notes)

    on, off = steps(1.0), steps(0.0)
    assert on[6] + on[10] > off[6] + off[10]
    assert on[4] + on[12] < off[4] + off[12]


def test_phase_from_tracker_downbeats():
    mc = _mine()
    beats = np.arange(0, 20, 0.8)
    assert mc.phase_from(beats, beats[2::4] + 0.01) == (2, 1.0)
    assert mc.phase_from(beats, beats[0::2])[0] is None      # double-time tracker: a tie
    assert mc.phase_from(beats, None) == (None, 0.0)


def test_sixteenths_follow_the_records(monkeypatch):
    shares = [0.5, 0.1, 0.3, 0.4, 0.4, 0.1, 0.3, 0.2, 0.4, 0.1, 0.3, 0.35, 0.4, 0.1, 0.3, 0.1]
    cells = _fake_cells(monkeypatch, {"steppers_dub": {"stats": {"steps": shares}}})
    assert cells.sixteenths("steppers_dub") == {3, 11}
    assert cells.sixteenths("uk_garage") == {3, 11}           # no fit: the a of 1 and 3


def test_the_one_is_a_whole_song_choice(monkeypatch):
    fit = _fit({0, 4, 6, 10, 14}, 3)
    cells = _fake_cells(monkeypatch, {"steppers_dub": fit})
    for seed in range(10):
        verse, peak = cells.bass_pair("steppers_dub", random.Random(seed))
        assert verse.hits[0].step == 0 and peak.hits[0].step == 0
    fit["stats"]["one_never"] = 1.0
    for seed in range(10):
        verse, peak = cells.bass_pair("steppers_dub", random.Random(seed))
        assert all(h.step != 0 for h in verse.hits + peak.hits)
