"""The ears, the flaws and the judge (riddim/ears.py, defects.py, judge.py)."""

import os

import numpy as np
import pytest

from riddim import defects, ears, judge

SR = ears.SR


def _beat(seconds=20.0, seed=0):
    """A crude record-ish stereo loop at 140: kick + sub, snare noise, hats, a
    chord - enough spectrum and movement for every feature to be defined."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    beat = 60 / 140
    ph = (t % beat) / beat
    kick = np.sin(2 * np.pi * (50 + 80 * np.exp(-ph * 30)) * t) * np.exp(-ph * 8)
    sub = 0.5 * np.sin(2 * np.pi * 49 * t)
    sn_ph = ((t + beat) % (2 * beat)) / (2 * beat)
    snare = rng.standard_normal(n) * np.exp(-sn_ph * 40) * 0.4
    hat_ph = (t % (beat / 2)) / (beat / 2)
    hats = np.diff(rng.standard_normal(n + 1)) * np.exp(-hat_ph * 60) * 0.15
    chord = 0.1 * sum(np.sin(2 * np.pi * f * t) for f in (220, 262, 330))
    m = kick + sub + snare + hats + chord
    side = 0.3 * (hats + 0.5 * chord * np.sin(2 * np.pi * 0.2 * t))
    x = np.stack([m + side, m - side], 1).astype(np.float32)
    return ears.normalize(x)


def test_features_are_named_and_finite():
    x = _beat()
    w = ears.windows(x)
    assert w.shape[1] == len(ears.NAMES) == len(set(ears.NAMES))
    assert w.shape[0] == 4                      # 20 s in 8 s windows every 4 s
    assert np.all(np.isfinite(w))
    assert "balance" in ears.describe(w.mean(0))


def test_features_hear_the_obvious():
    x = _beat()
    w = ears.windows(x).mean(0)
    i = ears.INDEX
    dull = ears.windows(ears.normalize(defects.apply(x, "lowpass", 3500))).mean(0)
    assert dull[i["tone_9051"]] < w[i["tone_9051"]] - 10
    mono = ears.windows(ears.normalize(defects.apply(x, "mono", 60))).mean(0)
    assert mono[i["width_air"]] < w[i["width_air"]] - 20
    wet = ears.windows(ears.normalize(defects.apply(x, "reverb", 0.55))).mean(0)
    assert wet[i["tails_himid"]] > w[i["tails_himid"]]


@pytest.mark.parametrize("name,group,sizes", defects.CATALOGUE)
def test_every_flaw_runs_and_changes_the_audio(name, group, sizes):
    x = _beat(10.0)
    for s in sizes:
        y = defects.apply(x, name, s)
        assert y.shape == x.shape and np.all(np.isfinite(y))
        assert not np.allclose(y, x)
    assert defects.group(name) == group in defects.GROUPS


needs_model = pytest.mark.skipif(not os.path.exists(judge.MODEL), reason="no fitted judge model")


@needs_model
def test_the_same_take_is_cant_tell():
    x = _beat()
    w = ears.windows(x)
    v = judge.compare_features(w, w.copy(), "dubstep_140")
    assert v.winner is None and v.confidence == "can't tell"


@needs_model
def test_a_gross_flaw_loses_and_says_why():
    x = _beat()
    bad = ears.normalize(defects.apply(x, "lowpass", 3500))
    v = judge.compare_features(ears.windows(x), ears.windows(bad), "dubstep_140")
    assert v.gap > 0                            # never the wrong way round
    assert v.reasons
    flip = judge.compare_features(ears.windows(bad), ears.windows(x), "dubstep_140")
    assert flip.gap == pytest.approx(-v.gap)


@needs_model
def test_model_matches_the_ears():
    panel = judge.load_model()
    assert set(panel.heads) == {"dist", "full"}
    assert not np.any(panel.heads["dist"].u)            # distance only: no direction is better
    for m in panel.heads.values():
        assert m.names == ears.NAMES
        assert np.all(m.w >= 0) and m.tau > 0
        for style in judge.STYLES:
            c, s = m.ref(style)
            assert c.shape == s.shape == (len(ears.NAMES),) and np.all(s > 0)
            assert style in m.meta.get("record_median", {})


def test_stem_balance_reads_the_rig():
    x = _beat(10.0)
    drums = x * 0.5
    bass = x * 0.7
    chords = x * 0.1
    rows = {r[0]: r for r in judge.stem_balance({"drums": drums, "bass": bass, "chords": chords}, "dubstep_140")}
    assert set(rows) == {"drums", "bass", "other", "vocals"}
    assert rows["vocals"][1] is None
    # the sum is coherent here, so each stem's share is its amplitude over 1.3
    assert rows["bass"][1] == pytest.approx(20 * np.log10(0.7 / 1.3), abs=0.1)
    assert rows["bass"][1] > rows["drums"][1] > rows["other"][1]
