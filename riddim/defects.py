"""Known flaws, put on real records: the ground truth the judge is tested on.

A record as released is the side that should win. The same record with a
flaw on it - a band boosted, the master squashed, the sub gone wide, the
mids washed in reverb - is the side that should lose. These pairs need no
listener: the answer is known, at a known size, on thousands of records.

Each flaw belongs to a group, so a test can hold a whole group out (train on
EQ and dynamics flaws, test on stereo and space ones) and say whether the
judge hears flaws it was never shown. Sizes run from about the smallest a
mix engineer would notice to obvious.

    from riddim import defects
    y = defects.apply(x, "sub_up", 6.0)       # x: (n, 2) float at ears.SR
    for name, group, sizes in defects.CATALOGUE: ...
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve, sosfilt

from .ears import SR


def _biquad(kind: str, f0: float, gain_db: float, q: float = 0.7, sr: int = SR) -> np.ndarray:
    """RBJ cookbook biquad as one SOS row."""
    a = 10 ** (gain_db / 40)
    w = 2 * np.pi * f0 / sr
    cw, sw = np.cos(w), np.sin(w)
    alpha = sw / (2 * q)
    if kind == "peak":
        b = [1 + alpha * a, -2 * cw, 1 - alpha * a]
        d = [1 + alpha / a, -2 * cw, 1 - alpha / a]
    elif kind in ("low", "high"):
        s = 2 * np.sqrt(a) * alpha
        if kind == "low":
            b = [a * ((a + 1) - (a - 1) * cw + s), 2 * a * ((a - 1) - (a + 1) * cw),
                 a * ((a + 1) - (a - 1) * cw - s)]
            d = [(a + 1) + (a - 1) * cw + s, -2 * ((a - 1) + (a + 1) * cw), (a + 1) + (a - 1) * cw - s]
        else:
            b = [a * ((a + 1) + (a - 1) * cw + s), -2 * a * ((a - 1) + (a + 1) * cw),
                 a * ((a + 1) + (a - 1) * cw - s)]
            d = [(a + 1) - (a - 1) * cw + s, 2 * ((a - 1) - (a + 1) * cw), (a + 1) - (a - 1) * cw - s]
    elif kind == "lp":
        b = [(1 - cw) / 2, 1 - cw, (1 - cw) / 2]
        d = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "hp":
        b = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]
        d = [1 + alpha, -2 * cw, 1 - alpha]
    else:
        raise ValueError(kind)
    return np.array([[b[0] / d[0], b[1] / d[0], b[2] / d[0], 1.0, d[1] / d[0], d[2] / d[0]]])


def _filt(x, *rows):
    return sosfilt(np.vstack(rows), x, axis=0).astype(np.float32)


def _ms(x):
    return (x[:, 0] + x[:, 1]) * 0.5, (x[:, 0] - x[:, 1]) * 0.5


def _lr(m, s):
    return np.stack([m + s, m - s], 1).astype(np.float32)


def _env(x, sr, ms):
    """One-pole envelope follower of |x| (mono), time constant ms."""
    from scipy.signal import lfilter
    a = np.exp(-1.0 / (sr * ms / 1000.0))
    return lfilter([1 - a], [1, -a], np.abs(x))


def _drive(x, amount):          # soft saturation into a limiter-ish ceiling
    g = 10 ** (amount / 20)
    return (np.tanh(x * g) / np.tanh(g)).astype(np.float32) if g > 1 else x


def _clip(x, db_over):
    c = np.max(np.abs(x)) * 10 ** (-db_over / 20)
    return np.clip(x, -c, c).astype(np.float32)


def _compress(x, amount, sr=SR):
    """Fast, hard bus compression: `amount` dB of gain reduction on the peaks."""
    m = np.max(np.abs(x), 1)
    e = _env(m, sr, 5.0)
    thr = np.percentile(e, 50)
    over = np.maximum(e / (thr + 1e-9), 1.0)
    ratio = 1 + amount / 3.0
    gain = over ** (1 / ratio - 1)
    return (x * gain[:, None]).astype(np.float32)


def _smear(x, ms, sr=SR):
    """Soften attacks: each hit's first `ms` ms ducked (a transient shaper set
    to minus)."""
    m = np.max(np.abs(x), 1)
    fast, slow = _env(m, sr, 1.0), _env(m, sr, ms)
    g = np.clip(slow / (fast + 1e-9), 0.25, 1.0)
    return (x * g[:, None]).astype(np.float32)


def _reverb(x, wet, rt60=2.2, sr=SR, seed=0):
    rng = np.random.default_rng(seed)
    n = int(rt60 * sr)
    t = np.arange(n) / sr
    ir = rng.standard_normal((n, 2)) * np.exp(-6.9 * t / rt60)[:, None]
    ir[: int(0.01 * sr)] = 0
    ir /= np.sqrt((ir ** 2).sum(0, keepdims=True))
    y = np.stack([fftconvolve(x[:, c], ir[:, c])[: len(x)] for c in range(2)], 1)
    y *= np.sqrt(np.mean(x ** 2) / (np.mean(y ** 2) + 1e-12))
    return ((1 - wet) * x + wet * y).astype(np.float32)


def _echo(x, wet, ms=375.0, fb=0.45, sr=SR):
    d = int(ms * sr / 1000)
    y = np.zeros_like(x)
    tap, g = x.copy(), 1.0
    for _ in range(6):
        g *= fb
        tap = np.concatenate([np.zeros((d, 2), x.dtype), tap[:-d]])
        y += g * tap
    return (x + wet * y).astype(np.float32)


def _wide_low(x, amount, sr=SR):
    """Low end pushed into the sides: the sub goes out of phase on a mono rig."""
    m, s = _ms(x)
    lo = sosfilt(_biquad("lp", 150, 0, 0.7, sr), m)
    d = int(0.004 * sr)
    s = s + amount * np.concatenate([np.zeros(d), lo[:-d]])
    return _lr(m, s)


def _width(x, side_db):
    m, s = _ms(x)
    return _lr(m, s * 10 ** (side_db / 20))


def _noise(x, db_under, seed=0):
    rng = np.random.default_rng(seed)
    n = rng.standard_normal(x.shape).astype(np.float32)
    n = _filt(n, _biquad("hp", 2000, 0))
    rms = np.sqrt(np.mean(x ** 2))
    return (x + n * rms * 10 ** (-db_under / 20) / (np.sqrt(np.mean(n ** 2)) + 1e-12)).astype(np.float32)


def _hum(x, db_under, sr=SR):
    t = np.arange(len(x)) / sr
    h = sum(np.sin(2 * np.pi * 50 * k * t) / k for k in (1, 2, 3, 5))
    rms = np.sqrt(np.mean(x ** 2))
    h = h / np.sqrt(np.mean(h ** 2)) * rms * 10 ** (-db_under / 20)
    return (x + h[:, None]).astype(np.float32)


def _gate(x, depth_db, sr=SR):
    """A choppy, pumping level: the whole mix ducked on every beat at 140."""
    t = np.arange(len(x)) / sr
    ph = (t * 140 / 60) % 1.0
    g = 1 - (1 - 10 ** (-depth_db / 20)) * np.exp(-ph / 0.18)
    return (x * g[:, None]).astype(np.float32)


# name: (group, function(x, size) -> y, sizes from subtle to obvious)
_F = {
    "sub_up":      ("eq", lambda x, s: _filt(x, _biquad("low", 55, s, 0.7)), (2, 4, 8)),
    "sub_down":    ("eq", lambda x, s: _filt(x, _biquad("low", 55, -s, 0.7)), (2, 4, 8)),
    "bass_up":     ("eq", lambda x, s: _filt(x, _biquad("peak", 120, s, 0.9)), (2, 4, 8)),
    "bass_down":   ("eq", lambda x, s: _filt(x, _biquad("peak", 120, -s, 0.9)), (2, 4, 8)),
    "mud":         ("eq", lambda x, s: _filt(x, _biquad("peak", 350, s, 1.0)), (2, 4, 8)),
    "honk":        ("eq", lambda x, s: _filt(x, _biquad("peak", 1000, s, 1.2)), (2, 4, 8)),
    "harsh":       ("eq", lambda x, s: _filt(x, _biquad("peak", 3500, s, 1.0)), (2, 4, 8)),
    "scoop":       ("eq", lambda x, s: _filt(x, _biquad("peak", 800, -s, 0.5)), (2, 4, 8)),
    "air_up":      ("eq", lambda x, s: _filt(x, _biquad("high", 8000, s, 0.7)), (2, 4, 8)),
    "dull":        ("eq", lambda x, s: _filt(x, _biquad("high", 5000, -s, 0.7)), (2, 4, 8)),
    "lowpass":     ("band", lambda x, s: _filt(x, _biquad("lp", s, 0, 0.7), _biquad("lp", s, 0, 0.7)), (9000, 6000, 3500)),
    "highpass":    ("band", lambda x, s: _filt(x, _biquad("hp", s, 0, 0.7), _biquad("hp", s, 0, 0.7)), (40, 70, 120)),
    "drive":       ("dyn", lambda x, s: _drive(x * 3.0, s), (6, 12, 20)),
    "clip":        ("dyn", lambda x, s: _clip(x, s), (3, 6, 10)),
    "squash":      ("dyn", lambda x, s: _compress(x, s), (3, 6, 12)),
    "smear":       ("dyn", lambda x, s: _smear(x, s), (10, 25, 60)),
    "pump":        ("dyn", lambda x, s: _gate(x, s), (2, 4, 8)),
    "reverb":      ("space", lambda x, s: _reverb(x, s), (0.15, 0.3, 0.55)),
    "echo":        ("space", lambda x, s: _echo(x, s), (0.2, 0.4, 0.8)),
    "mono":        ("stereo", lambda x, s: _width(x, -s), (6, 12, 60)),
    "too_wide":    ("stereo", lambda x, s: _width(x, s), (3, 6, 10)),
    "wide_low":    ("stereo", lambda x, s: _wide_low(x, s), (0.25, 0.5, 1.0)),
    "hiss":        ("noise", lambda x, s: _noise(x, s), (40, 30, 20)),
    "hum":         ("noise", lambda x, s: _hum(x, s), (40, 30, 20)),
}

CATALOGUE = [(name, g, sizes) for name, (g, _, sizes) in _F.items()]
GROUPS = sorted({g for g, _, _ in _F.values()})


def apply(x: np.ndarray, name: str, size: float) -> np.ndarray:
    """The record `x` (n, 2) float32 at ears.SR with flaw `name` at `size`."""
    return _F[name][1](np.asarray(x, np.float32), size)


def group(name: str) -> str:
    return _F[name][0]
