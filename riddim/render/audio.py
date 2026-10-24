"""Render a song to a WAV, for auditioning without the rig.

This is a sketchpad, not the record - the Deluge and the rack are the
instruments - but it is what a set is chosen by, so it has to sound like a
record rather than like a MIDI file. What a record has that a bare synth
does not, and this renders:

- **a room.** Nothing on a record stops dead: every part sits in a space,
  the snare in a long plate, the skank in a spring, the lead in a hall, and
  the desk's tape hiss is under all of it. Digital black - a gap of pure
  silence - is the plainest sign of a computer.
- **width.** A mono sub and kick under a wide top: hats, shakers, skanks and
  returns spread across the stereo field, never two parts on one point.
- **movement.** A 140 bass is a sub *and* a mid-bass growl whose filter
  moves in time; the sub pumps under the kick; a skank's filter opens with
  how hard it is hit; no two drum hits are quite the same.
- **the desk.** A dotted-eighth ping-pong delay whose repeats darken and
  saturate, thrown harder in the dub sections and run up into every drop;
  a high-pass sweep through the end of each build; bus glue and tape.

    sub      sine with a whisper of harmonics, pumped by the kick, mono
    bass     (140 profiles) the mid-bass: detuned saw and square an octave up,
             its filter wobbling in time; (dub) a bass-guitar pluck on the sub
    chords   dub: a plucked guitar chop into the spring; garage/140: an organ
    pad      a string machine: six detuned saws, chorused wide, slow
    lead     dub: a reed with breath and a scoop; 140: a filtered saw-square
    drums    kick (body, click, drive), snare (two modes and noise, into the
             plate), 808-style metallic hats, rim, shaker, crash, riser

Needs numpy and scipy (`pip install numpy scipy`).
"""

from __future__ import annotations

import wave

import numpy as np
from scipy.signal import butter, lfilter, oaconvolve, sosfilt

from ..song import Song
from ..units import TICKS_PER_16TH, TICKS_PER_BAR

SR = 44100
_HERE = __import__("os").path.dirname(__file__)
_SAMPLES: dict = {}


def _sample(name: str) -> np.ndarray:
    """A stereo float sample from render/samples (see LICENSES.md there)."""
    if name not in _SAMPLES:
        with wave.open(__import__("os").path.join(_HERE, "samples", name + ".wav")) as w:
            raw = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
            _SAMPLES[name] = (raw.reshape(-1, w.getnchannels()).astype(np.float32) / 32768.0)
    return _SAMPLES[name]


def _play(name, rate=1.0, gain=1.0):
    """A sample played back at `rate` (pitch and length together, as a
    sampler does), mono-summed for the kit's own panning."""
    x = _sample(name).mean(axis=1)
    if abs(rate - 1.0) > 1e-4:
        idx = np.arange(0, len(x) - 1, rate)
        x = np.interp(idx, np.arange(len(x)), x).astype(np.float32)
    return (x * gain).astype(np.float32)

# Deluge kit rows (see compose.KIT_ROWS).
KICK, SNARE, HAT, RIM, PERC, SHAKER, CRASH, RISER = 36, 37, 38, 39, 40, 41, 42, 43
# The hands (theory/percussion.py): row -> (sample, pan, gain).
HANDS = {44: ("hands/conga_lo", -0.3, 1.0), 45: ("hands/conga_hi", 0.3, 0.9),
         46: ("hands/bongo", 0.5, 0.75), 47: ("hands/block", 0.6, 0.55),
         48: ("hands/tamb", -0.55, 0.5), 49: ("hands/tom_lo", -0.4, 1.0),
         50: ("hands/tom_hi", 0.4, 0.95)}

# Desk move switches (for auditioning what each one does).
SWEEP = True
# The melodic parts sit back: band-limited, driven, quieter, more of them in
# the echo than in front of it. A clean synth lead up front is what makes a
# render read as MIDI; a bright pad in a big hall is what makes it new age.
TREAT_LEAD = False
TREAT_PAD = True
# Where the tune sits: in the mix, not on it. A lead riding on top of a
# record is the loudest tell there is (measured: the same drop with its lead
# 10 dB down goes from 0.12 to 0.64 "real record" with the judge).
LEAD_LEVEL = 0.4

# Per-part stereo position (-1 left .. 1 right) and level.
PAN = {"sub": 0.0, "bass": 0.0, "drums": 0.0, "perc": 0.45, "chords": -0.3, "pad": 0.0, "lead": 0.12}
GAIN = {"sub": 0.45, "bass": 0.42, "drums": 0.75, "perc": 0.5, "chords": 0.5, "pad": 0.32,
        "lead": 0.42}
# Sends: (plate, spring, hall, delay)
SENDS = {"drums": (0.0, 0.0, 0.05, 0.0), "chords": (0.0, 0.35, 0.1, 0.5),
         "pad": (0.0, 0.0, 0.45, 0.0), "lead": (0.0, 0.0, 0.3, 0.3),
         "perc": (0.0, 0.08, 0.2, 0.06), "bass": (0.0, 0.0, 0.0, 0.0), "sub": (0.0, 0.0, 0.0, 0.0)}


class _Rand:
    """Deterministic randomness per song: the same song renders the same."""

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def noise(self, n):
        return self.rng.uniform(-1, 1, n).astype(np.float32)

    def jitter(self, amount):
        return float(self.rng.uniform(-amount, amount))


def _f(midi) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12.0)


_SOS: dict = {}


def _sos(kind, lo, hi=None, order=2):
    key = (kind, round(lo, 1), None if hi is None else round(hi, 1), order)
    if key not in _SOS:
        ny = SR / 2
        lo = min(lo, ny * 0.95)
        if kind == "bp":
            _SOS[key] = butter(order, [lo / ny, min(hi, ny * 0.95) / ny], btype="band", output="sos")
        elif kind == "hp":
            _SOS[key] = butter(order, lo / ny, btype="high", output="sos")
        else:
            _SOS[key] = butter(order, lo / ny, btype="low", output="sos")
    return _SOS[key]


def _lp(x, fc, order=2):
    return sosfilt(_sos("lp", fc, order=order), x, axis=0)


def _hp(x, fc, order=2):
    return sosfilt(_sos("hp", fc, order=order), x, axis=0)


def _bp(x, lo, hi, order=2):
    return sosfilt(_sos("bp", lo, hi, order=order), x, axis=0)


def _env(n, attack, release, decay=None, hold=None):
    """Attack ramp, optional exponential decay, release ramp at the end."""
    t = np.arange(n, dtype=np.float32) / SR
    env = np.minimum(1.0, t / max(attack, 1e-4))
    if decay:
        env = env * np.exp(-np.maximum(0.0, t - attack) / decay)
    r = max(1, int(release * SR))
    if r < n:
        env[-r:] *= np.linspace(1, 0, r, dtype=np.float32)
    return env.astype(np.float32)


def _vel(v):
    return (v / 127.0) ** 1.3


def _phase(freq, n, start=0.0):
    if np.isscalar(freq):
        return (start + freq * np.arange(n) / SR).astype(np.float64)
    return start + np.cumsum(freq) / SR


def _saw(ph):
    return (2.0 * (ph % 1.0) - 1.0).astype(np.float32)


def _sq(ph, width=0.5):
    return np.where((ph % 1.0) < width, 1.0, -1.0).astype(np.float32)


def _pan(x, p):
    """Equal-power pan of a mono signal to stereo."""
    a = (p + 1) * np.pi / 4
    return np.stack([x * np.cos(a), x * np.sin(a)], axis=1).astype(np.float32)


def _sat(x, drive=1.5):
    return (np.tanh(drive * x) / np.tanh(drive)).astype(np.float32)


# --------------------------------------------------------------------------
# Instruments (mono, starting at the onset)
# --------------------------------------------------------------------------

def _sub(midi, dur, vel, rnd):
    n = int((dur + 0.04) * SR)
    f = _f(midi)
    ph = _phase(f * (1 + 0.012 * np.exp(-np.arange(n) / (0.03 * SR))), n, rnd.jitter(0.5))
    x = np.sin(2 * np.pi * ph) + 0.12 * np.sin(4 * np.pi * ph) + 0.04 * np.sin(6 * np.pi * ph)
    x = _sat(x.astype(np.float32), 1.3)
    return x * _env(n, 0.005, 0.035) * _vel(vel)


def _bass_guitar(midi, dur, vel, rnd):
    """The dub bass's mid: a round pluck an octave above the sub's
    fundamental, so the line reads on small speakers too."""
    n = int((dur + 0.06) * SR)
    f = _f(midi)
    ph = _phase(f, n, rnd.jitter(0.5))
    x = 0.7 * _saw(ph) + 0.5 * np.sin(2 * np.pi * ph)
    t = np.arange(n) / SR
    bright = _lp(x, 900 + 900 * vel / 127)
    dark = _lp(x, 260)
    y = dark + (bright - dark) * np.exp(-t / 0.08)
    return (y * _env(n, 0.003, 0.05, decay=0.9) * _vel(vel) * 0.8).astype(np.float32)


def _wobble(midi, dur, vel, rnd, rate_hz):
    """The 140 mid-bass: detuned saws and a square an octave above the sub,
    through a filter that opens and closes `rate_hz` times a second."""
    n = int((dur + 0.03) * SR)
    f = _f(midi) * 2
    a = _saw(_phase(f * 1.004, n, rnd.jitter(0.5)))
    b = _saw(_phase(f * 0.996, n, rnd.jitter(0.5)))
    c = _sq(_phase(f / 2, n, rnd.jitter(0.5)), 0.42)
    x = 0.5 * a + 0.5 * b + 0.45 * c
    t = np.arange(n) / SR
    lfo = 0.5 - 0.5 * np.cos(2 * np.pi * rate_hz * t)          # 0..1, starts closed
    dark = _lp(x, 180, order=4)
    mid = _lp(x, 650, order=4)
    bright = _lp(x, 1900, order=4)
    y = np.where(lfo < 0.5, dark + (mid - dark) * (lfo * 2), mid + (bright - mid) * (lfo * 2 - 1))
    y = _sat(y * 1.4, 2.0)
    return (_hp(y, 70) * _env(n, 0.006, 0.04) * _vel(vel)).astype(np.float32)


def _pluck(midi, dur, vel, rnd):
    """A plucked guitar chop: Karplus-Strong - a burst of noise in a
    tuned, damped loop - choked short, as a skank is."""
    f = _f(midi)
    period = SR / f
    N = int(period)
    frac = period - N
    n = int((min(dur, 0.25) + 0.08) * SR)
    burst = np.zeros(n, dtype=np.float32)
    k = min(n, N)
    burst[:k] = _lp(rnd.noise(k), 2500 + 3500 * vel / 127)
    g = 0.996
    a = np.zeros(N + 2)
    a[0] = 1.0
    a[N] = -g * (1 - frac) * 0.5 - g * 0.5 * (1 - frac)
    a[N + 1] = -g * frac * 0.5 - g * 0.5 * frac
    y = lfilter([1.0], a, burst)
    y = _hp(_bp(y, 300, 4200), 200)
    choke = _env(n, 0.001, 0.05, decay=0.12)
    return (y * choke * _vel(vel) * 1.6).astype(np.float32)


def _organ(midi, dur, vel, rnd):
    """A drawbar organ stab: 16', 8', 5 1/3', 4' and a little percussion."""
    n = int((max(dur, 0.08) + 0.08) * SR)
    f = _f(midi)
    ph = _phase(f, n, rnd.jitter(0.5))
    x = (0.5 * np.sin(np.pi * ph) + 1.0 * np.sin(2 * np.pi * ph) + 0.6 * np.sin(3 * np.pi * ph)
         + 0.5 * np.sin(4 * np.pi * ph) + 0.25 * np.sin(8 * np.pi * ph))
    t = np.arange(n) / SR
    x = x + 0.6 * np.sin(6 * np.pi * ph) * np.exp(-t / 0.06)       # percussion
    x = _sat(x.astype(np.float32) * 0.6, 1.4)
    return (x * _env(n, 0.003, 0.06) * _vel(vel) * 0.45).astype(np.float32)


def _strings(midi, dur, vel, rnd):
    """String machine: six detuned saws, dark, slow - returned in stereo."""
    n = int((dur + 1.0) * SR)
    f = _f(midi)
    t = np.arange(n) / SR
    out = np.zeros((n, 2), dtype=np.float32)
    for i, cents in enumerate((-14, -8, -3, 3, 8, 14)):
        ff = f * 2 ** (cents / 1200) * (1 + 0.002 * np.sin(2 * np.pi * (0.3 + 0.07 * i) * t))
        v = _saw(_phase(ff, n, rnd.jitter(0.5)))
        out[:, i % 2] += v
    out = _lp(out, 1500) * 0.22
    return out * (_env(n, 0.5, 0.9) * _vel(vel))[:, None]


def _reed(midi, dur, vel, rnd, glide_from=None):
    """A melodica: a reed's odd-heavy spectrum, breath, a scoop into the
    note and vibrato that arrives late."""
    n = int((dur + 0.12) * SR)
    t = np.arange(n) / SR
    f = _f(midi)
    scoop = 1 - 0.018 * np.exp(-t / 0.035)
    vib = 1 + 0.005 * np.sin(2 * np.pi * (5.0 + rnd.jitter(0.4)) * t) * np.clip((t - 0.2) / 0.3, 0, 1)
    ff = f * scoop * vib
    if glide_from is not None:
        ff = ff * (1 + (2 ** ((glide_from - midi) / 12.0) - 1) * np.exp(-t / 0.03))
    ph = _phase(ff, n, rnd.jitter(0.5))
    y = (np.sin(2 * np.pi * ph) + 0.55 * np.sin(6 * np.pi * ph) + 0.25 * np.sin(4 * np.pi * ph)
         + 0.18 * np.sin(10 * np.pi * ph) + 0.08 * np.sin(14 * np.pi * ph))
    breath = _bp(rnd.noise(n), 1200, 5000) * (0.12 + 0.2 * np.exp(-t / 0.05))
    y = _lp((y * 0.5 + breath).astype(np.float32), 3200)
    return (y * _env(n, 0.025, 0.1) * _vel(vel) * 0.55).astype(np.float32)


def _synth_lead(midi, dur, vel, rnd, glide_from=None):
    n = int((dur + 0.12) * SR)
    t = np.arange(n) / SR
    f = _f(midi)
    vib = 1 + 0.004 * np.sin(2 * np.pi * 5.5 * t) * np.clip((t - 0.15) / 0.2, 0, 1)
    if glide_from is not None:
        vib = vib * (1 + (2 ** ((glide_from - midi) / 12.0) - 1) * np.exp(-t / 0.025))
    ph = _phase(f * vib, n, rnd.jitter(0.5))
    ph2 = _phase(f * vib * 1.007, n, rnd.jitter(0.5))
    raw = 0.4 * _saw(ph) + 0.4 * _saw(ph2) + 0.3 * _sq(ph * 0.5)
    open_ = _lp(raw, 2400 + 2600 * vel / 127)
    closed = _lp(raw, 900)
    y = closed + (open_ - closed) * np.exp(-t / 0.15)
    return (_sat(y, 1.6) * _env(n, 0.006, 0.1) * _vel(vel) * 0.4).astype(np.float32)


# 808 metal: six square oscillators at inharmonic ratios.
_METAL = (205.3, 304.4, 369.6, 522.7, 540.0, 800.0)


def _drum(row, dur, vel, rnd, garage=False, sampled=True):
    """A kit voice. With `sampled`, the recorded Pearl kit does the work -
    a different snare take each hit, every hit a hair off in pitch, as a
    sampler with round-robin and a drummer give - with the synthesised
    voice layered under it for the weight a sound system wants."""
    v = _vel(vel)
    if sampled and row in (KICK, SNARE, HAT, CRASH):
        rate = 1.0 + rnd.jitter(0.025)
        if row == KICK:
            top = _hp(_play("pearl/kick-01", rate * 0.92), 70)
            body = _drum(KICK, dur, vel, rnd, garage, sampled=False)
            m = max(len(top), len(body))
            out = np.zeros(m, dtype=np.float32)
            out[:len(body)] += body * 0.85
            out[:len(top)] += top * 1.6 * v
            return out
        if row == SNARE:
            take = ("pearl/snare-01", "pearl/snare-02", "pearl/snare-03")[int(rnd.rng.integers(0, 3))]
            hit = _play(take, rate * 0.94) * 1.7 * v
            body = _drum(SNARE, dur, vel, rnd, garage, sampled=False) * 0.35
            m = max(len(hit), len(body))
            out = np.zeros(m, dtype=np.float32)
            out[:len(hit)] += hit
            out[:len(body)] += body
            return out
        if row == HAT:
            name = "pearl/hihat-open" if dur > 0.2 else "pearl/hihat-closed"
            x = _hp(_play(name, rate), 2500)
            cut = int((min(dur, 0.45) if name.endswith("open") else 0.12) * SR)
            x = x[:cut] * np.linspace(1, 0, min(cut, len(x)), dtype=np.float32)[: len(x[:cut])]
            return (x * 1.5 * v).astype(np.float32)
        if row == CRASH:
            return (_hp(_play("pearl/crash-01", rate), 300) * 1.2 * v).astype(np.float32)
    if row == KICK:
        n = int(0.5 * SR)
        t = np.arange(n) / SR
        drop = 0.035 * (1 + rnd.jitter(0.05))
        freq = 47 + 95 * np.exp(-t / drop) + 400 * np.exp(-t / 0.004)
        body = np.sin(2 * np.pi * np.cumsum(freq) / SR) * np.exp(-t / 0.38)
        click = _bp(rnd.noise(n), 1500, 6000) * np.exp(-t / 0.004) * 0.5
        return (_sat((body + click).astype(np.float32), 2.2) * v).astype(np.float32)
    if row == SNARE:
        n = int(0.4 * SR)
        t = np.arange(n) / SR
        tune = 1 + rnd.jitter(0.015)
        body = (np.sin(2 * np.pi * 185 * tune * t) * np.exp(-t / 0.06)
                + 0.5 * np.sin(2 * np.pi * 330 * tune * t) * np.exp(-t / 0.035))
        wires = _bp(rnd.noise(n), 1500, 9000) * np.exp(-t / (0.14 + rnd.jitter(0.015)))
        crack = _hp(rnd.noise(n), 3000) * np.exp(-t / 0.008)
        return ((0.6 * body + 0.95 * wires + 0.4 * crack) * v * 0.85).astype(np.float32)
    if row == HAT:
        n = int(0.09 * SR)
        t = np.arange(n) / SR
        metal = sum(_sq(_phase(fr * (1 + rnd.jitter(0.002)), n, rnd.jitter(0.5))) for fr in _METAL)
        x = _hp(_bp(metal / 6.0, 7000, 12000), 7000) + 0.25 * _hp(rnd.noise(n), 9000)
        decay = 0.028 + 0.012 * (1 - vel / 127) + rnd.jitter(0.004)
        return (x * np.exp(-t / decay) * v * 0.9).astype(np.float32)
    if row == RIM:
        n = int(0.06 * SR)
        t = np.arange(n) / SR
        x = np.sin(2 * np.pi * 1750 * t) * 0.8 + 0.6 * _bp(rnd.noise(n), 1500, 5000)
        x = x + 0.6 * np.sin(2 * np.pi * 520 * t)
        return (x * np.exp(-t / 0.014) * v * 0.5).astype(np.float32)
    if row in (SHAKER, PERC):
        n = int(0.11 * SR)
        t = np.arange(n) / SR
        env = np.minimum(1, t / (0.01 + rnd.jitter(0.003))) * np.exp(-t / 0.04)
        return (_bp(rnd.noise(n), 4500, 12000) * env * v * 0.55).astype(np.float32)
    if row == CRASH:
        n = int(3.0 * SR)
        t = np.arange(n) / SR
        metal = sum(_sq(_phase(fr * 1.7, n, rnd.jitter(0.5))) for fr in _METAL) / 6.0
        x = _hp(rnd.noise(n) * 0.7 + metal * 0.5, 3500) * np.exp(-t / 0.9)
        return (x * v * 0.42).astype(np.float32)
    if row == RISER:
        n = max(int(dur * SR), SR // 4)
        t = np.arange(n) / SR
        prog = t / t[-1]
        out = np.zeros(n, dtype=np.float32)
        noise = rnd.noise(n)
        block = 4096
        for i in range(0, n, block):
            fc = 300 * (40 ** prog[i])
            out[i:i + block] = _hp(noise[i:i + block], fc)
        tone = np.sin(2 * np.pi * np.cumsum(200 * 8 ** prog) / SR) * 0.12
        return ((out * 0.8 + tone) * prog ** 2 * v * 0.45).astype(np.float32)
    return np.zeros(1, dtype=np.float32)


# Where each kit voice sits: hats and shaker off centre, the rest in the middle.
DRUM_PAN = {HAT: 0.28, SHAKER: -0.4, PERC: 0.4, RIM: -0.18, CRASH: 0.0, RISER: 0.0}


# --------------------------------------------------------------------------
# Spaces
# --------------------------------------------------------------------------

def _ir(seconds, damp_hz, seed, predelay=0.0, density_ms=None):
    """A stereo impulse response: decorrelated noise, decaying exponentially,
    darker as it goes (the high end dies first, as in any real space)."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    out = np.zeros((n, 2), dtype=np.float32)
    rt = seconds / 6.9        # -60 dB at `seconds`
    for ch in range(2):
        noise = rng.standard_normal(n).astype(np.float32)
        bright = _hp(noise, 2500) * np.exp(-t / (rt * 0.45))
        mid = _bp(noise, 400, 2500) * np.exp(-t / (rt * 0.8))
        low = _lp(noise, 400) * np.exp(-t / rt)
        out[:, ch] = _lp(bright + mid + 0.7 * low, damp_hz)
    out[:, :] *= np.minimum(1.0, t / 0.008)[:, None]
    if predelay:
        out = np.concatenate([np.zeros((int(predelay * SR), 2), dtype=np.float32), out])
    return out / (np.sqrt((out ** 2).sum(axis=0).mean()) + 1e-9)


def _spring_ir(seed):
    """A spring tank: a dispersive chirp that recurs every ~35 ms and fades -
    the boing under every dub skank."""
    rng = np.random.default_rng(seed)
    n = int(2.2 * SR)
    t = np.arange(n) / SR
    out = np.zeros((n, 2), dtype=np.float32)
    for ch, period in ((0, 0.033), (1, 0.039)):
        chirp_n = int(0.03 * SR)
        tc = np.arange(chirp_n) / SR
        chirp = np.sin(2 * np.pi * (3500 * tc - 40000 * tc ** 2)) * np.hanning(chirp_n)
        for k in range(int(2.2 / period)):
            i = int(k * period * SR) + rng.integers(0, 60)
            if i + chirp_n < n:
                out[i:i + chirp_n, ch] += chirp * (0.72 ** k)
        out[:, ch] += 0.35 * _bp(rng.standard_normal(n), 500, 4000) * np.exp(-t / 0.45)
    out = _bp(out, 350, 4500)
    return out / (np.sqrt((out ** 2).sum(axis=0).mean()) + 1e-9)


def _norm_ir(ir):
    return ir / (np.sqrt((ir ** 2).sum(axis=0).mean()) + 1e-9)


def _convolve(x, ir):
    """Stereo (or mono-to-stereo) convolution."""
    if x.ndim == 1:
        x = np.stack([x, x], axis=1)
    out = np.stack([oaconvolve(x[:, c], ir[:, c])[: len(x)] for c in range(2)], axis=1)
    return out.astype(np.float32)


def _dub_delay(send, d, fb, taps=10):
    """Ping-pong delay at `d` samples whose repeats get darker, thinner and
    a little dirtier each time round - a tape echo, not a digital copy.
    `fb` is per-sample feedback (an array, so the desk can ride it)."""
    n = len(send)
    out = np.zeros((n, 2), dtype=np.float32)
    tap = send.astype(np.float32)
    gain = np.ones(n, dtype=np.float32)
    for k in range(1, taps + 1):
        tap = _sat(_lp(_hp(tap, 280), 3200), 1.2)
        gain = gain * fb
        shifted = np.zeros(n, dtype=np.float32)
        if k * d < n:
            shifted[k * d:] = (tap * gain)[: n - k * d]
        out[:, (k - 1) % 2] += shifted
        if float(np.max(gain)) < 0.02:
            break
    return out


# --------------------------------------------------------------------------
# Mix tools
# --------------------------------------------------------------------------

def _follower(x, attack, release):
    """A peak envelope follower: a fast one-pole for attack and a slow one
    for release, combined - close enough to a compressor's detector."""
    a = np.exp(-1.0 / (attack * SR))
    r = np.exp(-1.0 / (release * SR))
    fast = lfilter([1 - a], [1, -a], x)
    slow = lfilter([1 - r], [1, -r], x)
    return np.maximum(fast, slow)


def _compress(x, thresh_db, ratio, attack=0.01, release=0.15, makeup_db=0.0):
    level = np.abs(x).max(axis=1) if x.ndim == 2 else np.abs(x)
    env = _follower(level, attack, release)
    db = 20 * np.log10(env + 1e-9)
    over = np.maximum(0.0, db - thresh_db)
    gain = 10 ** ((-over * (1 - 1 / ratio) + makeup_db) / 20)
    return (x * (gain[:, None] if x.ndim == 2 else gain)).astype(np.float32)


def _tape(x, rnd):
    """Tape and desk: a gentle asymmetric saturation, the top rolled off,
    a little hiss - so the record is never digitally black."""
    y = np.tanh(1.15 * x + 0.04) - np.tanh(0.04)
    y = _lp(y, 15500)
    hiss = np.stack([_hp(rnd.noise(len(x)), 3000), _hp(rnd.noise(len(x)), 3000)], axis=1)
    return (y + hiss * 10 ** (-66 / 20)).astype(np.float32)


def _add(buf, start, x):
    if start >= len(buf) or start < 0:
        return
    end = min(len(buf), start + len(x))
    buf[start:end] += x[: end - start]


# --------------------------------------------------------------------------
# The mix
# --------------------------------------------------------------------------

def render(song: Song, window: tuple[int, int] | None = None) -> np.ndarray:
    """The song as a stereo float array, shape (n, 2). `window` (start, end
    ticks) renders just that stretch - with what rings into it from the
    four seconds before - for auditioning one section."""
    spt = 60.0 / song.bpm / 96.0          # seconds per tick
    w0 = 0
    if window is not None:
        w0 = max(0, window[0] - int(4.0 / spt))
        lead_in = int((window[0] - w0) * spt * SR)
        total = (window[1] - w0) * spt + 3.0
    else:
        lead_in = 0
        total = song.arrangement.length_ticks * spt + 5.0
    n = int(total * SR)
    prof = song.params.profile if song.params is not None else ""
    is140 = song.bpm >= 120 or "dubstep" in prof
    garage = "garage" in prof
    rnd = _Rand((song.seed or 0) * 7919 + 13)

    stems = {r: np.zeros((n, 2), dtype=np.float32) for r in PAN}
    kick_env = np.zeros(n, dtype=np.float32)
    beat = 60.0 / song.bpm

    for part in song.parts:
        role = part.role
        buf = stems.setdefault(role, np.zeros((n, 2), dtype=np.float32))
        prev = None
        for note in song.notes_with_coda(part):
            if window is not None and not (w0 <= note.start < window[1]):
                continue
            start = int((note.start - w0) * spt * SR)
            dur = note.length * spt
            if part.output == "kit" and note.midi in HANDS:
                name, p, g = HANDS[note.midi]
                # A hand: the take a hair off in pitch and level every stroke.
                x = _play(name, 1.0 + rnd.jitter(0.02), g * _vel(note.velocity) * 1.4)
                _add(buf, start, _pan(x, p + rnd.jitter(0.05)))
                continue
            if part.output == "kit":
                x = _drum(note.midi, dur, note.velocity, rnd, garage)
                p = DRUM_PAN.get(note.midi, 0.0)
                if note.midi == HAT:
                    p += rnd.jitter(0.08)
                _add(buf, start, _pan(x, p))
                if note.midi == KICK:
                    k = int(0.3 * SR)
                    _add(kick_env, start, np.exp(-np.arange(k) / (0.1 * SR)).astype(np.float32))
                continue
            if role == "sub":
                _add(buf, start, _pan(_sub(note.midi, dur, note.velocity, rnd), 0.0))
                # The bass's mid: a growl at 140, a pluck in dub.
                if is140:
                    rate = 2.0 / beat if note.length * spt > beat else 1.0 / beat
                    mb = _wobble(note.midi, dur, note.velocity, rnd, rate)
                else:
                    mb = _bass_guitar(note.midi + 12, dur, note.velocity, rnd)
                _add(stems["bass"], start, _pan(mb, 0.0))
                continue
            if role == "chords":
                x = _organ(note.midi, dur, note.velocity, rnd) if (garage or is140) \
                    else _pluck(note.midi, dur, note.velocity, rnd)
                _add(buf, start, _pan(x, PAN["chords"] + rnd.jitter(0.05)))
                continue
            if role == "pad":
                _add(buf, start, _strings(note.midi, dur, note.velocity, rnd))
                continue
            glide = prev.midi if (prev is not None and prev.end > note.start) else None
            x = (_synth_lead if is140 else _reed)(note.midi, dur, note.velocity, rnd, glide)
            _add(buf, start, _pan(x, PAN["lead"]))
            prev = note

    # The kick pumps the sub (a little), the bass, the stabs and the pad.
    kick = np.clip(kick_env, 0, 1)
    for r, depth in (("sub", 0.25), ("bass", 0.5), ("chords", 0.45), ("pad", 0.6)):
        stems[r] *= (1.0 - depth * kick)[:, None]

    # Clear the mud: only the kick and the sub own the bottom.
    for r, hp in (("chords", 220), ("pad", 160), ("lead", 200), ("perc", 120), ("bass", 70)):
        stems[r] = _hp(stems[r], hp)
    stems["lead"] = stems["lead"] * LEAD_LEVEL
    sends = dict(SENDS)
    if TREAT_LEAD:
        stems["lead"] = _sat(_lp(_hp(stems["lead"], 380, order=4), 3000, order=4) * 1.6, 1.8) * 0.45
        sends["lead"] = (0.0, 0.12, 0.35, 0.6)
    if TREAT_PAD:
        stems["pad"] = _lp(_hp(stems["pad"], 260), 950) * 0.7
        sends["pad"] = (0.0, 0.0, 0.2, 0.0)

    # The desk. Delay feedback rides up through the dub sections and runs
    # away into every drop; a high-pass sweeps through the end of a build.
    fb = np.full(n, 0.42, dtype=np.float32)
    throw = np.full(n, 0.6, dtype=np.float32)
    sweep = np.zeros(n, dtype=np.float32)       # 0 = open, 1 = thinned
    secs = song.arrangement.sections
    for i, s in enumerate(secs):
        a = min(n, max(0, int((s.start_tick - w0) * spt * SR)))
        b = min(n, max(0, int((s.start_tick + s.length_ticks - w0) * spt * SR)))
        if s.dub_throw:
            throw[a:b] = 1.5
            fb[a:b] = 0.55
        nxt = secs[i + 1] if i + 1 < len(secs) else None
        if nxt is not None and nxt.energy >= 0.8 and s.energy < nxt.energy:
            ramp = int(2 * TICKS_PER_BAR * spt * SR)
            lo = max(a, b - ramp)
            fb[lo:b] = np.linspace(0.45, 0.72, b - lo)
            sweep[lo:b] = np.linspace(0.0, 1.0, b - lo) ** 2

    plate_send = np.zeros((n, 2), dtype=np.float32)
    spring_send = np.zeros((n, 2), dtype=np.float32)
    hall_send = np.zeros((n, 2), dtype=np.float32)
    delay_send = np.zeros(n, dtype=np.float32)
    # The snare alone into the plate: the dubstep snare's whole size is its
    # room. The riser and the crash go into the hall and the echo, so the
    # drop's gap rings with what led into it instead of falling into silence.
    if song.part("drums") is not None:
        sn = np.zeros(n, dtype=np.float32)
        fx = np.zeros(n, dtype=np.float32)
        for note in song.notes_with_coda(song.part("drums")):
            if window is not None and not (w0 <= note.start < window[1]):
                continue
            at = int((note.start - w0) * spt * SR)
            if note.midi in (SNARE, RIM):
                _add(sn, at, _drum(note.midi, note.length * spt, note.velocity, rnd, garage)
                     * (1.0 if note.midi == SNARE else 0.4))
            elif note.midi in (RISER, CRASH):
                _add(fx, at, _drum(note.midi, note.length * spt, note.velocity, rnd))
        for perc_part in [p for p in song.parts if p.role == "perc"]:
            for note in song.notes_with_coda(perc_part):
                if note.midi in (49, 50) and (window is None or w0 <= note.start < window[1]):
                    _add(fx, int((note.start - w0) * spt * SR),
                         _play(HANDS[note.midi][0], 1.0, 0.6 * _vel(note.velocity)))
        plate_send += np.stack([sn, sn], axis=1)
        hall_send += np.stack([fx, fx], axis=1) * 0.5
        delay_send += fx * 0.35 + sn * 0.12
    for r, (pl, sp, ha, de) in sends.items():
        x = stems[r]
        if pl:
            plate_send += x * pl
        if sp:
            spring_send += x * sp
        if ha:
            hall_send += x * ha
        if de:
            delay_send += x.mean(axis=1) * de
    delay_send *= throw

    plate = _convolve(_hp(plate_send, 300), _norm_ir(_sample("ir/emt_140_dark_2")))
    spring = _convolve(spring_send, _spring_ir(17))
    hall = _convolve(_hp(hall_send, 250), _norm_ir(_sample("ir/large_long_echo_hall")))
    # The kit in a room: a little of a real drum room on every hit.
    stems["drums"] = stems["drums"] + 0.22 * _convolve(
        _hp(stems["drums"], 120), _norm_ir(_sample("ir/nice_drum_room")))
    d = int(3 * TICKS_PER_16TH * spt * SR)
    echo = _dub_delay(_hp(delay_send, 250), d, fb)

    music = np.zeros((n, 2), dtype=np.float32)
    for role, x in stems.items():
        music += x * GAIN.get(role, 0.2)
    music += plate * 0.18 + spring * 0.12 + hall * 0.2 + echo * 0.32

    # The build's sweep thins everything but the riser and the sub's absence.
    if sweep.any() and SWEEP:
        thin = _hp(music, 900)
        music = music + (thin - music) * (sweep * 0.85)[:, None]

    # Master: glue, tape, a soft ceiling.
    mix = _hp(music, 24)
    mix /= (np.max(np.abs(mix)) or 1.0)
    mix = _compress(mix, -14.0, 2.5, attack=0.015, release=0.2, makeup_db=4.0)
    mix = _tape(mix, rnd)
    mix = np.tanh(1.3 * mix / (np.max(np.abs(mix)) or 1.0)) / np.tanh(1.3)
    mix *= 0.89 / (np.max(np.abs(mix)) or 1.0)
    return mix[lead_in:].astype(np.float32)


def write(song: Song, path: str, window: tuple[int, int] | None = None) -> str:
    """Render `song` (or a `window` of it, in ticks) to a 16-bit stereo WAV."""
    data = render(song, window)
    pcm = (np.clip(data, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return path
