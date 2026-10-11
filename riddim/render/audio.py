"""Render a song to a WAV, for auditioning without the rig.

This is a sketchpad, not the record: the Deluge and the rack are the
instruments. But choosing forty songs for a show by reading piano rolls is
not possible, and plugging in the rig for each one takes too long - so this
renders every part with a small synth that knows what the part is for:

    sub      sine and a whisper of the octave, soft-saturated, mono
    chords   two detuned saws through a filter pluck, into the dub delay
    pad      three detuned saws, dark and slow, into the reverb
    lead     dub: a reedy melodica; 140: a brighter saw-square with vibrato
    drums    synthesised kick, snare, hats, rim, crash and a noise riser
    perc     a shaker

with the mix moves the genre needs: a dotted-eighth ping-pong delay that
throws harder in the sections marked as throws, a reverb on the pad and the
lead, the kick ducking the pad and stabs, and a soft-clipping master.
Needs numpy and scipy (`pip install numpy scipy`).
"""

from __future__ import annotations

import wave

import numpy as np
from scipy.signal import butter, lfilter, sosfilt

from ..song import Song
from ..units import TICKS_PER_16TH

SR = 44100
_RNG = np.random.default_rng(7)
_NOISE = _RNG.uniform(-1, 1, SR * 6).astype(np.float32)

# Deluge kit rows (see compose.KIT_ROWS).
KICK, SNARE, HAT, RIM, PERC, SHAKER, CRASH, RISER = 36, 37, 38, 39, 40, 41, 42, 43

PAN = {"sub": 0.0, "drums": 0.0, "perc": 0.35, "chords": -0.25, "pad": 0.2, "lead": 0.1}
GAIN = {"sub": 0.42, "drums": 0.62, "perc": 0.2, "chords": 0.27, "pad": 0.13, "lead": 0.25}


def _f(midi) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12.0)


def _noise(n: int) -> np.ndarray:
    if n <= len(_NOISE):
        start = _RNG.integers(0, len(_NOISE) - n + 1)
        return _NOISE[start:start + n].copy()
    return _RNG.uniform(-1, 1, n).astype(np.float32)


def _onepole(x, fc):
    a = np.exp(-2 * np.pi * min(fc, SR * 0.45) / SR)
    return lfilter([1 - a], [1, -a], x)


def _lp2(x, fc):
    return _onepole(_onepole(x, fc), fc)


_SOS = {}


def _band(kind, lo, hi=None):
    key = (kind, lo, hi)
    if key not in _SOS:
        if kind == "bp":
            _SOS[key] = butter(2, [lo / (SR / 2), hi / (SR / 2)], btype="band", output="sos")
        elif kind == "hp":
            _SOS[key] = butter(2, lo / (SR / 2), btype="high", output="sos")
        else:
            _SOS[key] = butter(2, lo / (SR / 2), btype="low", output="sos")
    return _SOS[key]


def _saw(freq, n, detune_cents=0.0, phase=0.0):
    f = freq * 2 ** (detune_cents / 1200.0)
    t = np.arange(n) / SR
    return (2.0 * ((f * t + phase) % 1.0) - 1.0).astype(np.float32)


def _env(n, attack, release, decay=None):
    """Linear attack, optional exponential decay, linear release at the end."""
    env = np.ones(n, dtype=np.float32)
    a = max(1, int(attack * SR))
    env[:a] = np.linspace(0, 1, min(a, n), dtype=np.float32)[: min(a, n)] if a <= n else np.linspace(0, 1, n)
    if decay:
        t = np.arange(n) / SR
        env *= np.exp(-t / decay).astype(np.float32)
    r = max(1, int(release * SR))
    if r < n:
        env[-r:] *= np.linspace(1, 0, r, dtype=np.float32)
    return env


def _vel(v):
    return (v / 127.0) ** 1.4


# --------------------------------------------------------------------------
# Instruments: each returns a mono float array starting at the note's onset
# --------------------------------------------------------------------------

def _sub(midi, dur, vel):
    n = int((dur + 0.03) * SR)
    t = np.arange(n) / SR
    f = _f(midi)
    x = np.sin(2 * np.pi * f * t) + 0.18 * np.sin(4 * np.pi * f * t)
    x = np.tanh(1.6 * x) / np.tanh(1.6)
    return (x * _env(n, 0.004, 0.03) * _vel(vel)).astype(np.float32)


def _stab(midi, dur, vel):
    n = int((max(dur, 0.06) + 0.12) * SR)
    f = _f(midi)
    x = _saw(f, n, -7) + _saw(f, n, 7, 0.37)
    bright = _lp2(x, 2600 + 2400 * vel / 127)
    dark = _lp2(x, 420)
    t = np.arange(n) / SR
    mix = np.exp(-t / 0.07)
    y = dark + (bright - dark) * mix
    return (y * _env(n, 0.002, 0.08, decay=0.35) * _vel(vel) * 0.5).astype(np.float32)


def _pad(midi, dur, vel):
    n = int((dur + 0.9) * SR)
    f = _f(midi)
    x = _saw(f, n, -9) + _saw(f, n, 9, 0.21) + 0.6 * _saw(f / 2, n, 0, 0.5)
    y = _lp2(x, 900)
    return (y * _env(n, 0.45, 0.85) * _vel(vel) * 0.35).astype(np.float32)


def _lead(midi, dur, vel, bright: bool, glide_from=None):
    n = int((dur + 0.15) * SR)
    t = np.arange(n) / SR
    f = _f(midi)
    vib = 1 + 0.004 * np.sin(2 * np.pi * 5.3 * t) * np.clip((t - 0.15) / 0.2, 0, 1)
    if glide_from is not None:
        # A slide: start on the previous note and glide in over 70ms.
        ratio = 2 ** ((glide_from - midi) / 12.0)
        vib = vib * (1 + (ratio - 1) * np.exp(-t / 0.025))
    phase = np.cumsum(f * vib) / SR
    if bright:
        # Two detuned saws over a square an octave down, with filter bite.
        saw = (2 * (phase % 1.0) - 1) + (2 * ((phase * 1.006) % 1.0) - 1)
        sq = np.sign(np.sin(np.pi * phase))
        raw = 0.45 * saw + 0.3 * sq
        cut = 2600 + 3000 * vel / 127
        bitey = _lp2(raw, cut)
        y = _lp2(raw, 1500) + (bitey - _lp2(raw, 1500)) * np.exp(-t / 0.12)
        y = np.tanh(1.5 * y)
    else:
        # A reed: odd-heavy, with a breathy edge.
        y = (np.sin(2 * np.pi * phase) + 0.45 * np.sin(6 * np.pi * phase)
             + 0.2 * np.sin(4 * np.pi * phase) + 0.12 * np.sin(10 * np.pi * phase))
        y = y + 0.04 * _lp2(_noise(n), 3000)
        y = _lp2(y, 2600)
    return (y * _env(n, 0.012, 0.12) * _vel(vel) * 0.5).astype(np.float32)


def _drum(row, dur, vel):
    v = _vel(vel)
    if row == KICK:
        n = int(0.45 * SR)
        t = np.arange(n) / SR
        freq = 46 + 120 * np.exp(-t / 0.028)
        x = np.sin(2 * np.pi * np.cumsum(freq) / SR) * np.exp(-t / 0.32)
        x[: int(0.004 * SR)] += _noise(int(0.004 * SR)) * 0.4
        return (np.tanh(2.0 * x) * v).astype(np.float32)
    if row == SNARE:
        n = int(0.28 * SR)
        t = np.arange(n) / SR
        body = np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.045)
        crack = sosfilt(_band("bp", 1200, 7000), _noise(n)) * np.exp(-t / 0.11)
        return ((0.55 * body + 1.1 * crack) * v * 0.8).astype(np.float32)
    if row == HAT:
        n = int(0.07 * SR)
        t = np.arange(n) / SR
        return (sosfilt(_band("hp", 7500), _noise(n)) * np.exp(-t / 0.022) * v * 0.55).astype(np.float32)
    if row == RIM:
        n = int(0.05 * SR)
        t = np.arange(n) / SR
        x = np.sin(2 * np.pi * 1650 * t) + 0.6 * sosfilt(_band("bp", 1500, 4000), _noise(n))
        return (x * np.exp(-t / 0.012) * v * 0.45).astype(np.float32)
    if row in (SHAKER, PERC):
        n = int(0.09 * SR)
        t = np.arange(n) / SR
        env = np.minimum(1, t / 0.006) * np.exp(-t / 0.035)
        return (sosfilt(_band("bp", 5000, 11000), _noise(n)) * env * v * 0.5).astype(np.float32)
    if row == CRASH:
        n = int(2.4 * SR)
        t = np.arange(n) / SR
        x = sosfilt(_band("hp", 4200), _noise(n)) * np.exp(-t / 0.75)
        return (x * v * 0.45).astype(np.float32)
    if row == RISER:
        n = max(int(dur * SR), SR // 4)
        t = np.arange(n) / SR
        prog = t / t[-1]
        # Noise swept up through a filter, block by block, with a rising tone.
        out = np.zeros(n, dtype=np.float32)
        noise = _noise(n)
        block = 2048
        for i in range(0, n, block):
            fc = 300 * (40 ** prog[i])
            sos = butter(2, min(fc, SR * 0.45) / (SR / 2), btype="high", output="sos")
            out[i:i + block] = sosfilt(sos, noise[i:i + block])
        tone = np.sin(2 * np.pi * np.cumsum(200 * 8 ** prog) / SR) * 0.15
        return ((out * 0.8 + tone) * prog ** 2 * v * 0.5).astype(np.float32)
    return np.zeros(1, dtype=np.float32)


# --------------------------------------------------------------------------
# Effects
# --------------------------------------------------------------------------

def _feedback_delay(x, d, fb):
    """y[n] = x[n] + fb * y[n - d], computed a block of `d` at a time."""
    y = x.astype(np.float32).copy()
    for i in range(d, len(y), d):
        j = min(i + d, len(y))
        y[i:j] += fb * y[i - d:j - d]
    return y


def _reverb(x):
    """A Schroeder reverb: four combs into two allpasses."""
    out = np.zeros_like(x)
    for d, g in ((1557, 0.82), (1617, 0.81), (1491, 0.83), (1422, 0.82)):
        out += _feedback_delay(x, d, g)
    out *= 0.25
    for d, g in ((225, 0.7), (556, 0.7)):
        y = _feedback_delay(out, d, g)
        delayed = np.concatenate([np.zeros(d, dtype=np.float32), y[:-d]])
        out = -g * out + delayed * (1 - g * g)
    return _lp2(out, 5000)


def _add(buf, start, x):
    if start >= len(buf):
        return
    end = min(len(buf), start + len(x))
    buf[start:end] += x[: end - start]


# --------------------------------------------------------------------------
# The mix
# --------------------------------------------------------------------------

def render(song: Song) -> np.ndarray:
    """The song as a stereo float array, shape (n, 2)."""
    spt = 60.0 / song.bpm / 96.0          # seconds per tick
    total = song.arrangement.length_ticks * spt + 4.0
    n = int(total * SR)
    bright = song.bpm >= 120 or (song.params is not None and "dubstep" in song.params.profile)
    stems = {r: np.zeros(n, dtype=np.float32) for r in PAN}
    kick_env = np.zeros(n, dtype=np.float32)

    for part in song.parts:
        role = part.role
        buf = stems.setdefault(role, np.zeros(n, dtype=np.float32))
        prev = None
        for note in song.notes_with_coda(part):
            start = int(note.start * spt * SR)
            dur = note.length * spt
            if part.output == "kit":
                x = _drum(note.midi, dur, note.velocity)
                if note.midi == KICK:
                    k = int(0.25 * SR)
                    _add(kick_env, start, np.exp(-np.arange(k) / (0.09 * SR)).astype(np.float32))
            elif role == "sub":
                x = _sub(note.midi, dur, note.velocity)
            elif role == "chords":
                x = _stab(note.midi, dur, note.velocity)
            elif role == "pad":
                x = _pad(note.midi, dur, note.velocity)
            else:
                glide = prev.midi if (prev is not None and prev.end > note.start) else None
                x = _lead(note.midi, dur, note.velocity, bright, glide)
                prev = note
            _add(buf, start, x)

    # Kick ducks the pad and the stabs - the room breathes with the kick.
    duck = 1.0 - 0.55 * np.clip(kick_env, 0, 1)
    for r in ("pad", "chords"):
        stems[r] *= duck

    # Delay sends: a dotted eighth, ping-ponged, harder through the throws.
    throw = np.full(n, 0.35, dtype=np.float32)
    for s in song.arrangement.sections:
        if s.dub_throw:
            a = int(s.start_tick * spt * SR)
            b = int((s.start_tick + s.length_ticks) * spt * SR)
            throw[a:b] = 0.85
    send = (stems["chords"] * 1.0 + stems["lead"] * 0.45) * throw
    send = _lp2(sosfilt(_band("hp", 250), send), 3500)
    d = int(3 * TICKS_PER_16TH * spt * SR)
    left = _feedback_delay(send, 2 * d, 0.42)
    right = np.concatenate([np.zeros(d, dtype=np.float32), left[:-d]])

    verb = _reverb(stems["pad"] * 0.8 + stems["lead"] * 0.35 + stems["drums"] * 0.06)

    mix = np.zeros((n, 2), dtype=np.float32)
    for role, x in stems.items():
        g = GAIN.get(role, 0.2)
        p = PAN.get(role, 0.0)
        mix[:, 0] += x * g * (1 - max(0.0, p))
        mix[:, 1] += x * g * (1 + min(0.0, p))
    mix[:, 0] += left * 0.10 + verb * 0.22
    mix[:, 1] += right * 0.10 + verb * 0.22

    # Master: clear the very bottom, soft-clip, normalise to -1 dBFS.
    mix = sosfilt(_band("hp", 22), mix, axis=0)
    peak = np.max(np.abs(mix)) or 1.0
    mix = np.tanh(1.4 * mix / peak) / np.tanh(1.4)
    mix *= 0.89 / (np.max(np.abs(mix)) or 1.0)
    return mix


def write(song: Song, path: str) -> str:
    """Render `song` to a 16-bit stereo WAV at `path`."""
    data = render(song)
    pcm = (np.clip(data, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return path
