"""Ears: what a recording sounds like, as named numbers a model can reason about.

The judge (riddim/judge.py) and the rig loop both hear through this. Nothing
here is learned and nothing needs torch or a download: plain DSP on a
loudness-matched signal, so the same numbers come out on a laptop, in CI and
on a recording of the rig, and every number has a meaning in words.

Per window (8 s by default) of a stereo signal it measures:

- **tone** - the long-term spectrum in 18 half-octave bands, 25 Hz - 15 kHz,
  each in dB against the whole: the mix's tonal balance (a mastering
  engineer's first look);
- **movement** - how much each band's level moves over 0.4 s blocks (sd, dB):
  a pumping sidechain, a filter sweep or a wobble shows up here;
- **punch** - per coarse band (sub, bass, low mid, high mid, air), how far the
  peaks of the 16 ms envelope stand over its median (p95 - p50, dB): kicks
  and snares that hit, or a squashed limiter;
- **tails** - per coarse band, the quiet floor under the loud frames
  (p10 - p90, dB): dry and gated, or washed in reverb and dub delay;
- **width** - per coarse band, side over mid (dB): a mono sub under a wide top;
- **grit** - per coarse band, spectral flatness (noise-like vs tonal): drive,
  distortion, hiss;
- **groove** - per coarse band, where its envelope's modulation sits: slow
  swells (0.5-2 Hz), bar-level rhythm (2-6 Hz), sixteenths and rolls
  (6-12 Hz), buzz and wobble (12-24 Hz), as shares;
- **overall** - crest (peak to RMS), loudness range, centroid, flux.

    from riddim import ears
    x = ears.load("take.wav")              # (n, 2) float32 at ears.SR
    w = ears.windows(x)                     # (n_windows, n_features)
    ears.NAMES                              # the feature names, in order
    print(ears.describe(w.mean(0)))         # in words

The signal is set to -14 dBFS RMS first (the blind A/B pages match loudness
too), so loudness itself is not a feature: it is measured on its own.
"""

from __future__ import annotations

import subprocess

import numpy as np
from scipy.signal import stft

SR = 32000
WIN_S = 8.0
HOP_S = 4.0
NORM_DB = -14.0
NFFT = 4096
HOP = 512                       # 16 ms at 32 kHz
EDGES = np.array([25, 35, 50, 71, 100, 141, 200, 283, 400, 566, 800, 1131, 1600,
                  2263, 3200, 4525, 6400, 9051, 15000], dtype=float)
COARSE = {"sub": (25, 60), "bass": (60, 250), "lowmid": (250, 1000),
          "himid": (1000, 5000), "air": (5000, 15000)}
MOD_RATES = {"swell": (0.5, 2.0), "beat": (2.0, 6.0), "sixteenth": (6.0, 12.0),
             "buzz": (12.0, 24.0)}


def _names() -> list[str]:
    n = [f"tone_{int(EDGES[i])}" for i in range(len(EDGES) - 1)]
    n += [f"move_{int(EDGES[i])}" for i in range(len(EDGES) - 1)]
    for kind in ("punch", "tails", "width", "grit"):
        n += [f"{kind}_{c}" for c in COARSE]
    for c in COARSE:
        n += [f"groove_{c}_{r}" for r in MOD_RATES]
    n += ["crest", "loud_range", "centroid", "flux"]
    return n


NAMES = _names()
INDEX = {k: i for i, k in enumerate(NAMES)}


def load(path: str, sr: int = SR, seconds: float | None = None, start: float = 0.0,
         norm: bool = True) -> np.ndarray:
    """Decode any audio file to float32 stereo (n, 2) at `sr`, set to -14 dBFS
    RMS unless norm=False."""
    cmd = ["ffmpeg", "-v", "error"]
    if start:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", path]
    if seconds:
        cmd += ["-t", f"{seconds:.3f}"]
    cmd += ["-f", "f32le", "-ac", "2", "-ar", str(sr), "-"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    x = np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).copy()
    return normalize(x) if norm else x


def normalize(x: np.ndarray, db: float = NORM_DB) -> np.ndarray:
    rms = float(np.sqrt(np.mean(np.asarray(x, np.float64) ** 2)))
    return x if rms <= 0 else (x * (10 ** (db / 20) / rms)).astype(np.float32)


def _db(p, floor=1e-12):
    return 10 * np.log10(np.maximum(p, floor))


_CACHE: dict = {}


def _bins(sr: int):
    key = sr
    if key not in _CACHE:
        f = np.fft.rfftfreq(NFFT, 1 / sr)
        band = [np.where((f >= EDGES[i]) & (f < EDGES[i + 1]))[0] for i in range(len(EDGES) - 1)]
        coarse = [np.where((f >= lo) & (f < hi))[0] for lo, hi in COARSE.values()]
        _CACHE[key] = (f, band, coarse)
    return _CACHE[key]


def features(x: np.ndarray, sr: int = SR) -> np.ndarray:
    """The feature vector (len(NAMES),) of one window of stereo audio (n, 2)."""
    x = np.asarray(x, np.float32)
    if x.ndim == 1:
        x = np.stack([x, x], 1)
    mid, side = (x[:, 0] + x[:, 1]) * 0.5, (x[:, 0] - x[:, 1]) * 0.5
    _, _, M = stft(mid, sr, nperseg=NFFT, noverlap=NFFT - HOP, boundary=None, padded=False)
    _, _, S = stft(side, sr, nperseg=NFFT, noverlap=NFFT - HOP, boundary=None, padded=False)
    PM, PS = np.abs(M) ** 2, np.abs(S) ** 2
    f, band, coarse = _bins(sr)
    fps = sr / HOP
    total = PM.sum() + 1e-12
    out = []
    # tone: long-term spectrum, dB against the whole
    bp = np.array([PM[b].sum(0) for b in band])                # (18, T)
    out += list(_db(bp.sum(1) / total))
    # movement: sd of each band's level over 0.4 s blocks
    blk = max(1, int(round(0.4 * fps)))
    nb = bp.shape[1] // blk
    blocks = bp[:, : nb * blk].reshape(len(band), nb, blk).mean(2) if nb else bp
    out += list(np.std(_db(blocks + 1e-10), axis=1))
    cp = np.array([PM[b].sum(0) for b in coarse])              # (5, T)
    cs = np.array([PS[b].sum(0) for b in coarse])
    lv = _db(cp + 1e-10)
    out += list(np.percentile(lv, 95, 1) - np.percentile(lv, 50, 1))          # punch
    out += list(np.percentile(lv, 10, 1) - np.percentile(lv, 90, 1))          # tails
    out += list(_db(cs.sum(1) + 1e-12) - _db(cp.sum(1) + 1e-12))              # width
    grit = []
    for b in coarse:                                                          # flatness
        p = PM[b] + 1e-12
        fl = np.exp(np.mean(np.log(p), 0)) / np.mean(p, 0)
        grit.append(float(np.mean(_db(fl))))
    out += grit
    # groove: modulation spectrum of each coarse band's envelope, as shares
    env = np.sqrt(cp)
    env = env - env.mean(1, keepdims=True)
    spec = np.abs(np.fft.rfft(env * np.hanning(env.shape[1]), axis=1)) ** 2
    mf = np.fft.rfftfreq(env.shape[1], 1 / fps)
    for k in range(len(coarse)):
        sh = np.array([spec[k, (mf >= lo) & (mf < hi)].sum() for lo, hi in MOD_RATES.values()])
        out += list(sh / (sh.sum() + 1e-12))
    # overall
    peak = float(np.max(np.abs(x))) + 1e-9
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) + 1e-9
    out.append(20 * np.log10(peak / rms))
    tot = _db(blocks.sum(0) + 1e-10)
    out.append(float(np.percentile(tot, 95) - np.percentile(tot, 10)))
    spec_mean = PM.mean(1)
    out.append(float(np.log2((f * spec_mean).sum() / (spec_mean.sum() + 1e-12) + 1.0)))
    lm = np.log(PM + 1e-10)
    out.append(float(np.mean(np.maximum(np.diff(lm, axis=1), 0))))
    return np.asarray(out, dtype=np.float64)


def windows(x: np.ndarray, sr: int = SR, win: float = WIN_S, hop: float = HOP_S) -> np.ndarray:
    """Features on overlapping windows: (n_windows, len(NAMES)). A clip shorter
    than one window is measured whole."""
    n, h = int(win * sr), int(hop * sr)
    if len(x) <= n:
        return features(x, sr)[None]
    starts = range(0, len(x) - n + 1, h)
    return np.stack([features(x[s:s + n], sr) for s in starts])


# --- in words ---------------------------------------------------------------

_PLAIN = {
    "tone": "level of the {b} Hz band", "move": "movement of the {b} Hz band",
    "punch": "punch in the {b}", "tails": "tails / space in the {b}",
    "width": "stereo width of the {b}", "grit": "grit (noise, drive) in the {b}",
    "groove": "{b} envelope at {r} rate", "crest": "crest factor",
    "loud_range": "loudness range", "centroid": "brightness (centroid)", "flux": "attack density (flux)",
}


def plain(name: str) -> str:
    """A feature's name in words."""
    parts = name.split("_")
    kind = parts[0] if parts[0] != "loud" else "loud_range"
    t = _PLAIN.get(kind, name)
    if kind == "groove":
        return t.format(b=parts[1], r=parts[2])
    if kind in ("tone", "move", "punch", "tails", "width", "grit"):
        return t.format(b=parts[1])
    return t


def describe(v: np.ndarray) -> str:
    """One feature vector (e.g. a clip's mean over windows) as a few lines a
    producer would recognise."""
    g = dict(zip(NAMES, v))
    tone = [g[f"tone_{int(e)}"] for e in EDGES[:-1]]
    sub = 10 * np.log10(sum(10 ** (t / 10) for t, e in zip(tone, EDGES[:-1]) if e < 60) + 1e-12)
    low = 10 * np.log10(sum(10 ** (t / 10) for t, e in zip(tone, EDGES[:-1]) if 60 <= e < 250) + 1e-12)
    top = 10 * np.log10(sum(10 ** (t / 10) for t, e in zip(tone, EDGES[:-1]) if e >= 5000) + 1e-12)
    lines = [f"balance: sub {sub:+.1f} dB, bass {low:+.1f} dB, air {top:+.1f} dB of the whole; "
             f"centroid {2 ** g['centroid'] - 1:.0f} Hz",
             f"punch (peak over median): sub {g['punch_sub']:.1f}, bass {g['punch_bass']:.1f}, "
             f"high mid {g['punch_himid']:.1f} dB; crest {g['crest']:.1f} dB, loudness range {g['loud_range']:.1f} dB",
             f"space (floor under the peaks): bass {g['tails_bass']:.1f}, mids {g['tails_lowmid']:.1f}, "
             f"top {g['tails_himid']:.1f} dB",
             f"width (side vs mid): sub {g['width_sub']:.1f}, mids {g['width_lowmid']:.1f}, "
             f"top {g['width_air']:.1f} dB"]
    return "\n".join(lines)


_UNITS = {"tone": "dB", "move": "dB", "punch": "dB", "tails": "dB", "width": "dB", "grit": "dB",
          "groove": "share", "crest": "dB", "loud": "dB", "centroid": "oct", "flux": ""}


def diff(Fa: np.ndarray, Fb: np.ndarray, top: int = 6) -> list:
    """What changed from take A to take B, in the measurements' own units,
    biggest first (set against how much each one moves between windows of the
    same take, so a steady 1 dB beats a jumpy 3 dB). For hearing what a knob
    did: the same bars before and after the turn."""
    Fa, Fb = np.atleast_2d(Fa), np.atleast_2d(Fb)
    d = Fb.mean(0) - Fa.mean(0)
    noise = np.sqrt((Fa.var(0) + Fb.var(0)) / 2) + 1e-6
    if len(Fa) < 2:
        noise = np.abs(Fa.mean(0)) * 0.05 + 0.1
    order = np.argsort(-np.abs(d) / noise)[:top]
    out = []
    for i in order:
        k = NAMES[i].split("_")[0]
        out.append((plain(NAMES[i]), float(d[i]), _UNITS.get(k, ""), float(abs(d[i]) / noise[i])))
    return out


def diff_text(Fa: np.ndarray, Fb: np.ndarray, top: int = 6) -> str:
    rows = diff(Fa, Fb, top)
    return "\n".join(f"  {name}: {v:+.2f} {u} (x{r:.1f} its jitter)" for name, v, u, r in rows)
