"""Read a separated stem back into notes on riddim's grid.

The other half of `listen`: that module measures how a record feels, this
one writes down what its bass player (and its skank) actually played, so the
rhythms riddim draws from can come from records rather than from a hand-made
table. It works on stems (Demucs's bass, drums and other), which is what makes
a monophonic reading good enough: a sub bass on its own is one note at a time.

- `bass_notes`: frame-wise pitch (autocorrelation) and level, split into notes
  where the level jumps or the pitch moves and stays moved.
- `beat_grid`: the beat from the drums' low end (kick and snare body), at the
  tempo nearest the family's felt tempo, re-fitted beat by beat so a live
  drummer's drift doesn't smear the grid.
- `bar_phase`: which beat is the one - the rotation where the bass and the kick
  land together most.
- `bars`: notes onto the 16-step grid, bar by bar; a note too far between
  two steps (a fill, a flam, a swung push past the grid) is left out.

Numpy and scipy only, like `listen`.
"""

from __future__ import annotations

import numpy as np

from . import listen
from .units import STEPS_PER_BAR

SR = listen.SR
HOP = 256
FPS = SR / HOP


def _frames(x: np.ndarray, win: int, hop: int = HOP) -> np.ndarray:
    n = max(0, 1 + (len(x) - win) // hop)
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    return x[idx] if n else np.zeros((0, win))


def pitch_track(x: np.ndarray, fmin: float = 28.0, fmax: float = 400.0,
                win: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Fundamental (Hz, 0 where unvoiced) and RMS level per hop of a mono
    signal. The first autocorrelation peak near the best one is the period,
    as in `library.f0`, so a strong second harmonic doesn't halve the note."""
    pad = np.concatenate([np.zeros(win // 2), x, np.zeros(win // 2)])
    fr = _frames(pad, win)
    fr = fr - fr.mean(axis=1, keepdims=True)
    level = np.sqrt((fr ** 2).mean(axis=1) + 1e-12)
    n = 1 << int(np.ceil(np.log2(2 * win)))
    ac = np.fft.irfft(np.abs(np.fft.rfft(fr * np.hanning(win), n, axis=1)) ** 2, axis=1)[:, :win]
    lo, hi = int(SR / fmax), min(win - 2, int(SR / fmin))
    f0 = np.zeros(len(fr))
    for i in range(len(fr)):
        a = ac[i]
        if a[0] <= 0:
            continue
        seg = a[lo:hi] / a[0]
        best = float(seg.max())
        if best < 0.45:
            continue
        lag = lo + int(np.argmax(seg > 0.9 * best))
        while lag + 1 < hi and a[lag + 1] > a[lag]:
            lag += 1
        p, q, r = a[lag - 1], a[lag], a[lag + 1]
        frac = 0.5 * (p - r) / (p - 2 * q + r) if (p - 2 * q + r) else 0.0
        f0[i] = SR / (lag + frac)
    return f0, level


def bass_notes(x: np.ndarray, floor_db: float = -30.0, min_len: float = 0.05,
               jump: float = 0.8) -> list[tuple[float, float, float]]:
    """(start s, end s, midi) for each note of a mono bass stem."""
    x = listen._band(x, 25, 350) if len(x) > 4 * HOP else x
    f0, level = pitch_track(np.asarray(x, dtype=float))
    if not len(level) or level.max() <= 0:
        return []
    voiced = (f0 > 0) & (level > level.max() * 10 ** (floor_db / 20))
    midi = np.where(f0 > 0, 69 + 12 * np.log2(np.maximum(f0, 1e-6) / 440.0), 0.0)
    hits = set((listen._onsets(level, FPS, thresh=1.8, min_gap=0.06)[0] * FPS).round().astype(int))
    notes, start, cur = [], None, []

    def close(end: int) -> None:
        if start is not None and (end - start) / FPS >= min_len and cur:
            notes.append((start / FPS, end / FPS, float(np.median(cur))))

    i = 0
    while i < len(f0):
        if not voiced[i]:
            close(i)
            start, cur = None, []
            i += 1
            continue
        moved = (cur and abs(midi[i] - np.median(cur)) > jump and
                 all(voiced[j] and abs(midi[j] - midi[i]) < jump for j in range(i, min(i + 3, len(f0)))))
        if start is None or (i in hits and i - start > 2) or moved:
            close(i)
            start, cur = i, []
        cur.append(midi[i])
        i += 1
    close(len(f0))
    return notes


def onset_times(x: np.ndarray, lo: float, hi: float, thresh: float = 1.8) -> np.ndarray:
    """Onsets (s) of one band of a mono stem."""
    env = listen._envelope(listen._band(x, lo, hi))
    return listen._onsets(env, FPS, thresh=thresh)[0]


def _flux(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    env = listen._envelope(listen._band(x, lo, hi))
    log = np.log(env + 1e-6)
    return np.maximum(0.0, np.diff(log, prepend=log[0]))


def beat_grid(drums: np.ndarray, target_bpm: float, spread: float = 0.2) -> np.ndarray:
    """Beat times (s) from a mono drum stem, at the tempo nearest
    `target_bpm` (within `spread`), each beat nudged onto the nearest kick or
    snare so the grid follows the band."""
    flux = _flux(drums, 40, 250)
    times = listen._onsets(listen._envelope(listen._band(drums, 40, 250)), FPS)[0]
    bpm = listen._tempo(times, lo=target_bpm * (1 - spread), hi=target_bpm * (1 + spread),
                        length=len(drums) / SR) or target_bpm
    period = 60.0 / bpm * FPS
    # The phase where the most low-end energy sits on the beat.
    k = np.arange(int((len(flux) - 1) / period))
    if not len(k):
        return np.array([])
    phases = np.arange(0, period, 0.5)
    score = [flux[np.clip((ph + k * period).round().astype(int), 0, len(flux) - 1)].sum()
             for ph in phases]
    ph = float(phases[int(np.argmax(score))])
    beats = ph + k * period
    # Beat by beat: the strongest flux within an eighth of a beat, smoothed so
    # one ghost note can't drag the grid.
    w = max(1, int(period / 8))
    off = []
    for b in beats:
        i = int(round(b))
        lo, hi = max(0, i - w), min(len(flux), i + w + 1)
        seg = flux[lo:hi]
        off.append((lo + int(np.argmax(seg)) - b) if len(seg) and seg.max() > 0 else 0.0)
    off = np.array(off)
    smooth = np.array([np.median(off[max(0, j - 4): j + 5]) for j in range(len(off))])
    return (beats + smooth) / FPS


def to_step(t: float, beats: np.ndarray, first: int) -> float | None:
    """A time as a fractional step counted from beat `first` (4 steps a beat)."""
    j = int(np.searchsorted(beats, t, side="right")) - 1
    if j < 0 or j + 1 >= len(beats):
        return None
    frac = (t - beats[j]) / (beats[j + 1] - beats[j])
    return 4.0 * (j - first + frac)


def bar_phase(beats: np.ndarray, notes, drums: np.ndarray | None = None) -> int:
    """Which of the first four beats is the one: where bass notes start and
    the kick lands together most."""
    # The kick's level, not its rise: the one is struck harder, and a rise in
    # log level can't tell a hard kick from a soft one.
    kick = listen._envelope(listen._band(drums, 40, 120)) if drums is not None else None
    best, score = 0, -1.0
    for r in range(4):
        downs = beats[r::4]
        if not len(downs):
            continue
        s = sum(1.0 for t0, _, _ in notes if np.min(np.abs(downs - t0)) < 0.06) / len(downs)
        if kick is not None:
            w = int(0.03 * FPS) + 1
            s += float(np.mean([kick[max(0, i - w): i + w + 1].max() for i in
                                np.clip((downs * FPS).round().astype(int), 0, len(kick) - 1)]
                               )) / (float(kick.max()) or 1.0)
        if s > score:
            best, score = r, s
    return best


def bars(notes, beats: np.ndarray, phase: int, tolerance: float = 0.35) -> list[list[tuple]]:
    """Notes bar by bar on the 16-step grid: [(step, length in steps, midi,
    offset from the step)].
    Only whole bars inside the grid; notes too far off a step are left out.
    The offsets are kept for the swing reading."""
    n = (len(beats) - 1 - phase) // 4
    out: list[list[tuple]] = [[] for _ in range(max(0, n))]
    for t0, t1, m in notes:
        s = to_step(t0, beats, phase)
        if s is None:
            continue
        q = int(round(s))
        if abs(s - q) > tolerance:
            continue
        b, step = divmod(q, STEPS_PER_BAR)
        if not 0 <= b < len(out):
            continue
        e = to_step(t1, beats, phase)
        length = max(0.25, (e - s) if e is not None else 1.0)
        out[b].append((step, round(float(length), 2), round(float(m), 2), round(float(s - q), 3)))
    return out


def mask(bar) -> int:
    """A bar's onsets as a 16-bit mask, bit k for step k."""
    m = 0
    for note in bar:
        m |= 1 << int(note[0])
    return m


def mask_string(m: int, steps: int = STEPS_PER_BAR) -> str:
    return "".join("x" if m >> k & 1 else "." for k in range(steps))
