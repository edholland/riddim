"""Ears: listen to a recording the way a producer would, in numbers and pictures.

The engine cannot hear its own renders, and neither can the model that works
on it - so this turns audio into what can be read and seen:

- **a picture**: a log-frequency spectrogram over the song with the band
  energies under it, and a close-up of one bar of the drop - where the kick,
  the sub, the skank and the hats are visible as shapes;
- **the groove**: tempo, where the hits fall against the sixteenth grid
  (swing is the late offbeat; push and drag are the average lean), and how
  much a drum hit varies from one to the next - identical hits are the
  sound of a sample being retriggered by a machine;
- **the weight**: how the energy splits across sub, bass, mids and air, how
  loud the song is and how much it breathes (crest, dynamic range), and how
  the energy moves across the song;
- **the space**: stereo width per band - a mono sub under a wide top.

Any file ffmpeg can read. Needs numpy, scipy and (for pictures) matplotlib.

    riddim-cli listen track.m4a --png track.png
    riddim-cli listen a.wav b.m4a        # side by side
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.signal import butter, sosfilt, stft

SR = 22050

BANDS = {             # Hz
    "sub": (20, 60),
    "bass": (60, 250),
    "low_mid": (250, 1000),
    "high_mid": (1000, 5000),
    "air": (5000, 11000),
}


def load(path: str, sr: int = SR, seconds: float | None = None) -> np.ndarray:
    """Decode any audio file to float32 stereo, shape (n, 2)."""
    cmd = ["ffmpeg", "-v", "error", "-i", path]
    if seconds:
        cmd += ["-t", str(seconds)]
    cmd += ["-f", "f32le", "-ac", "2", "-ar", str(sr), "-"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).copy()


def _band(x, lo, hi, sr=SR):
    hi = min(hi, sr * 0.45)
    sos = butter(4, [lo / (sr / 2), hi / (sr / 2)], btype="band", output="sos")
    return sosfilt(sos, x, axis=0)


def _envelope(x, sr=SR, hop=256):
    """RMS envelope at sr/hop frames per second."""
    n = len(x) // hop
    frames = x[: n * hop].reshape(n, hop)
    return np.sqrt((frames ** 2).mean(axis=1) + 1e-12)


def _onsets(env, fps, thresh=1.6, min_gap=0.07):
    """Onset times (s) and strengths from an envelope: rises in log energy
    above a local median, at least `min_gap` apart."""
    log = np.log(env + 1e-6)
    flux = np.maximum(0.0, np.diff(log, prepend=log[0]))
    w = max(3, int(fps * 0.5))
    med = np.convolve(flux, np.ones(w) / w, mode="same")
    peaks = []
    gap = int(min_gap * fps)
    for i in range(1, len(flux) - 1):
        if flux[i] > thresh * med[i] + 0.15 and flux[i] >= flux[i - 1] and flux[i] >= flux[i + 1]:
            if peaks and i - peaks[-1] < gap:
                if flux[i] > flux[peaks[-1]]:
                    peaks[-1] = i
                continue
            peaks.append(i)
    idx = np.array(peaks, dtype=int)
    return idx / fps, flux[idx] if len(idx) else np.array([])


def _tempo(onset_times, lo=60, hi=180, length=None):
    """Tempo from the autocorrelation of an onset train, folded into a range
    and preferring the dance-music doubling where it is ambiguous."""
    if len(onset_times) < 8:
        return 0.0
    fps = 200
    n = int((length or onset_times[-1] + 1) * fps)
    train = np.zeros(n)
    train[np.clip((onset_times * fps).astype(int), 0, n - 1)] = 1.0
    train = np.convolve(train, np.hanning(9), mode="same")
    ac = np.correlate(train, train, mode="full")[n - 1:]
    best, score = 0.0, -1.0
    for bpm in np.arange(lo, hi, 0.25):
        lag = 60.0 / bpm * fps
        s = sum(ac[int(round(lag * k))] for k in (1, 2, 4) if int(round(lag * k)) < len(ac))
        if s > score:
            best, score = bpm, s
    return float(best)


@dataclass
class Report:
    path: str
    seconds: float
    bpm: float
    # groove
    swing: float = 0.0            # offbeat 16ths: 0.5 straight, 0.66 triplet swing
    lean_ms: float = 0.0          # mean offset of all hits from the grid (+ late)
    jitter_ms: float = 0.0        # spread of those offsets: 0 is a machine
    hit_variation: float = 0.0    # how different successive hat/snare hits sound (0 = identical)
    # weight
    band_share: dict = field(default_factory=dict)   # fraction of energy per band
    rms_db: float = 0.0
    crest_db: float = 0.0
    dynamic_range_db: float = 0.0  # loud sections vs quiet ones (95th - 10th pct, 1s windows)
    # space
    width: dict = field(default_factory=dict)        # side/mid energy per band
    # texture
    black: float = 0.0            # share of time the whole mix is near silence (digital black)
    sub_held: float = 0.0         # share of time the sub band is sounding
    sub_runs_beats: float = 0.0   # median length of an unbroken sub note/run, in beats
    mid_fill: float = 0.0         # share of time the 250Hz-2kHz band carries sustained sound
    # shape over time
    energy_curve: list = field(default_factory=list)  # dB per 8 bars

    def text(self) -> str:
        b = " ".join(f"{k} {v:.0%}" for k, v in self.band_share.items())
        w = " ".join(f"{k} {v:.2f}" for k, v in self.width.items())
        curve = " ".join(f"{v:+.0f}" for v in self.energy_curve)
        return (f"{self.path}\n"
                f"  {self.seconds / 60:.2f} min at {self.bpm:.1f} bpm\n"
                f"  groove   swing {self.swing:.2f}  lean {self.lean_ms:+.1f}ms  "
                f"jitter {self.jitter_ms:.1f}ms  hit variation {self.hit_variation:.2f}\n"
                f"  weight   {b}\n"
                f"           {self.rms_db:.1f} dB RMS, crest {self.crest_db:.1f} dB, "
                f"dynamic range {self.dynamic_range_db:.1f} dB\n"
                f"  width    {w}\n"
                f"  texture  black {self.black:.1%}  sub held {self.sub_held:.0%} "
                f"(runs {self.sub_runs_beats:.1f} beats)  mid fill {self.mid_fill:.0%}\n"
                f"  energy per 8 bars (dB vs mean): {curve}")


def analyze(path: str, bpm: float | None = None, seconds: float | None = None) -> Report:
    x = load(path, seconds=seconds)
    mono = x.mean(axis=1)
    side = (x[:, 0] - x[:, 1]) / 2
    secs = len(mono) / SR
    hop = 256
    fps = SR / hop

    # Drums by band: kick low, snare mid, hats high.
    kick_env = _envelope(_band(mono, 35, 120), hop=hop)
    snare_env = _envelope(_band(mono, 180, 2500), hop=hop)
    hat_env = _envelope(_band(mono, 6000, 10500), hop=hop)
    k_on, _ = _onsets(kick_env, fps)
    s_on, _ = _onsets(snare_env, fps)
    h_on, _ = _onsets(hat_env, fps, min_gap=0.05)
    all_on = np.sort(np.concatenate([k_on, s_on, h_on]))
    if bpm is None:
        bpm = _tempo(all_on, length=secs)
        # Half-time music reads at its felt tempo: fold into the written range.
        if 60 <= bpm < 95 and len(h_on) / max(1.0, secs) > 3.5:
            bpm *= 2
    rep = Report(path=path, seconds=secs, bpm=bpm)

    if bpm > 0 and len(all_on) > 16:
        six = 60.0 / bpm / 4            # one sixteenth, seconds
        # Phase: the grid offset that best fits the kicks (or everything).
        ref = k_on if len(k_on) > 8 else all_on
        phases = np.linspace(0, six, 48, endpoint=False)
        cost = [np.mean(np.abs(((ref - p + six / 2) % six) - six / 2)) for p in phases]
        phase = phases[int(np.argmin(cost))]
        pos = (all_on - phase) / six
        idx = np.round(pos)
        off = (pos - idx) * six * 1000           # ms from the grid
        near = np.abs(off) < six * 1000 * 0.4
        # Swing from the hats: how far through the eighth the offbeat 16th lands.
        hp = (h_on - phase) / six
        eighth_pos = (hp % 2) / 2                 # 0 .. 1 through an eighth
        late = eighth_pos[(eighth_pos > 0.3) & (eighth_pos < 0.8)]
        rep.swing = float(np.median(late)) if len(late) > 8 else 0.5
        on_grid = off[near & (np.abs(off) < 25)]
        rep.lean_ms = float(np.mean(on_grid)) if len(on_grid) else 0.0
        rep.jitter_ms = float(np.std(on_grid)) if len(on_grid) else 0.0

    # Hit variation: the spectral shape of successive hat hits.
    rep.hit_variation = _hit_variation(mono, h_on)

    total = 0.0
    shares = {}
    for name, (lo, hi) in BANDS.items():
        e = float(np.mean(_band(mono, lo, hi) ** 2))
        shares[name] = e
        total += e
        se = float(np.mean(_band(side, lo, hi) ** 2))
        rep.width[name] = round(float(np.sqrt(se / (e + 1e-12))), 3)
    rep.band_share = {k: round(v / (total or 1.0), 4) for k, v in shares.items()}

    _texture(rep, mono, bpm)

    rms = np.sqrt(np.mean(mono ** 2) + 1e-12)
    rep.rms_db = float(20 * np.log10(rms))
    rep.crest_db = float(20 * np.log10(np.max(np.abs(mono)) / rms))
    win = SR
    n = len(mono) // win
    if n >= 4:
        loud = 20 * np.log10(np.sqrt((mono[: n * win].reshape(n, win) ** 2).mean(axis=1)) + 1e-9)
        loud = loud[loud > loud.max() - 50]
        rep.dynamic_range_db = float(np.percentile(loud, 95) - np.percentile(loud, 10))
    if bpm > 0:
        block = int(8 * 4 * 60.0 / bpm * SR)
        m = len(mono) // block
        if m >= 1:
            e = 10 * np.log10((mono[: m * block].reshape(m, block) ** 2).mean(axis=1) + 1e-12)
            rep.energy_curve = [round(float(v), 1) for v in e - e.mean()]
    return rep


def _texture(rep, mono, bpm, hop=256):
    """How full the sound is from moment to moment. Judged on the body of
    the track (frames within 30 dB of its loud level), so intros that fade
    in from nothing do not count against it."""
    fps = SR / hop
    whole = 20 * np.log10(_envelope(mono, hop=hop) + 1e-9)
    loud = np.percentile(whole, 95)
    body = whole > loud - 30
    rep.black = float(np.mean(whole[body] < loud - 45)) if body.any() else 0.0
    # Inside the body: sections where the music plays.
    sub = 20 * np.log10(_envelope(_band(mono, 25, 70), hop=hop) + 1e-9)
    sub_on = sub > np.percentile(sub, 90) - 18
    rep.sub_held = float(np.mean(sub_on[body]))
    runs, k = [], 0
    for v in sub_on:
        if v:
            k += 1
        elif k:
            runs.append(k)
            k = 0
    if k:
        runs.append(k)
    beat = 60.0 / (bpm or 140.0)
    long_runs = [r / fps / beat for r in runs if r / fps > 0.05]
    rep.sub_runs_beats = float(np.median(long_runs)) if long_runs else 0.0
    mid = 20 * np.log10(_envelope(_band(mono, 250, 2000), hop=hop) + 1e-9)
    # "Sustained": the band within 20 dB of its loud level, not just the
    # instant of a hit - smoothed over a sixteenth so a click does not fill.
    k16 = max(1, int(fps * beat / 4))
    mid_s = np.convolve(mid, np.ones(k16) / k16, mode="same")
    rep.mid_fill = float(np.mean((mid_s > np.percentile(mid, 95) - 20)[body]))


def _hit_variation(mono, onsets, n=1024, max_hits=200):
    """Mean spectral distance between successive hits, normalised: a drum
    machine retriggering one sample scores near 0, a player or a desk with
    things moving on it well above."""
    specs = []
    for t in onsets[:max_hits]:
        i = int(t * SR)
        seg = mono[i:i + n]
        if len(seg) < n:
            break
        s = np.abs(np.fft.rfft(seg * np.hanning(n)))[n // 8:]
        s = np.log(s + 1e-6)
        specs.append(s - s.mean())
    if len(specs) < 4:
        return 0.0
    d = [np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-9) for a, b in zip(specs, specs[1:])]
    return float(np.median(d))


def picture(path: str, out: str, bpm: float | None = None, title: str | None = None,
            seconds: float | None = None, zoom_at: float | None = None) -> str:
    """A spectrogram of the whole track with band energies under it, and a
    two-bar close-up (at `zoom_at` seconds, default the loudest stretch)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = load(path, seconds=seconds)
    mono = x.mean(axis=1)
    secs = len(mono) / SR
    f, t, z = stft(mono, fs=SR, nperseg=2048, noverlap=2048 - 512)
    db = 20 * np.log10(np.abs(z) + 1e-7)
    fig = plt.figure(figsize=(16, 10), dpi=80)
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1.2, 3])

    ax = fig.add_subplot(gs[0])
    ax.pcolormesh(t, f[1:], db[1:], shading="auto", vmin=db.max() - 80, vmax=db.max(), cmap="magma")
    ax.set_yscale("log")
    ax.set_ylim(25, 11000)
    ax.set_ylabel("Hz")
    ax.set_title(title or path)

    ax2 = fig.add_subplot(gs[1], sharex=ax)
    hop = 1024
    for name, (lo, hi) in BANDS.items():
        env = _envelope(_band(mono, lo, hi), hop=hop)
        tt = np.arange(len(env)) * hop / SR
        ax2.plot(tt, 20 * np.log10(env + 1e-6), lw=0.8, label=name)
    ax2.set_ylabel("dB")
    ax2.legend(loc="lower right", ncol=5, fontsize=8)
    ax2.set_xlim(0, secs)

    # Close-up: two bars where the song is loudest (the drop).
    bar = 4 * 60.0 / (bpm or 140.0)
    if zoom_at is None:
        env = _envelope(mono, hop=SR // 2)
        k = max(1, int(8 * bar * 2))
        smooth = np.convolve(env, np.ones(k) / k, mode="same")
        zoom_at = float(np.argmax(smooth)) * 0.5
    a = int(zoom_at * SR)
    b = min(len(mono), a + int(2 * bar * SR))
    f2, t2, z2 = stft(mono[a:b], fs=SR, nperseg=1024, noverlap=1024 - 128)
    db2 = 20 * np.log10(np.abs(z2) + 1e-7)
    ax3 = fig.add_subplot(gs[2])
    ax3.pcolormesh(t2, f2[1:], db2[1:], shading="auto", vmin=db2.max() - 70, vmax=db2.max(), cmap="magma")
    ax3.set_yscale("log")
    ax3.set_ylim(25, 11000)
    for k in range(0, 33):     # the sixteenth grid
        ax3.axvline(k * bar / 16, color="w", lw=0.6 if k % 4 else 1.2, alpha=0.25 if k % 4 else 0.5)
    ax3.set_title(f"two bars from {zoom_at:.1f}s (lines: sixteenths, bold: beats)")
    ax3.set_xlabel("s")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def to_json(rep: Report) -> str:
    return json.dumps(asdict(rep), indent=1)
