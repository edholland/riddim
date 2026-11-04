"""Hearing the real rig: send MIDI, record what comes back, measure it.

The closed loop with no person in it. A *take* plays a list of timed MIDI
messages into the rig while recording every input of the audio interface
(the L6max gives one per mixer channel), then `analyse` says what each channel
actually did: did it sound, how late, how loud, what pitch, how bright.

The MIDI send time and the audio share one clock (the stream's own), so the
latency reported is the rig's and not the scheduler's.

`Backend` is the only part that touches hardware. `SimRig` stands in for it in
the tests and with `--sim`; its numbers say nothing about the real rig.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

SILENCE_MARGIN_DB = 12.0  # a channel "sounds" this far above its own noise
FLOOR_DB = -120.0
CLIP_DBFS = -0.1


class RigError(RuntimeError):
    pass


def db(x: float) -> float:
    return FLOOR_DB if x <= 1e-6 else max(FLOOR_DB, 20.0 * float(np.log10(x)))


def note_name(midi: float) -> str:
    names = "C C# D D# E F F# G G# A A# B".split()
    n = int(round(midi))
    return f"{names[n % 12]}{n // 12 - 1}"


_midi_warm = False


def warm_midi(tries: int = 4) -> None:
    """Start CoreMIDI from a throwaway process.

    The MIDI server quits when idle, and a first client racing its restart
    takes python-rtmidi down with an uncatchable C++ abort. An unattended loop
    must survive that, so the first touch happens somewhere it can die.
    """
    global _midi_warm
    if _midi_warm:
        return
    probe_code = "import mido; mido.get_output_names(); mido.get_input_names()"
    for attempt in range(tries):
        done = subprocess.run([sys.executable, "-c", probe_code], capture_output=True)
        if done.returncode == 0:
            _midi_warm = True
            return
        time.sleep(0.5 * (attempt + 1))
    raise RigError(f"CoreMIDI would not start: {done.stderr.decode()[-200:]}")


# -- what gets sent ----------------------------------------------------------


@dataclass(frozen=True)
class Event:
    """One MIDI message at a time in seconds from the start of the take.

    `data` is raw bytes (status first), so a note, a CC and a clock are all
    the same thing here.
    """

    at: float
    data: tuple[int, ...]


def note_events(channel: int, note: int, velocity: int, at: float, hold: float) -> list[Event]:
    """A note on `channel` (1-16) held for `hold` seconds."""
    if not 1 <= channel <= 16:
        raise RigError(f"MIDI channel {channel} is outside 1-16")
    c = channel - 1
    return [Event(at, (0x90 | c, note, velocity)), Event(at + hold, (0x80 | c, note, 0))]


def silence_events(channels) -> list[tuple[int, ...]]:
    """All-notes-off on each channel: a take must never leave a note hanging."""
    return [(0xB0 | (c - 1), 123, 0) for c in channels]


# -- the take ----------------------------------------------------------------


@dataclass
class Take:
    audio: np.ndarray  # (frames, channels) float32
    rate: int
    sent: list[tuple[float, tuple[int, ...]]]  # (seconds into the recording, data)
    device: str = ""
    midi_port: str = ""


class Backend:
    """Sends `events` while recording `seconds` of audio."""

    def record(self, events: list[Event], seconds: float, cleanup=()) -> Take:
        raise NotImplementedError


def audio_inputs() -> list[dict]:
    import sounddevice as sd

    return [
        {"index": i, "name": d["name"], "channels": d["max_input_channels"],
         "rate": d["default_samplerate"]}
        for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0
    ]


def find_input(hint: str | None) -> dict:
    """The interface to record. Prefers a hint, then the L6max, never the Mac's mic."""
    inputs = audio_inputs()
    names = ", ".join(d["name"] for d in inputs) or "none"
    wanted = [hint] if hint else ["l6", "livetrak", "zoom"]
    for want in wanted:
        for d in inputs:
            if want.lower() in d["name"].lower():
                return d
    raise RigError(f"no audio input matching {wanted[0]!r}; available: {names}")


class Hardware(Backend):
    def __init__(self, midi: str | None = None, audio: str | None = None,
                 channels: int | None = None, rate: int | None = None):
        self.midi_hint = midi
        self.device = find_input(audio)
        self.channels = channels or self.device["channels"]
        self.rate = int(rate or self.device["rate"])

    def record(self, events, seconds, cleanup=()):
        import mido
        import sounddevice as sd

        from .deluge.client import DelugeError, find_port

        warm_midi()
        try:
            port_name = find_port(mido.get_output_names(), self.midi_hint)
        except DelugeError as exc:
            raise RigError(str(exc)) from exc

        blocks: list[np.ndarray] = []
        started = threading.Event()
        start_time = [0.0]  # stream time of the first recorded sample

        def callback(indata, frames, when, status):
            if not started.is_set():
                start_time[0] = when.inputBufferAdcTime
                started.set()
            blocks.append(indata.copy())

        sent = []
        with mido.open_output(port_name) as out, sd.InputStream(
            device=self.device["index"], channels=self.channels, samplerate=self.rate,
            dtype="float32", callback=callback,
        ) as stream:
            if not started.wait(5.0):
                raise RigError(f"{self.device['name']} delivered no audio")
            try:
                for event in sorted(events, key=lambda e: e.at):
                    wait = event.at - (stream.time - start_time[0])
                    if wait > 0:
                        time.sleep(wait)
                    out.send(mido.Message.from_bytes(list(event.data)))
                    sent.append((stream.time - start_time[0], event.data))
                left = seconds - (stream.time - start_time[0])
                if left > 0:
                    time.sleep(left)
            finally:
                for data in cleanup:
                    out.send(mido.Message.from_bytes(list(data)))

        audio = (np.concatenate(blocks) if blocks
                 else np.zeros((0, self.channels), dtype=np.float32))
        return Take(audio, self.rate, sent, self.device["name"], port_name)


class SimRig(Backend):
    """A pretend rig: each MIDI channel in `routes` sounds on one audio channel.

    `routes` maps MIDI channel -> audio channel (0-based). Saw-ish tone with a
    decay, a fixed latency, and a low noise floor.
    """

    def __init__(self, routes: dict[int, int] | None = None, channels: int = 4,
                 rate: int = 48000, latency: float = 0.012, noise_db: float = -80.0,
                 level: float = 0.5, seed: int = 0):
        self.routes = {1: 1} if routes is None else routes
        self.channels = channels
        self.rate = rate
        self.latency = latency
        self.noise = 10 ** (noise_db / 20)
        self.level = level
        self.rng = np.random.default_rng(seed)

    def record(self, events, seconds, cleanup=()):
        n = int(seconds * self.rate)
        audio = (self.rng.standard_normal((n, self.channels)) * self.noise).astype(np.float32)
        ons = {}
        for e in sorted(events, key=lambda e: e.at):
            kind, chan = e.data[0] & 0xF0, (e.data[0] & 0x0F) + 1
            if kind == 0x90 and e.data[2] > 0:
                ons[(chan, e.data[1])] = e
            elif kind == 0x80 and (chan, e.data[1]) in ons:
                on = ons.pop((chan, e.data[1]))
                if chan in self.routes:
                    self._sound(audio, self.routes[chan], on, e.at)
        return Take(audio, self.rate, [(e.at, e.data) for e in events], "sim", "sim")

    def _sound(self, audio, target, on, off):
        start = int((on.at + self.latency) * self.rate)
        stop = min(len(audio), int((off + self.latency) * self.rate))
        if start >= stop:
            return
        t = np.arange(stop - start) / self.rate
        f0 = 440.0 * 2 ** ((on.data[1] - 69) / 12)
        tone = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6))
        env = np.exp(-t / 1.5) * (on.data[2] / 127) * self.level
        audio[start:stop, target] += (tone / 1.5 * env).astype(np.float32)


# -- what came back ----------------------------------------------------------


@dataclass
class ChannelReport:
    channel: int  # 1-based audio input
    sounding: bool
    noise_dbfs: float
    peak_dbfs: float
    rms_dbfs: float
    clipped: bool
    latency_ms: float | None = None
    f0_hz: float | None = None
    note: str | None = None
    cents: float | None = None  # against the note named above
    centroid_hz: float | None = None
    sustain_s: float | None = None  # above -20 dB from peak


def _envelope(x: np.ndarray, rate: int, hop_s: float = 0.001) -> tuple[np.ndarray, int]:
    hop = max(1, int(rate * hop_s))
    n = len(x) // hop
    if n == 0:
        return np.zeros(0), hop
    return np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1)), hop


def estimate_f0(x: np.ndarray, rate: int, low: float = 25.0, high: float = 2000.0) -> float | None:
    """Pitch by autocorrelation; takes the shortest lag nearly as strong as the
    best one, so a clean tone isn't read an octave down. None when unpitched."""
    x = x - np.mean(x)
    if len(x) < rate / low * 2 or not np.any(x):
        return None
    w = x * np.hanning(len(x))
    size = 1 << int(np.ceil(np.log2(2 * len(w))))
    ac = np.fft.irfft(np.abs(np.fft.rfft(w, size)) ** 2)[: len(w)]
    # Undo the taper of the window's own autocorrelation, or long lags read weak.
    wa = np.fft.irfft(np.abs(np.fft.rfft(np.hanning(len(x)), size)) ** 2)[: len(w)]
    lo, hi = int(rate / high), min(int(rate / low), len(w) // 2)
    if ac[0] <= 0 or hi <= lo + 2:
        return None
    norm = ac / np.maximum(wa, wa[0] * 1e-3) / (ac[0] / wa[0])
    seg = norm[lo:hi]
    peaks = np.flatnonzero((seg[1:-1] > seg[:-2]) & (seg[1:-1] >= seg[2:])) + 1
    if len(peaks) == 0:
        return None
    best = seg[peaks].max()
    if best < 0.5:
        return None
    i = lo + int(peaks[seg[peaks] >= 0.9 * best][0])
    a, b, c = norm[i - 1], norm[i], norm[i + 1]
    shift = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
    return rate / (i + shift)


def spectral_centroid(x: np.ndarray, rate: int) -> float | None:
    if len(x) < 64:
        return None
    mag = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    total = mag.sum()
    if total <= 0:
        return None
    return float((np.fft.rfftfreq(len(x), 1 / rate) * mag).sum() / total)


def analyse_channel(x: np.ndarray, rate: int, channel: int, first_send: float) -> ChannelReport:
    """Everything before `first_send` is the channel's own noise."""
    pre = x[: max(0, int(first_send * rate))]
    post = x[len(pre):]
    noise = db(float(np.sqrt(np.mean(pre ** 2)))) if len(pre) else FLOOR_DB
    peak = float(np.max(np.abs(post))) if len(post) else 0.0
    report = ChannelReport(
        channel=channel, sounding=False, noise_dbfs=round(noise, 1),
        peak_dbfs=round(db(peak), 1),
        rms_dbfs=round(db(float(np.sqrt(np.mean(post ** 2)))) if len(post) else FLOOR_DB, 1),
        clipped=db(peak) >= CLIP_DBFS,
    )
    env, hop = _envelope(post, rate)
    if len(env) == 0:
        return report
    env_peak = db(float(env.max()))
    if env_peak < noise + SILENCE_MARGIN_DB:
        return report
    report.sounding = True

    above = np.flatnonzero(env >= 10 ** (max(noise + SILENCE_MARGIN_DB, env_peak - 30) / 20))
    onset = int(above[0])
    report.latency_ms = round(onset * hop / rate * 1000, 1)

    held = np.flatnonzero(env >= 10 ** ((env_peak - 20) / 20))
    report.sustain_s = round(float((held[-1] - onset) * hop / rate), 3)

    # Skip the attack; measure pitch and brightness on the body of the sound.
    start = (onset + int(0.03 * rate / hop)) * hop
    stop = min(held[-1] * hop, start + int(1.0 * rate))
    body = post[start:stop]
    report.centroid_hz = (round(c, 1) if (c := spectral_centroid(body, rate)) else None)
    f0 = estimate_f0(body, rate)
    if f0:
        midi = 69 + 12 * np.log2(f0 / 440)
        report.f0_hz = round(float(f0), 2)
        report.note = note_name(midi)
        report.cents = round(float((midi - round(midi)) * 100), 1)
    return report


def analyse(take: Take) -> list[ChannelReport]:
    first = min((t for t, _ in take.sent), default=0.0)
    return [analyse_channel(take.audio[:, i], take.rate, i + 1, first)
            for i in range(take.audio.shape[1])]


# -- moves -------------------------------------------------------------------


def probe(backend: Backend, channel: int = 1, note: int = 36, velocity: int = 100,
          hold: float = 1.0, pre: float = 0.5, tail: float = 1.0) -> tuple[Take, list[ChannelReport]]:
    """One note, one recording. `pre` seconds of nothing first: that's the
    noise reference every channel is judged against."""
    events = note_events(channel, note, velocity, pre, hold)
    take = backend.record(events, pre + hold + tail, cleanup=silence_events([channel]))
    return take, analyse(take)


@dataclass
class ScanResult:
    routes: dict[int, list[int]] = field(default_factory=dict)  # MIDI ch -> audio inputs
    silent: list[int] = field(default_factory=list)
    reports: dict[int, list[ChannelReport]] = field(default_factory=dict)


def scan(backend: Backend, midi_channels=range(1, 17), note: int = 48, velocity: int = 100,
         hold: float = 0.6, pre: float = 0.4, tail: float = 0.6, on_step=None) -> ScanResult:
    """Learn the patching: fire each MIDI channel in turn, see which inputs answer."""
    result = ScanResult()
    for channel in midi_channels:
        _, reports = probe(backend, channel, note, velocity, hold, pre, tail)
        result.reports[channel] = reports
        heard = [r.channel for r in reports if r.sounding]
        if heard:
            result.routes[channel] = heard
        else:
            result.silent.append(channel)
        if on_step:
            on_step(channel, heard)
    return result


# -- keeping it --------------------------------------------------------------


def write_wav(path: Path, x: np.ndarray, rate: int) -> None:
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def save_take(take: Take, reports: list[ChannelReport], folder: Path) -> Path:
    """Stems of the channels that sounded, plus the numbers for all of them."""
    folder.mkdir(parents=True, exist_ok=True)
    for r in reports:
        if r.sounding:
            write_wav(folder / f"in{r.channel:02d}.wav", take.audio[:, r.channel - 1], take.rate)
    (folder / "report.json").write_text(json.dumps({
        "device": take.device, "midi_port": take.midi_port, "rate": take.rate,
        "seconds": round(len(take.audio) / take.rate, 3),
        "sent": [{"at": round(t, 4), "data": list(d)} for t, d in take.sent],
        "channels": [asdict(r) for r in reports],
    }, indent=2))
    return folder


def describe(reports: list[ChannelReport]) -> str:
    lines = []
    for r in reports:
        if not r.sounding:
            continue
        pitch = f"{r.note} {r.f0_hz}Hz {r.cents:+.0f}c" if r.f0_hz else "unpitched"
        lines.append(
            f"  in{r.channel:<2} {r.latency_ms:6.1f}ms  peak {r.peak_dbfs:6.1f}  "
            f"noise {r.noise_dbfs:6.1f}  {pitch}  centroid {r.centroid_hz}Hz  "
            f"sustain {r.sustain_s}s{'  CLIPPED' if r.clipped else ''}"
        )
    return "\n".join(lines) or "  nothing sounded on any input"
