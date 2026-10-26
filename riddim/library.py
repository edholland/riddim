"""The sample library: recorded one-shots in place of the preview's synths.

The free packs in `samples/packs/` (fetched by `ml/fetch_samples.py`, each
licence in `LICENCES.tsv` there) hold what the records are made of and the
synthesised stand-ins only imitate: drawbar organs, a Telecaster skank,
drum-machine hats, hardware subs, sirens. This finds them, says what each
file is, and plays them as instruments:

    riddim-cli samples index            # samples/index.jsonl: every WAV, its role
    riddim-cli samples picks            # what each family plays from the library

What is used where (`PICKS`, matched against each file's path):

    organ    the reggae bubble and the garage/140 organ stabs - a drawbar
             organ multisample, played at the nearest sampled note
    skank    dub's guitar chop - a recorded chord stab (major or minor, by
             root), the nearest one transposed to the chord being played
    sub      the 140 sub - a hardware synth's sub multisample, looped to length
    hat      garage hats - a drum machine's closed and open hats
    shaker   garage shaker
    siren    the sound-system siren (kit row 51)

Nothing from the library is required: with no packs on disk every voice
falls back to the synth that was there before, so a fresh clone, the tests
and the cloud all render as they did. `RIDDIM_LIBRARY` points at another
packs folder; `RIDDIM_LIBRARY=off` turns the library off.

The packs' terms allow their use in your own music but not passing the raw
files on: nothing here is committed, and kits made from them stay on your
card.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import wave
from dataclasses import asdict, dataclass

import numpy as np

SR = 44100
AUDIO_EXT = (".wav", ".flac", ".aif", ".aiff")
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ROOT = os.path.join(_REPO, "samples", "packs")

NOTE_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

# --------------------------------------------------------------------------
# What a file is, from its path
# --------------------------------------------------------------------------

# Ordered: the first rule whose pattern matches the lowercased path wins.
# Folder names in the packs are consistent enough that the path says what a
# file is far more reliably than the audio would.
ROLE_RULES: list[tuple[str, str]] = [
    ("siren", r"siren|alarm"),
    ("vox", r"vocal|vox|acapella|ad-?lib|female|male vox|quotes|hooks"),
    ("organ", r"organ"),
    ("skank", r"guitar chop|telechop|stratchop"),
    ("guitar", r"guitar"),
    ("keys", r"piano|rhodes|clavinet|keys|chord hits"),
    ("sub", r"sub bass weight/multis|/multis/[^/]*sub|sine|sub kicks|/sub\d"),
    ("kick", r"bass drum|kick|\bbd\b|_bd"),
    ("snare", r"snare|\bsd\b|snr"),
    ("clap", r"clap"),
    ("hat_open", r"open (hi )?hat|open hi hat|\boh\b"),
    ("hat", r"hat|\bch\b"),
    ("shaker", r"shaker|maracas"),
    ("rim", r"rim|stick|claves"),
    ("cymbal", r"cymbal|crash|ride|gong"),
    ("tom", r"tom"),
    ("perc", r"perc|conga|bongo|cowbell|block|tamb"),
    ("bass", r"bass"),
    ("brass", r"brass|horn|sax|trumpet|trombone"),
    ("fx", r"(?<![a-z])fx|sweep|riser|noise|texture|dub ?out"),
    ("beat", r"beat|loop|drum"),
]

_DRUMS = ("kick", "snare", "clap", "hat_open", "hat", "shaker", "rim", "cymbal", "tom", "perc")
_LOOP = re.compile(r"loop|beats?\b|lines|bpm|\(\d+ ?bpm\)", re.I)
_BPM = re.compile(r"(?<!\d)(\d{2,3}) ?bpm|[_ -](\d{2,3})(?:[a-g]?)[-_ ]\d{2}\b|_(\d{2,3})-[a-g]", re.I)
_NOTE = re.compile(r"[ _-]([A-G])([#b]?)(-?\d)$")
_CHORD = re.compile(r"[ _-]([A-G])([#b]?)(maj|min)$", re.I)


def role_of(path: str) -> str:
    """What a file is, by its folders and name. In a kit's folder the file
    name decides (a pack called "acapellas" has drum kits in it too)."""
    p = path.lower().replace("\\", "/")
    if re.search(r"drum ?kit|drum.?hits|/hits/", p):
        # A kit's folder: the file's own name says which drum it is.
        name = p.rsplit("/", 1)[-1]
        for role, pat in ROLE_RULES:
            if role in _DRUMS and re.search(pat, name):
                return role
    for role, pat in ROLE_RULES:
        if re.search(pat, p):
            return role
    return "other"


def bpm_of(path: str) -> int | None:
    for m in _BPM.finditer(path):
        v = int(next(g for g in m.groups() if g))
        if 60 <= v <= 200:
            return v
    return None


def note_of(path: str) -> int | None:
    """A multisample zone's MIDI note from its name ("SK_Organ01_D#3",
    "Solid Sub F#1"): C3 is MIDI 60, as the packs (and the Deluge) count."""
    m = _NOTE.search(os.path.splitext(os.path.basename(path))[0])
    if not m:
        return None
    pc = NOTE_PC[m.group(1)] + {"#": 1, "b": -1, "": 0}[m.group(2)]
    return 60 + 12 * (int(m.group(3)) - 3) + pc


def chord_of(path: str) -> tuple[int, str] | None:
    """A chord stab's (root pitch class, "maj"/"min") from its name."""
    m = _CHORD.search(os.path.splitext(os.path.basename(path))[0])
    if not m:
        return None
    pc = (NOTE_PC[m.group(1)] + {"#": 1, "b": -1, "": 0}[m.group(2)]) % 12
    return pc, m.group(3).lower()


@dataclass
class Entry:
    path: str                 # relative to the library root
    pack: str
    role: str
    bpm: int | None = None
    note: int | None = None
    chord: str | None = None  # "A min"
    loop: bool = False        # played in time (a loop or line), not a one-shot
    licence: str = ""
    seconds: float | None = None
    peak_db: float | None = None
    rms_db: float | None = None


def _licences(root: str) -> dict[str, str]:
    out = {}
    path = os.path.join(root, "LICENCES.tsv")
    if os.path.exists(path):
        for line in open(path):
            cols = line.rstrip("\n").split("\t")
            if len(cols) >= 3:
                out[os.path.splitext(cols[0])[0]] = cols[2]
    return out


def scan(root: str = DEFAULT_ROOT, analyse: bool = False, progress=None) -> list[Entry]:
    """Every audio file under `root`, what it is, and (with `analyse`) its
    length and level."""
    lic = _licences(root)
    out = []
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d != "__MACOSX" and not d.startswith("."))
        for f in sorted(files):
            if not f.lower().endswith(AUDIO_EXT) or f.startswith("._"):
                continue
            full = os.path.join(dirpath, f)
            rel = os.path.relpath(full, root)
            pack = rel.split(os.sep, 1)[0]
            ch = chord_of(rel)
            e = Entry(path=rel, pack=pack, role=role_of(rel), bpm=bpm_of(rel), note=note_of(rel),
                      chord=f"{'C C# D D# E F F# G G# A A# B'.split()[ch[0]]} {ch[1]}" if ch else None,
                      loop=bool(_LOOP.search(rel.split(os.sep, 1)[-1])),
                      licence=lic.get(pack, ""))
            if analyse:
                try:
                    x = load(full).mean(axis=1)
                    e.seconds = round(len(x) / SR, 3)
                    e.peak_db = round(20 * np.log10(float(np.max(np.abs(x))) + 1e-9), 1)
                    e.rms_db = round(10 * np.log10(float(np.mean(x ** 2)) + 1e-12), 1)
                except Exception as err:  # noqa: BLE001 - a bad file is skipped, not fatal
                    if progress:
                        progress(f"unreadable {rel}: {err}")
            out.append(e)
            if progress and len(out) % 2000 == 0:
                progress(f"{len(out)} files")
    return out


def write_index(entries: list[Entry], path: str) -> str:
    with open(path, "w") as f:
        for e in entries:
            f.write(json.dumps({k: v for k, v in asdict(e).items() if v is not None}) + "\n")
    return path


# --------------------------------------------------------------------------
# Reading audio
# --------------------------------------------------------------------------

_CACHE: dict[str, np.ndarray] = {}


def load(path: str) -> np.ndarray:
    """A file as float32 stereo at 44.1 kHz, cached. WAV is read directly
    (8/16/24/32-bit PCM); anything else - float WAV, FLAC, AIFF - through
    soundfile if it is installed, else ffmpeg."""
    if path in _CACHE:
        return _CACHE[path]
    x, sr = None, SR
    try:
        with wave.open(path) as w:
            sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
            raw = w.readframes(w.getnframes())
        if width == 3:
            b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
            v = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8)
                 | (b[:, 2].astype(np.int32) << 16))
            v = np.where(v >= 1 << 23, v - (1 << 24), v)
            x = v.astype(np.float32) / (1 << 23)
        else:
            dt = {1: np.uint8, 2: "<i2", 4: "<i4"}[width]
            x = np.frombuffer(raw, dtype=dt).astype(np.float32)
            x = (x - 128) / 128 if width == 1 else x / float(1 << (8 * width - 1))
        x = x.reshape(-1, ch)
    except Exception:  # noqa: BLE001 - not a plain PCM WAV
        try:
            import soundfile
            x, sr = soundfile.read(path, dtype="float32", always_2d=True)
        except ImportError:
            raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "2",
                                  "-ar", str(SR), "-"], check=True, capture_output=True).stdout
            x, sr = np.frombuffer(raw, dtype=np.float32).reshape(-1, 2), SR
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    x = x[:, :2]
    if sr != SR:
        from scipy.signal import resample_poly
        g = np.gcd(int(sr), SR)
        x = resample_poly(x, SR // g, int(sr) // g, axis=0)
    x = np.ascontiguousarray(x, dtype=np.float32)
    _CACHE[path] = x
    return x


def _resample(x: np.ndarray, rate: float) -> np.ndarray:
    """Play `x` back at `rate` (pitch and length together, as a sampler does)."""
    if abs(rate - 1.0) < 1e-4:
        return x
    idx = np.arange(0, len(x) - 1, rate)
    if x.ndim == 1:
        return np.interp(idx, np.arange(len(x)), x).astype(np.float32)
    return np.stack([np.interp(idx, np.arange(len(x)), x[:, c]) for c in range(x.shape[1])],
                    axis=1).astype(np.float32)


def _env_db(x: np.ndarray, hop: int) -> np.ndarray:
    m = x.mean(axis=1) if x.ndim == 2 else x
    n = len(m) // hop
    if n == 0:
        return np.array([-120.0])
    frames = m[: n * hop].reshape(n, hop)
    return 10 * np.log10((frames ** 2).mean(axis=1) + 1e-12)


def first_hit(x: np.ndarray, max_seconds: float = 0.6) -> np.ndarray:
    """The first hit of a loop of hits (a guitar chop played in time): from
    its onset until the next onset, the level falling 30 dB, or
    `max_seconds` - with a short fade so it never clicks."""
    hop = 256
    db = _env_db(x, hop)
    top = float(db.max())
    on = int(np.argmax(db > top - 24))
    start = max(0, on * hop - hop)
    peak = on + int(np.argmax(db[on:on + 40]))
    end = len(db)
    for i in range(peak + 2, len(db)):
        if db[i] < top - 30 or (db[i] > db[i - 2] + 9 and db[i] > top - 18):
            end = i
            break
    stop = min(len(x), end * hop, start + int(max_seconds * SR))
    y = x[start:stop].copy()
    n = min(len(y), int(0.012 * SR))
    y[len(y) - n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32)[:, None]
    return y


def _held(x: np.ndarray, n: int) -> np.ndarray:
    """`x` lasting `n` samples: cut (with a release) if it is longer, its
    sustain looped with crossfades if it is shorter - a sampler's loop."""
    if len(x) >= n:
        return x[:n].copy()
    seg_n = max(int(0.12 * SR), len(x) // 3)
    seg = x[len(x) - seg_n - int(0.02 * SR): len(x) - int(0.02 * SR)]
    out = x.copy()
    fade = min(len(seg) // 2, int(0.03 * SR))
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)[:, None]
    while len(out) < n:
        tail = out[-fade:] * (1 - ramp) + seg[:fade] * ramp
        out = np.concatenate([out[:-fade], tail, seg[fade:]])
    return out[:n]


# --------------------------------------------------------------------------
# Instruments
# --------------------------------------------------------------------------

def _release(y: np.ndarray, seconds: float) -> np.ndarray:
    r = min(len(y), max(1, int(seconds * SR)))
    y = y.copy()
    y[len(y) - r:] *= np.linspace(1.0, 0.0, r, dtype=np.float32)[:, None]
    return y


def f0(x: np.ndarray) -> float | None:
    """The fundamental of a pitched one-shot (autocorrelation over its
    first half second, after the attack), or None if it has none."""
    m = x.mean(axis=1) if x.ndim == 2 else x
    m = m[int(0.03 * SR): int(0.53 * SR)]
    if len(m) < 4096 or float(np.max(np.abs(m))) < 1e-4:
        return None
    m = m - m.mean()
    n = 1 << int(np.ceil(np.log2(2 * len(m))))
    ac = np.fft.irfft(np.abs(np.fft.rfft(m, n)) ** 2)[: len(m)]
    if ac[0] <= 0:
        return None
    # Past the zero-lag lobe (where the signal still resembles itself just
    # because it is smooth), up to 25 Hz.
    neg = np.nonzero(ac[: int(SR / 25)] < 0)[0]
    lo, hi = (int(neg[0]) if len(neg) else int(SR / 1000)), int(SR / 25)
    seg = ac[lo:hi]
    if len(seg) == 0:
        return None
    # The first lag that comes near the best one: the fundamental, not a
    # multiple of its period.
    best = float(seg.max())
    lag = lo + int(np.argmax(seg > 0.9 * best))
    while lag + 1 < hi and ac[lag + 1] > ac[lag]:
        lag += 1                                   # up to that peak's top
    if ac[lag] < 0.3 * ac[0]:
        return None
    a, b, c = ac[lag - 1], ac[lag], ac[lag + 1]    # and between samples
    frac = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) else 0.0
    return SR / (lag + frac)


def octave_offset(zones: dict[int, str]) -> int:
    """How many octaves the packs' note names are off from MIDI as riddim
    counts it (C3 = 60): packs disagree - one calls 110 Hz "A1", another
    calls 55 Hz "A1". Measured on the zones, the median wins."""
    offs = []
    for midi, path in zones.items():
        f = f0(load(path))
        if f:
            offs.append(round(12 * np.log2(f / (440.0 * 2 ** ((midi - 69) / 12.0))) / 12))
    return int(np.median(offs)) if offs else 0


class Multisample:
    """A pitched instrument from note-named zones: each note played from the
    nearest zone (below first, as a sampler maps them), transposed. The
    zones' octave is checked against their pitch (`octave_offset`)."""

    def __init__(self, zones: dict[int, str], name: str = "", check_octave: bool = True):
        k = octave_offset(zones) if check_octave else 0
        self.zones = dict(sorted((m + 12 * k, p) for m, p in zones.items()))
        self.name = name

    def play(self, midi: int, seconds: float, vel: int, release: float = 0.06,
             loop: bool = False) -> np.ndarray:
        root = min(self.zones, key=lambda z: (abs(z - midi), z > midi))
        x = _resample(load(self.zones[root]), 2 ** ((midi - root) / 12.0))
        n = int((seconds + release) * SR)
        y = _held(x, n) if loop else x[:n].copy()
        return _release(y, release) * ((vel / 127.0) ** 1.3)


class ChordStabs:
    """Recorded chord stabs (a guitar's skank), one per root and quality:
    a chord is played by the nearest stab of its quality, transposed to its
    root - never more than a tritone, so the guitar keeps its voice."""

    def __init__(self, stabs: dict[tuple[int, str], str], name: str = ""):
        self.stabs = stabs
        self.name = name

    def play(self, root_pc: int, quality: str, seconds: float, vel: int) -> np.ndarray:
        pool = [k for k in self.stabs if k[1] == quality] or list(self.stabs)

        def shift(k):
            d = (root_pc - k[0]) % 12
            return d - 12 if d > 6 else d

        key = min(pool, key=lambda k: (abs(shift(k)), k))
        x = first_hit(load(self.stabs[key]))
        y = _resample(x, 2 ** (shift(key) / 12.0))
        y = y[: int((min(seconds, 0.4) + 0.12) * SR)]
        return _release(y, 0.05) * ((vel / 127.0) ** 1.3)


class OneShots:
    """Unpitched hits, a different take each time (the round-robin a drum
    machine's samples do not have)."""

    def __init__(self, paths: list[str], name: str = ""):
        self.paths = paths
        self.name = name

    def play(self, rng, vel: int, seconds: float | None = None) -> np.ndarray:
        x = load(self.paths[int(rng.integers(0, len(self.paths)))])
        if seconds is not None:
            x = _release(x[: int((seconds + 0.03) * SR)], 0.03)
        return x * ((vel / 127.0) ** 1.3)


# --------------------------------------------------------------------------
# The picks: what each family plays
# --------------------------------------------------------------------------

# For each voice, path patterns in order of preference; a song draws among
# the first pattern's matches by its seed (the same song always gets the same
# organ). Edit these to audition other sounds - `riddim-cli samples picks`
# shows what they resolve to.
PICKS: dict[str, dict[str, list[str]]] = {
    "dub": {
        "organ": [r"Soulful Keys/Multisamples/Organ0[234]/"],
        "skank": [r"Guitar Chops/DD_TeleChop_", r"Guitar Chops/DD_StratChop_"],
        "siren": [r"alarm-siren-samples/100bpm/AaS_(Echo|Verb)Proc"],
    },
    "dubstep": {
        "organ": [r"Soulful Keys/Multisamples/Organ0[34]/"],
        "sub": [r"Sub Bass Weight/Multis/(Soft Sub|Solid Sub|Firm Sub|Juno Sub|SH101 Sub)/"],
        "siren": [r"alarm-siren-samples/140bpm/AaS_(Echo|Verb|Filter)Proc"],
    },
    "garage": {
        "organ": [r"Soulful Keys/Multisamples/Organ0[1234]/"],
        "hat": [r"Hats/Drum machine/Closed hat", r"Nu Garage/Drum_hits/Hat"],
        "hat_open": [r"Hats/Drum machine/Open hat"],
        "shaker": [r"Drum Kits/Kit . Hits/UK_DHit.-Shaker"],
        "siren": [r"alarm-siren-samples/140bpm/AaS_(Filter|Modu)Proc"],
    },
}


class Library:
    """The packs on disk, and the instruments the picks make from them."""

    def __init__(self, root: str):
        self.root = root
        self._files: list[str] | None = None

    @property
    def files(self) -> list[str]:
        if self._files is None:
            self._files = [e.path.replace(os.sep, "/") for e in scan(self.root)]
        return self._files

    def match(self, pattern: str) -> list[str]:
        rx = re.compile(pattern, re.I)
        return [f for f in self.files if rx.search(f)]

    def _draw(self, family: str, voice: str, seed: int) -> list[str]:
        """The files a song plays for `voice`: the first pick with matches,
        and within it one sound (a folder, or a file stem's family) chosen by
        the song's seed."""
        for pat in PICKS.get(family, {}).get(voice, []):
            got = self.match(pat)
            if got:
                return got
        return []

    def _group(self, files: list[str], seed: int, key) -> list[str]:
        groups: dict[str, list[str]] = {}
        for f in files:
            groups.setdefault(key(f), []).append(f)
        names = sorted(groups)
        return groups[names[seed % len(names)]] if names else []

    def organ(self, family: str, seed: int) -> Multisample | None:
        files = self._group(self._draw(family, "organ", seed), seed, os.path.dirname)
        zones = {note_of(f): os.path.join(self.root, f) for f in files if note_of(f) is not None}
        # An organ's pitch is its 8' drawbar's; the 16' an octave under it
        # fools a pitch detector, so the names are trusted.
        return Multisample(zones, os.path.dirname(files[0]), check_octave=False) if zones else None

    def sub(self, family: str, seed: int) -> Multisample | None:
        files = self._group(self._draw(family, "sub", seed), seed, os.path.dirname)
        zones = {note_of(f): os.path.join(self.root, f) for f in files if note_of(f) is not None}
        return Multisample(zones, os.path.dirname(files[0])) if zones else None

    def skank(self, family: str, seed: int, bpm: float) -> ChordStabs | None:
        files = self._draw(family, "skank", seed)
        if not files:
            return None
        # The chop recorded nearest the song's tempo: its length and its
        # choke are the player's at that tempo.
        tempos = sorted({bpm_of(f) or 0 for f in files})
        t = min(tempos, key=lambda v: abs(v - bpm))
        files = [f for f in files if (bpm_of(f) or 0) == t]
        files = self._group(files, seed, lambda f: re.sub(r"_\d+-[A-G].*$", "", os.path.basename(f)))
        stabs = {}
        for f in files:
            ch = chord_of(f)
            if ch:
                stabs[ch] = os.path.join(self.root, f)
        return ChordStabs(stabs, os.path.basename(files[0]).split("-")[0]) if stabs else None

    def shots(self, family: str, voice: str, seed: int = 0) -> OneShots | None:
        files = self._draw(family, voice, seed)
        return OneShots([os.path.join(self.root, f) for f in files], voice) if files else None

    def siren(self, family: str, seed: int) -> str | None:
        files = self._draw(family, "siren", seed)
        return os.path.join(self.root, files[seed % len(files)]) if files else None


_LIB: dict[str, Library | None] = {}


def get(root: str | None = None) -> Library | None:
    """The library at `root` (default `$RIDDIM_LIBRARY`, else samples/packs
    in the repo), or None when there is none - every caller then falls back
    to its synth."""
    root = root or os.environ.get("RIDDIM_LIBRARY") or DEFAULT_ROOT
    if root.lower() in ("off", "none", "0", ""):
        return None
    if root not in _LIB:
        _LIB[root] = Library(root) if os.path.isdir(root) else None
        if _LIB[root] is not None and not _LIB[root].files:
            _LIB[root] = None
    return _LIB[root]


def picks_report(lib: Library, seed: int = 0) -> str:
    lines = [f"library: {lib.root} ({len(lib.files)} files)"]
    for fam in PICKS:
        lines.append(f"\n{fam}:")
        for voice, pats in PICKS[fam].items():
            files = lib._draw(fam, voice, seed)
            groups = sorted({os.path.dirname(f) for f in files})
            lines.append(f"  {voice:9s} {len(files):4d} files in {len(groups)} folder(s)"
                         + (f"  e.g. {files[0]}" if files else "  (none: synth fallback)"))
    return "\n".join(lines)
