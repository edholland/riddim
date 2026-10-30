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
             plate), 808-style metallic hats, rim, shaker, crash, riser, siren

With the sample library on disk (`riddim/library.py`: the free packs in
samples/packs/), recorded instruments take over where the records say they
matter: a drawbar organ for the bubble and the stabs, a guitar's skank for
dub's chords, a hardware synth's sub at 140, drum-machine hats and a shaker
in garage, and a real siren. Each is levelled to the synth it replaces, so
the mix the judge was tuned on keeps its balance; without the library every
voice is the synth above.

Needs numpy and scipy (`pip install numpy scipy`).
"""

from __future__ import annotations

import bisect
import wave

import numpy as np
from scipy.signal import butter, lfilter, oaconvolve, sosfilt

from .. import library as _library
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
SIREN = 51
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
# A dub version's melodica sits further back still (measured on six dub
# drops against 131 roots and steppers records: lead 0.40 -> 0.25 takes the
# judge's real-record probability 0.48 -> 0.68). Lower again and the tune
# stops carrying the song, whatever the judge says - this is the middle.
LEAD_LEVEL_DUB = 0.25
# The pad is the "new age" in every genre: at half level it halves.
PAD_LEVEL = 0.5
# The organ bubble (profile.pad_style "bubble"): its level against the pad's,
# and its place - opposite the skank, as the keys sit on a reggae desk.
BUBBLE_LEVEL = 1.0
# Per profile, multiplied in. Roots' bubble 4 dB down won the goal panel
# (roots_dub 8 seeds against 1bf1ba1: FAD 0.336 -> 0.332, closeness 0.822 ->
# 0.823, real 0.91 -> 0.95, still heard as steppers); a drawbar organ
# multisample (Broken Beat Kit04) and the synth organ both lost, and the skank
# +2 dB on top added nothing (FAD 0.331, real 0.95).
# steppers_dub's bubble 4 dB up moved its Demucs "other" stem toward the
# records' on every measure (16 seeds, ml/targets.py): 150-500 Hz share
# 0.316 -> 0.344 (records 0.356), 2.5-10 kHz 0.191 -> 0.174 (0.060), centroid
# 2285 -> 2226 Hz (1869), held 0.427 -> 0.440 (0.770); nothing moved away.
# With the bubble through every chords section (profile bubble_through) that
# 4 dB overshot the 150-500 Hz share (0.351 -> 0.399); back at 1.0 it holds
# it (0.346) and the held share still rises 0.520 -> 0.558.
BUBBLE_TRIM: dict[str, float] = {"roots_dub": 0.63}
BUBBLE_PAN = 0.25
# Dub's kit through a 70s desk - rolled-off top, a little tape - and its
# backbeat into the spring. Measured on five dub drops: the desk tone takes
# "hip hop beat" 0.50 -> 0.37 and "dub reggae" 0.09 -> 0.17 at the same
# real-record score; the spring is neutral (kept: it is the sound). The
# damped kick (`_dub_kick`) reads more "reggae" (0.05 -> 0.16) but its sine
# body reads as MIDI (real 0.94 -> 0.60): off until it has a recorded body.
DUB_KICK = False
# Where a pull thins the kick: the sub band goes with the bass.
PULL_KICK_HP = 90.0
DUB_KIT_TONE = True
DUB_SPRING_KIT = 0.25
# Dub's backbeat on a cross-stick (the one-drop's rimshot), not a snare.
DUB_STICK = True
# The Deluge's swing, applied as it applies it (profile.deluge_swing).
SWING = True
# Recorded instruments from the sample library where it is on disk.
LIBRARY = True
# Master: bus glue and soft clip. (A lighter master - nearer the records'
# 13-15 dB crest - measured worse with the judge: kept as it was.)
# Lighter than it was (2.5:1, +4 dB, clip 1.3): the renders' crest factor sat
# at 9 dB against the records' 12.5-16 - squashed - and now lands at 13-17.
MASTER = {"thresh": -14.0, "ratio": 2.0, "makeup": 2.0, "clip": 1.0}
# Drive into the soft clip per profile (dB), so each lands near its records'
# loudness (references/stats: RMS dubstep -11.6, uk_garage -11.9, dark_garage
# -13.0, steppers -13.2, roots -16.2). The clip normalises to its input peak,
# so a makeup gain before it would do nothing: only the drive moves the level.
# The judge is not gain-blind - 3 dB under the records cost dubstep and garage
# a third of their "real" score.
DRIVE: dict[str, float] = {"roots_dub": 2.4, "steppers_dub": 3.5, "dubstep_140": 5.5,
                            "dark_garage": 2.5, "uk_garage": 4.8}
# The bass guitar's mid an octave over the sub, per profile (dub only; the
# 140 profiles growl instead), multiplied in. Roots' at half won the goal
# panel (roots_dub 8 seeds against cb3cced, FAD / closeness / real): 1.0
# 0.332 0.823 0.95, 0.5 0.327 0.826 0.97, none 0.329 0.826 0.98; an octave
# lower 0.336 0.822 0.97. None of them is heard as roots yet, and the probes'
# "a piano" hardly moves (+1.7 to +1.8 sd): it is not the bass guitar.
BASS_MID: dict[str, float] = {"roots_dub": 0.5}
# The kit's character per profile, measured on Demucs drum stems against
# the records' (ml/drum_stem.py): "snare_body" scales the synthesised body
# under the snare (and under dub's cross-stick), "snare_hp" high-passes the
# whole sampled snare (Hz), "hat_gain" the hats.
# steppers_dub: snare body x0.3 and hats x1.35 moved both toward the records
# (16 seeds: snare body +8.3 -> +6.7 dB against 2-8 kHz, records +0.4; hat
# share 0.058 -> 0.064, records 0.082). The same in dubstep left its snare
# body alone (it is the sampled snare's) and pushed the hats' spread,
# brightness and decay away; in dark and uk garage the louder hats measured
# lower in the Demucs drum stem - so those stay as they are.
# uk_garage: the sampled snare high-passed at 350 Hz and its synthesised body
# x0.3, with the tone's sub 1.5 dB lower (the thinner snare had left the mix
# bassier): snare body +6.0 -> +3.9 dB (records -2.8), snare share 0.43 ->
# 0.31 (0.28), kick share 0.50 -> 0.63 (0.58), sub share 0.357 -> 0.345
# (0.246). The kick's onset-strength spread moved away (0.67 -> 0.73,
# records 0.61), an advisory target (ml/targets.py).
DRUM_SHAPE: dict[str, dict[str, float]] = {"steppers_dub": {"snare_body": 0.3, "hat_gain": 1.35},
                                           "uk_garage": {"snare_body": 0.3, "snare_hp": 350.0}}
# A low-pass on the chords per profile (Hz), for an "other" stem brighter
# than the records'.
CHORDS_LP: dict[str, float] = {}
# ... and the tone EQ's five bands on the chords alone (dB), per profile.
# An organ holding each chord under the dub stabs, at this gain, per profile
# (the records' keys sound far more of the time than ours). Roots, 16 seeds
# (ml/targets.py): the "other" stem sounds 0.548 -> 0.708 of the time
# (records 0.768); its 150-500 Hz share 0.237 -> 0.278 (0.384), 500-2500
# 0.553 -> 0.529 (0.476), 2.5-10k 0.210 -> 0.193 (0.087), all toward. The
# mix's lean moved away (-2.2 -> -3.2 ms), the drums-only bus's not at all.
# A 150 ms swell (HELD_ORGAN_SWELL) measured the same.
HELD_ORGAN: dict[str, float] = {"roots_dub": 1.0}
HELD_ORGAN_SWELL = 0.0     # seconds
# The chords' high-pass, per profile (220 Hz otherwise). uk_garage's organ at
# 120 Hz keeps its body (16 seeds, ml/targets.py): the "other" stem's 150-500
# Hz share 0.278 -> 0.288 (records 0.402), 500-2500 0.703 -> 0.693 (0.502),
# nothing away. On its own, the organ's bus already measures ~0.40 there; in
# the mix Demucs hears less of it as "other".
CHORDS_HP: dict[str, float] = {"uk_garage": 120.0}
# The organ stab's top note doubled an octave up, at this gain, per profile:
# lifts the chords' top end. Not kept for uk_garage (16 seeds, ml/targets.py):
# at 1.0 the "other" stem's 2.5-10 kHz share rose 0.019 -> 0.039 (records
# 0.061) but its 150-500 Hz share fell 0.278 -> 0.244 (0.402), and +4 dB at
# 500 Hz on the chords made that worse; 0.5 moved nothing. Voicing the chords
# up a fifth or an octave thinned the body the same way.
CHORDS_TOP_OCTAVE: dict[str, float] = {}
CHORDS_TONE: dict[str, dict[str, float]] = {}
# The reverb and delay returns per profile, multiplied in. The judge's text
# probes heard our garage and 140 as "a dry close mix" (+1.7-1.8 sd against
# the records). uk_garage x2.5 won the goal panel (garage FAD 0.392 -> 0.370,
# real 0.59 -> 0.64; x4 was worse). dark_garage x2.5 scored a little better
# but stopped being heard as dark garage (2/4 -> 0/4), and dubstep_140 x1.6
# lost, so those stay dry; roots x0.6 lost on both counts.
SPACE: dict[str, float] = {"uk_garage": 2.5}
# A drummer's timing per profile: each kit hit off the grid by a normal
# spread of this many ms. Roots records measure 14.3 ms of onset jitter
# against our 11.0 (the detector's own floor on a grid-tight kit), so ~8.5 ms
# more, added in quadrature, lands on the records' figure.
# The others, from the same gap measured on the head panel (16 seeds,
# ml/targets.py; jitter ms ours -> with this, records): steppers 10.8 -> 11.3
# (13.0), dubstep 9.5 -> 11.6 (13.9), dark garage 8.8 -> 11.1 (13.6), uk
# garage 11.9 -> 12.5 (13.7); no other target moved away beyond tolerance.
DRUMMER_MS: dict[str, float] = {"roots_dub": 8.5, "steppers_dub": 7.4, "dubstep_140": 10.2,
                                "dark_garage": 10.4, "uk_garage": 7.2}
# ... and where it sits against the beat: roots records lean -0.4 ms, ours
# -3.7, so the kit lays back by the difference.
# Lean ms, ours -> with this (records): steppers -6.9 -> -5.2 (-4.1), dubstep
# -8.6 -> -6.1 (-1.8), dark garage -9.6 -> -7.1 (-2.6), uk garage -6.2 -> -4.4
# (-2.6). The detector counts every onset, not only the kit's, so a lay-back
# shows at about a third of its size.
DRUMMER_LEAN_MS: dict[str, float] = {"roots_dub": 3.3, "steppers_dub": 3.9, "dubstep_140": 6.9,
                                     "dark_garage": 7.2, "uk_garage": 3.8}


# The engineer's hand on the master fader, per profile: how much of each
# bar's level swing (in dB, against the song's median bar) is ridden out.
# Roots records ride within about 2 dB block to block (p90-p10 of the 8-bar
# energy curve 2.0 dB, ours 4.8) - the section ladder's intros, dub-outs and
# peaks are holes and lifts in the band, not in the level.
# Off: on the goal panel (roots_dub 8 seeds against 314a667, FAD / closeness /
# real, spread / dynamic range) 0 scored 0.332 0.823 0.95 (4.8 / 6.9 dB), 0.3
# 0.334 0.824 0.95 (3.4 / 5.5), 0.5 0.335 0.824 0.95 (2.5 / 4.7), 0.7 0.333
# 0.825 0.96 (1.6 / 4.0): the spread lands on the records' but the judge
# does not move (still 0/8 heard as roots; "a big drop" stays +1.5 sd).
RIDE: dict[str, float] = {}


def _ride(x: np.ndarray, amount: float, bar_s: float) -> np.ndarray:
    """Pull each bar's level `amount` of the way toward the median bar's (in
    dB, at most 6 dB either way), the gain moving smoothly across a bar.
    Silent bars are left alone."""
    n = int(bar_s * SR)
    if n <= 0 or len(x) < 2 * n:
        return x
    bars = len(x) // n
    rms = np.sqrt(np.mean(x[: bars * n].reshape(bars, n, -1) ** 2, axis=(1, 2)) + 1e-12)
    live = rms > 1e-4
    if not live.any():
        return x
    db = 20 * np.log10(rms)
    ref = float(np.median(db[live]))
    gain_db = np.where(live, np.clip((ref - db) * amount, -6.0, 6.0), 0.0)
    centres = (np.arange(bars) + 0.5) * n
    g = 10 ** (np.interp(np.arange(len(x)), centres, gain_db) / 20.0)
    return (x * g[:, None]).astype(np.float32)


def _swing_ticks(tick: float, amount: int) -> float:
    """A Deluge-style swing as a time warp inside each eighth: the off-beat
    sixteenth lands `amount`% of the way through the eighth (50 straight),
    everything between moved in proportion, the beats untouched."""
    if amount == 50:
        return float(tick)
    eighth = 2 * TICKS_PER_16TH
    base = (tick // eighth) * eighth
    pos = tick - base
    mid = eighth * amount / 100.0
    half = eighth / 2.0
    if pos <= half:
        return base + pos * mid / half
    return base + mid + (pos - half) * (eighth - mid) / half

# Per-part stereo position (-1 left .. 1 right) and level.
PAN = {"sub": 0.0, "bass": 0.0, "drums": 0.0, "perc": 0.45, "chords": -0.3, "pad": 0.0, "lead": 0.12}
GAIN = {"sub": 0.45, "bass": 0.42, "drums": 0.75, "perc": 0.5, "chords": 0.5, "pad": 0.32,
        "lead": 0.42}
# The mix bus's tone per profile, in dB on five bands (sub < 60 Hz, bass to
# 250, low mid to 1k, high mid to 5k, air above): set so each profile's band
# shares land on its records' (references/stats, ~300 records a family, the
# same five bands as riddim.listen). Measured on the goal panel, every profile
# came out ~95% below 250 Hz, against 75-85% for the records - the judge heard
# nearly all of them as steppers, the darkest references.
TONE: dict[str, dict[str, float]] = {
    # Roots' sub a further 2 dB down won the goal panel (8 seeds against
    # 08b30d3: FAD 0.327 -> 0.324, closeness 0.826 -> 0.827, real 0.97); a
    # 70s colour on the master - top -3 dB, 40% width, wow, more tape - lost
    # (FAD 0.333 to 0.343), and a mono kit did nothing.
    "roots_dub": {"sub": -13.5, "low_mid": 3.3, "high_mid": 5.0, "air": 6.5},
    "steppers_dub": {"sub": -8.0, "low_mid": 2.2, "high_mid": 1.2, "air": 5.0},
    "dubstep_140": {"bass": -1.5, "low_mid": 2.7, "high_mid": 10.0, "air": 8.0},
    "dark_garage": {"sub": -3.0, "low_mid": -1.3, "high_mid": 9.5, "air": 7.0},
    "uk_garage": {"sub": -4.7, "low_mid": 0.2, "high_mid": 7.7, "air": 7.0},
}
_TONE_BANDS = (("low", 60.0), ("peak", 130.0), ("peak", 500.0), ("peak", 2200.0), ("high", 6000.0))
_TONE_NAMES = ("sub", "bass", "low_mid", "high_mid", "air")


def _biquad(kind: str, f0: float, db: float, q: float = 0.7) -> np.ndarray:
    """An RBJ shelf or peak as one second-order section."""
    a = 10 ** (db / 40)
    w = 2 * np.pi * f0 / SR
    cw, alpha = np.cos(w), np.sin(w) / (2 * q)
    if kind == "peak":
        b = [1 + alpha * a, -2 * cw, 1 - alpha * a]
        d = [1 + alpha / a, -2 * cw, 1 - alpha / a]
    else:
        sa = 2 * np.sqrt(a) * alpha
        sign = 1 if kind == "low" else -1
        b = [a * ((a + 1) - sign * (a - 1) * cw + sa), sign * 2 * a * ((a - 1) - sign * (a + 1) * cw),
             a * ((a + 1) - sign * (a - 1) * cw - sa)]
        d = [(a + 1) + sign * (a - 1) * cw + sa, -sign * 2 * ((a - 1) + sign * (a + 1) * cw),
             (a + 1) + sign * (a - 1) * cw - sa]
    return np.array([[b[0] / d[0], b[1] / d[0], b[2] / d[0], 1.0, d[1] / d[0], d[2] / d[0]]])


def _tone(x: np.ndarray, gains: dict[str, float]) -> np.ndarray:
    sos = [_biquad(k, f, gains[name]) for (k, f), name in zip(_TONE_BANDS, _TONE_NAMES)
           if gains.get(name)]
    return sosfilt(np.concatenate(sos), x, axis=0).astype(np.float32) if sos else x


# Per-family trims on GAIN, multiplied in: each family's balance measured
# against its own records (Demucs stems, `claude/loop-rounds.md`).
# Dub's kit sat ~4 dB over its records in the stems, and 2 dB down won the
# goal panel (dub FAD 0.313 -> 0.306, real 0.96 -> 0.94), but Ed preferred
# it at full level in the blind A/B (so did the clip judge): left as it was.
# Garage's chords down 4.4 dB, with uk_garage's shorter stabs (profile
# skank_length): they read as horns and melodica held up front.
FAMILY_GAIN: dict[str, dict[str, float]] = {"garage": {"chords": 0.6}}


def parse_gains(text: str) -> dict[str, dict[str, float]]:
    """"dub.drums=0.7,dubstep.sub=1.2" -> {"dub": {"drums": 0.7}, ...}: trims
    to audition (`RIDDIM_FAMILY_GAIN`, ml/library_ab.py --gain)."""
    out: dict[str, dict[str, float]] = {}
    for item in filter(None, (t.strip() for t in (text or "").split(","))):
        key, val = item.split("=")
        fam, role = key.split(".")
        out.setdefault(fam, {})[role] = float(val)
    return out
# Sends: (plate, spring, hall, delay)
SENDS = {"drums": (0.0, 0.0, 0.05, 0.0), "chords": (0.0, 0.35, 0.1, 0.5),
         "pad": (0.0, 0.0, 0.45, 0.0), "lead": (0.0, 0.0, 0.3, 0.3),
         "perc": (0.0, 0.08, 0.2, 0.06), "bass": (0.0, 0.0, 0.0, 0.0), "sub": (0.0, 0.0, 0.0, 0.0)}
# Per profile, multiplying a role's sends. The records' "other" stem (Demucs:
# keys, guitars, their tails) sounds 72-92% of the time, ours 39-43% in the
# dub and 140 profiles - stabs with gaps between. steppers_dub's chords
# sends x2 (16 seeds, ml/targets.py): held 0.438 -> 0.473 (records 0.770),
# centroid 2230 -> 2196 Hz (1869); nothing moved away. (All the returns x1.6
# also lifted it, but put the snare's tail in the drum stem.) The same for
# roots (held 0.394 -> 0.436, records 0.768), dubstep (0.436 -> 0.519, 0.747)
# and dark garage (0.438 -> 0.554, 0.902), nothing away in any.
SEND_TRIM: dict[str, dict[str, float]] = {p: {"chords": 2.0} for p in
                                          ("steppers_dub", "roots_dub", "dubstep_140", "dark_garage")}


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


def _dub_kick(dur, vel, rnd):
    """A reggae kick: the acoustic drum, damped - a short round thump
    under a felt beater, no 808 boom and no click (the sub is the low end)."""
    v = _vel(vel)
    rate = 0.86 + rnd.jitter(0.02)
    top = _lp(_hp(_play("pearl/kick-01", rate), 50), 2200)
    n = int(0.22 * SR)
    t = np.arange(n) / SR
    freq = 58 + 40 * np.exp(-t / 0.02)
    body = np.sin(2 * np.pi * np.cumsum(freq) / SR) * np.exp(-t / 0.09)
    m = max(len(top), n)
    out = np.zeros(m, dtype=np.float32)
    out[:n] += body.astype(np.float32) * 0.7
    out[:len(top)] += top * 1.5
    return (_sat(out, 1.3) * v).astype(np.float32)


def _drum(row, dur, vel, rnd, garage=False, sampled=True, shape=None):
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
            body = _drum(SNARE, dur, vel, rnd, garage, sampled=False) * 0.35 * (shape or {}).get("snare_body", 1.0)
            m = max(len(hit), len(body))
            out = np.zeros(m, dtype=np.float32)
            out[:len(hit)] += hit
            out[:len(body)] += body
            if shape and shape.get("snare_hp"):
                out = _hp(out, shape["snare_hp"]).astype(np.float32)
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


def _siren(dur, vel, rnd):
    """The dub siren: a square wave whose pitch an LFO throws up and down,
    a little faster as it goes, through a speaker's band."""
    n = int((max(dur, 0.5) + 0.2) * SR)
    t = np.arange(n) / SR
    rate = 3.2 * (1 + 0.6 * t / t[-1]) + rnd.jitter(0.3)
    lfo = 2.0 * np.abs(((np.cumsum(rate) / SR) % 1.0) - 0.5)        # triangle 0..1
    f = 520.0 * 2 ** (1.1 * lfo)
    x = 0.6 * _sq(_phase(f, n, rnd.jitter(0.5))) + 0.4 * np.sin(2 * np.pi * _phase(f, n))
    x = _bp(x.astype(np.float32), 400, 3500)
    return (_sat(x, 1.5) * _env(n, 0.02, 0.2) * _vel(vel) * 0.35).astype(np.float32)


def _rms(x) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0


def _level(ref, x) -> float:
    """The gain that puts a recorded instrument at the level of the synth
    it replaces (measured on one reference note)."""
    a, b = _rms(ref), _rms(x)
    return float(np.clip(a / b, 0.05, 20.0)) if a > 0 and b > 0 else 1.0


_MINOR_SET = (0, 3, 7, 10, 2, 5)
_MAJOR_SET = (0, 4, 7, 11, 2, 9)


def chord_name(midis, bass_pc=None) -> tuple[int, str]:
    """A stab's (root pitch class, "maj"/"min") - read with the bass, since
    the stabs are voiced rootless over it: the sounding bass note is the
    root if the stab agrees, else the triad the stab best spells."""
    pcs = {m % 12 for m in midis}
    best, key = None, None
    for r in range(12):
        for q, third, other, ext in (("min", 3, 4, _MINOR_SET), ("maj", 4, 3, _MAJOR_SET)):
            triad = {r, (r + third) % 12, (r + 7) % 12}
            score = (2 * len(triad & pcs) - len(pcs - {(r + e) % 12 for e in ext})
                     - 3 * (((r + other) % 12) in pcs) + (2.5 if r == bass_pc else 0.0))
            if best is None or score > best:
                best, key = score, (r, q)
    return key


# Where each kit voice sits: hats and shaker off centre, the rest in the middle.
DRUM_PAN = {HAT: 0.28, SHAKER: -0.4, PERC: 0.4, RIM: -0.18, CRASH: 0.0, RISER: 0.0, SIREN: 0.1}


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

def instruments(family: str, seed: int, bpm: float, is140: bool, garage: bool) -> dict:
    """The recorded instruments this song plays from the sample library (an
    empty dict without one), each with the gain that levels it to the synth
    it replaces."""
    lib = _library.get() if LIBRARY else None
    if lib is None or not family:
        return {}
    r = _Rand(1)
    out: dict = {}
    org = lib.organ(family, seed)
    if org is not None:
        out["organ"] = org
        out["organ_k"] = _level(_organ(60, 0.25, 100, r), org.play(60, 0.25, 100).mean(axis=1))
    if family == "dub":
        sk = lib.skank(family, seed, bpm)
        if sk is not None:
            ref = sum(_pluck(m, 0.15, 100, r)[: int(0.3 * SR)] for m in (69, 72, 76))
            out["skank"] = sk
            out["skank_k"] = _level(ref, sk.play(9, "min", 0.15, 100).mean(axis=1))
    if is140:
        sb = lib.sub(family, seed)
        if sb is not None:
            out["sub"] = sb
            out["sub_k"] = _level(_sub(36, 1.0, 100, r),
                                  sb.play(36, 1.0, 100, 0.035, loop=True).mean(axis=1))
    if garage:
        for voice, ref in (("hat", _drum(HAT, 0.1, 100, r, True)),
                           ("hat_open", _drum(HAT, 0.4, 100, r, True)),
                           ("shaker", _drum(SHAKER, 0.1, 100, r, True))):
            sh = lib.shots(family, voice, seed)
            if sh is not None:
                take = sh.play(np.random.default_rng(0), 100, 0.4 if voice == "hat_open" else 0.12)
                out[voice] = sh
                out[voice + "_k"] = _level(ref, _hp(take.mean(axis=1), 300))
    siren = lib.siren(family, seed)
    if siren is not None:
        out["siren"] = siren
        out["siren_k"] = _level(_siren(2.0, 100, r), _library.load(siren).mean(axis=1)[: int(2.2 * SR)])
    return out


def kit_hit(inst: dict, row: int, dur: float, vel: int, rnd, garage: bool = False,
            shape: dict | None = None) -> np.ndarray:
    """A kit voice, recorded where the library has it (garage's hats and
    shaker, the siren), else as `_drum` makes it."""
    if row == SIREN:
        if inst.get("siren"):
            x = _library.load(inst["siren"]).mean(axis=1)
            x = x[: int((max(dur, 0.5) + 1.0) * SR)].copy()
            r = min(len(x), int(0.8 * SR))
            x[len(x) - r:] *= np.linspace(1.0, 0.0, r, dtype=np.float32)
            return (x * inst["siren_k"] * _vel(vel)).astype(np.float32)
        return _siren(dur, vel, rnd)
    if row == HAT and inst.get("hat") is not None:
        voice = "hat_open" if (dur > 0.2 and inst.get("hat_open") is not None) else "hat"
        x = inst[voice].play(rnd.rng, vel, min(dur, 0.45) if voice == "hat_open" else 0.12)
        return (_hp(x.mean(axis=1), 300) * inst[voice + "_k"]).astype(np.float32)
    if row == SHAKER and inst.get("shaker") is not None:
        x = inst["shaker"].play(rnd.rng, vel, 0.12)
        return (_hp(x.mean(axis=1), 300) * inst["shaker_k"]).astype(np.float32)
    return _drum(row, dur, vel, rnd, garage, shape=shape)


def render(song: Song, window: tuple[int, int] | None = None, stems_out: dict | None = None) -> np.ndarray:
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
    from .. import profile as _profile_mod
    try:
        swing = _profile_mod.get(prof).deluge_swing if (SWING and prof) else 50
    except Exception:      # noqa: BLE001 - an unknown profile renders straight
        swing = 50

    try:
        family = _profile_mod.get(prof).family if prof else ""
    except Exception:      # noqa: BLE001
        family = ""
    cross = DUB_STICK and family == "dub"
    try:
        bubble = _profile_mod.get(prof).pad_style == "bubble" if prof else False
    except Exception:      # noqa: BLE001
        bubble = False
    inst = instruments(family, song.seed or 0, song.bpm, is140, garage)
    shape = DRUM_SHAPE.get(prof or "", {})

    def kit(midi, dur, vel):
        """A kit hit as this genre plays it: in dub the one-drop's backbeat
        is a cross-stick rimshot, not a snare."""
        if cross and midi in (SNARE, RIM):
            name = "hands/stick_m" if midi == SNARE else "hands/stick_h"
            x = _play(name, 1.0 + rnd.jitter(0.02), 2.2 * _vel(vel))
            if midi == SNARE:
                body = _drum(SNARE, dur, vel, rnd, False) * 0.3 * shape.get("snare_body", 1.0)
                m = max(len(x), len(body))
                out = np.zeros(m, dtype=np.float32)
                out[:len(x)] += x
                out[:len(body)] += body
                return out
            return x
        if cross and midi == KICK and DUB_KICK:
            return _dub_kick(dur, vel, rnd)
        return kit_hit(inst, midi, dur, vel, rnd, garage, shape)

    def at(tick):
        """Sample index of a tick, swung as the Deluge would play it."""
        return int((_swing_ticks(tick, swing) - w0) * spt * SR)

    def organ(midi, dur, vel):
        if inst.get("organ") is not None:
            return inst["organ"].play(midi, dur, vel, 0.06).mean(axis=1) * inst["organ_k"]
        return _organ(midi, dur, vel, rnd)

    # The stabs as chords, and the bass under them, for the guitar's skank.
    stab_notes: dict[int, list[int]] = {}
    skanked: set[int] = set()
    if song.part("chords") is not None:
        for nt in song.notes_with_coda(song.part("chords")):
            stab_notes.setdefault(nt.start, []).append(nt.midi)
    _bass = sorted((nt.start, nt.midi, nt.length) for nt in song.notes_with_coda(song.part("sub"))) \
        if song.part("sub") is not None else []

    def bass_pc(tick):
        """The bar's bass root: the pitch the bass holds longest in the bar
        (a reply or an approach note mid-bar is not the chord), else the
        last one before it."""
        a = tick - tick % TICKS_PER_BAR
        i, j = bisect.bisect_left(_bass, (a,)), bisect.bisect_left(_bass, (a + TICKS_PER_BAR,))
        held: dict[int, int] = {}
        for st, m, ln in _bass[i:j]:
            held[m % 12] = held.get(m % 12, 0) + ln
        if held:
            return max(held, key=held.get)
        return _bass[i - 1][1] % 12 if i > 0 else None

    stems = {r: np.zeros((n, 2), dtype=np.float32) for r in PAN}
    kick_env = np.zeros(n, dtype=np.float32)
    beat = 60.0 / song.bpm
    # Its own stream, so the hits' timing moves nothing else.
    feel_ms = DRUMMER_MS.get(prof or "", 0.0)
    lean_ms = DRUMMER_LEAN_MS.get(prof or "", 0.0)
    trng = np.random.default_rng((song.seed or 0) + 7919)

    for part in song.parts:
        role = part.role
        buf = stems.setdefault(role, np.zeros((n, 2), dtype=np.float32))
        prev = None
        # The desk's pulls: kick hits under a held high-pass on the kick row
        # lose their sub (compose._thin_kick_in_pulls).
        thin = [(a.start, a.start + a.length) for a in part.automation
                if a.param == "hpf" and a.row == KICK and a.end_value == a.start_value]
        for note in song.notes_with_coda(part):
            if window is not None and not (w0 <= note.start < window[1]):
                continue
            start = at(note.start)
            dur = note.length * spt
            if part.output == "kit" and note.midi in HANDS:
                name, p, g = HANDS[note.midi]
                # A hand: the take a hair off in pitch and level every stroke.
                x = _play(name, 1.0 + rnd.jitter(0.02), g * _vel(note.velocity) * 1.4)
                _add(buf, start, _pan(x, p + rnd.jitter(0.05)))
                continue
            if part.output == "kit":
                if feel_ms or lean_ms:
                    start = max(0, start + int((lean_ms + trng.normal(0.0, feel_ms)) / 1000.0 * SR))
                x = kit(note.midi, dur, note.velocity)
                if note.midi == HAT and shape.get("hat_gain"):
                    x = x * shape["hat_gain"]
                if note.midi == KICK and any(a <= note.start < b for a, b in thin):
                    x = _hp(x, PULL_KICK_HP, order=4)
                p = DRUM_PAN.get(note.midi, 0.0)
                if note.midi == HAT:
                    p += rnd.jitter(0.08)
                _add(buf, start, _pan(x, p))
                if note.midi == KICK:
                    k = int(0.3 * SR)
                    _add(kick_env, start, np.exp(-np.arange(k) / (0.1 * SR)).astype(np.float32))
                continue
            if role == "sub":
                if inst.get("sub") is not None:
                    x = inst["sub"].play(note.midi, dur, note.velocity, 0.035, loop=True)
                    x = x.mean(axis=1) * inst["sub_k"]
                else:
                    x = _sub(note.midi, dur, note.velocity, rnd)
                _add(buf, start, _pan(x, 0.0))
                # The bass's mid: a growl at 140, a pluck in dub.
                # (An M1 organ-bass pluck for uk_garage lost the goal panel
                # badly - garage FAD 0.391 -> 0.414, real 0.58 -> 0.17 - so
                # garage keeps the growl.)
                if is140:
                    rate = 2.0 / beat if note.length * spt > beat else 1.0 / beat
                    mb = _wobble(note.midi, dur, note.velocity, rnd, rate)
                    if prof == "dark_garage":
                        # Dark garage is the deep sub with a little growl over
                        # it; the judge heard a dubstep wobble in it (+1.5 sd).
                        mb = mb * 0.5
                else:
                    mb = _bass_guitar(note.midi + 12, dur, note.velocity, rnd) * BASS_MID.get(prof or "", 1.0)
                _add(stems["bass"], start, _pan(mb, 0.0))
                continue
            if role == "chords":
                if (garage or is140):
                    x = organ(note.midi, dur, note.velocity)
                    top = CHORDS_TOP_OCTAVE.get(prof or "", 0.0)
                    if top and note.midi == max(stab_notes.get(note.start, [note.midi])):
                        up = organ(note.midi + 12, dur, note.velocity) * top
                        m = max(len(x), len(up))
                        x = np.pad(x, (0, m - len(x))) + np.pad(up, (0, m - len(up)))
                elif inst.get("skank") is not None:
                    # The guitar plays the whole stab at once: one chop per
                    # chord, the stab's other notes are in it.
                    if note.start in skanked:
                        continue
                    skanked.add(note.start)
                    root, q = chord_name(stab_notes[note.start], bass_pc(note.start))
                    x = inst["skank"].play(root, q, dur, note.velocity).mean(axis=1) * inst["skank_k"]
                else:
                    x = _pluck(note.midi, dur, note.velocity, rnd)
                _add(buf, start, _pan(x, PAN["chords"] + rnd.jitter(0.05)))
                continue
            if role == "pad":
                if bubble:
                    # The organ bubble: drawbars, opposite the skank.
                    x = organ(note.midi, dur, note.velocity)
                    _add(buf, start, _pan(x, BUBBLE_PAN + rnd.jitter(0.04)))
                else:
                    _add(buf, start, _strings(note.midi, dur, note.velocity, rnd))
                continue
            glide = prev.midi if (prev is not None and prev.end > note.start) else None
            voice = _synth_lead if is140 else _reed
            x = voice(note.midi, dur, note.velocity, rnd, glide)
            _add(buf, start, _pan(x, PAN["lead"]))
            prev = note

    # The kick pumps the sub (a little), the bass, the stabs and the pad.
    kick = np.clip(kick_env, 0, 1)
    for r, depth in (("sub", 0.25), ("bass", 0.5), ("chords", 0.45), ("pad", 0.6)):
        stems[r] *= (1.0 - depth * kick)[:, None]

    # Clear the mud: only the kick and the sub own the bottom.
    for r, hp in (("chords", CHORDS_HP.get(prof or "", 220)), ("pad", 160), ("lead", 200), ("perc", 120), ("bass", 70)):
        stems[r] = _hp(stems[r], hp)
    if CHORDS_LP.get(prof or ""):
        stems["chords"] = _lp(stems["chords"], CHORDS_LP[prof])
    if CHORDS_TONE.get(prof or ""):
        stems["chords"] = _tone(stems["chords"], CHORDS_TONE[prof])
    stems["lead"] = stems["lead"] * (LEAD_LEVEL_DUB if family == "dub" else LEAD_LEVEL)
    stems["pad"] = stems["pad"] * (BUBBLE_LEVEL * BUBBLE_TRIM.get(prof or "", 1.0) if bubble else PAD_LEVEL)
    held = HELD_ORGAN.get(prof or "", 0.0)
    if held and stab_notes:
        # The organ holding each chord under the stabs, from its first stab
        # to the next chord (or a bar past its last stab).
        ticks = sorted(stab_notes)
        hbuf = np.zeros((n, 2), dtype=np.float32)
        i = 0
        while i < len(ticks):
            chord = sorted(set(stab_notes[ticks[i]]))
            j = i
            while j + 1 < len(ticks) and sorted(set(stab_notes[ticks[j + 1]])) == chord \
                    and ticks[j + 1] - ticks[j] <= TICKS_PER_BAR:
                j += 1
            nxt = ticks[j + 1] if j + 1 < len(ticks) else None
            end = nxt if nxt is not None and nxt - ticks[j] <= TICKS_PER_BAR else ticks[j] + TICKS_PER_BAR
            a = ticks[i]
            if window is None or w0 <= a < window[1]:
                dur = (end - a) * spt
                for midi in chord:
                    x = organ(midi, dur, 70)
                    # Swelled in on the volume pedal, not struck: the stabs
                    # keep the attack.
                    k = min(len(x), int(HELD_ORGAN_SWELL * SR))
                    x[:k] = x[:k] * np.linspace(0.0, 1.0, k, dtype=np.float32)
                    _add(hbuf, at(a), _pan(x, 0.0) / len(chord))
            i = j + 1
        stems["pad"] = stems["pad"] + _hp(hbuf, 160) * held
    sends = dict(SENDS)
    for role, k in SEND_TRIM.get(prof or "", {}).items():
        sends[role] = tuple(v * k for v in sends[role])
    if TREAT_LEAD:
        stems["lead"] = _sat(_lp(_hp(stems["lead"], 380, order=4), 3000, order=4) * 1.6, 1.8) * 0.45
        sends["lead"] = (0.0, 0.12, 0.35, 0.6)
    if TREAT_PAD and not bubble:
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
            at_ = at(note.start)
            if note.midi in (SNARE, RIM):
                _add(sn, at_, kit(note.midi, note.length * spt, note.velocity)
                     * (1.0 if note.midi == SNARE else 0.4))
            elif note.midi in (RISER, CRASH, SIREN):
                _add(fx, at_, kit_hit(inst, note.midi, note.length * spt, note.velocity, rnd)
                     * (1.6 if note.midi == SIREN else 1.0))
        for perc_part in [p for p in song.parts if p.role == "perc"]:
            for note in song.notes_with_coda(perc_part):
                if note.midi in (49, 50) and (window is None or w0 <= note.start < window[1]):
                    _add(fx, at(note.start),
                         _play(HANDS[note.midi][0], 1.0, 0.6 * _vel(note.velocity)))
        plate_send += np.stack([sn, sn], axis=1)
        if cross and DUB_SPRING_KIT:
            spring_send += np.stack([sn, sn], axis=1) * DUB_SPRING_KIT
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
    if cross and DUB_KIT_TONE:
        d = _lp(stems["drums"], 7000)
        k = 1.0 / (np.max(np.abs(d)) or 1.0)
        # Tape on the kit: peaks rounded, small-signal level unchanged.
        stems["drums"] = _sat(d * k, 1.4) * (np.tanh(1.4) / (1.4 * k))
    d = int(3 * TICKS_PER_16TH * spt * SR)
    echo = _dub_delay(_hp(delay_send, 250), d, fb)

    music = np.zeros((n, 2), dtype=np.float32)
    trim = {**FAMILY_GAIN.get(family, {}),
            **parse_gains(__import__("os").environ.get("RIDDIM_FAMILY_GAIN", "")).get(family, {})}
    for role, x in stems.items():
        music += x * GAIN.get(role, 0.2) * trim.get(role, 1.0)
    space = SPACE.get(prof or "", 1.0)
    music += (plate * 0.18 + spring * 0.12 + hall * 0.2 + echo * 0.32) * space
    if stems_out is not None:
        # The band grouped as Demucs splits a record - drums, bass, other -
        # at their level in the mix, through the profile's tone, for the
        # critic's per-stem judges (ml/critic). The returns go with "other",
        # where most of what feeds them sits: an approximation.
        tone = TONE.get(prof or "", TONE.get(family, {}))
        groups = {"drums": ("drums", "perc"), "bass": ("sub", "bass")}
        taken = {r for rs in groups.values() for r in rs}
        groups["other"] = tuple(r for r in stems if r not in taken)
        for g, roles in groups.items():
            y = np.zeros((n, 2), dtype=np.float32)
            for r in roles:
                if r in stems:
                    y += stems[r] * GAIN.get(r, 0.2) * trim.get(r, 1.0)
            if g == "other":
                y += (plate * 0.18 + spring * 0.12 + hall * 0.2 + echo * 0.32) * space
            stems_out[g] = _tone(y, tone)[lead_in:].astype(np.float32)

    # The build's sweep thins everything but the riser and the sub's absence.
    if sweep.any() and SWEEP:
        thin = _hp(music, 900)
        music = music + (thin - music) * (sweep * 0.85)[:, None]

    music = _tone(music, TONE.get(prof or "", TONE.get(family, {})))
    if RIDE.get(prof or ""):
        music = _ride(music, RIDE[prof], 4 * beat)

    # Nothing playing (every part muted): silence. The master would otherwise
    # normalise its own tape hiss up to full scale.
    if not np.any(music[lead_in:]):
        return np.zeros_like(music[lead_in:], dtype=np.float32)

    # Master: glue, tape, a soft ceiling.
    mix = _hp(music, 24)
    mix /= (np.max(np.abs(mix)) or 1.0)
    mix = _compress(mix, MASTER["thresh"], MASTER["ratio"], attack=0.015, release=0.2,
                    makeup_db=MASTER["makeup"])
    mix = _tape(mix, rnd)
    c = MASTER["clip"] * 10 ** (DRIVE.get(prof or "", 0.0) / 20)
    mix = np.tanh(c * mix / (np.max(np.abs(mix)) or 1.0)) / np.tanh(c)
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
