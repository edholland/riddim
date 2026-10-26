"""Deluge kits made from the preview's own drums.

The audio preview's kit - the recorded Pearl drums layered over a
synthesised body, the LinnDrum cross-stick, the hand percussion, a little
drum room on every hit, and for dub the 70s desk - exists only in the
preview. This puts it on the Deluge: every kit row rendered to a one-shot
WAV, and a kit preset (KITS/*.XML) that plays them, with the parts the Deluge
can do live left live - each row's level, pan, reverb and tape-synced delay
send, a velocity-to-level cable, and a hair of random pitch per hit for the
round-robin the Deluge does not have.

One kit per family, every kit the same fifteen rows in riddim's order (row 36
at the bottom, KIT_ROWS), so any kit clip plays any riddim part:

    36 kick   37 snare  38 hat    39 rim    40 perc   41 shaker 42 crash
    43 riser  44 conga lo 45 conga hi 46 bongo 47 block 48 tamb 49 tom lo 50 tom hi
    51 siren

With the sample library on disk (riddim/library.py) the kits carry what
the preview plays from it: a recorded siren on 51, and in the garage kit a
drum machine's hats and shaker. Without it those rows are the preview's
synths, as before.

    riddim-cli deluge-kits /Volumes/DELUGE      # the card, or any folder

writes SAMPLES/RIDDIM/<KIT>/*.WAV and KITS/RIDDIM <KIT>.XML. Load the kit on a
template's kit clips once, or let `send` load it per song (firmware
LOAD_CURRENT_KIT_PRESET, branch ai-kit-automation).

The XML follows the firmware's own writer (Kit::writeDataToFile,
Sound::writeToFile): every param a sample row needs is written explicitly,
because a sound's params that are absent from the file are not all given
sane defaults on load (the LPF cutoff among them).
"""

from __future__ import annotations

import os
import wave
from dataclasses import dataclass, field
from xml.sax.saxutils import quoteattr

import numpy as np

from ..compose import KIT_ROWS
from ..profile import KIT_PRESETS

FIRMWARE_VERSION = "c1.3.0"
EARLIEST_COMPATIBLE = "4.1.0-alpha"

# Families -> kit names on the card.
KITS = KIT_PRESETS

# Deluge loop modes (SampleRepeatMode).
CUT, ONCE, LOOP, STRETCH = 0, 1, 2, 3

# Delay sync (sync.h): an eighth, dotted - the preview's dub delay (three
# sixteenths) on the Deluge's tape-style delay.
SYNC_8TH, SYNC_DOTTED = 4, 19


def knob(u: float) -> str:
    """A Deluge knob position (0..50, as the screen shows it) as the hex
    param value the files store (getParamFromUserValue's default case)."""
    u = max(0.0, min(50.0, float(u)))
    v = int(round(u * 85899345)) - 2147483648
    return f"0x{v & 0xFFFFFFFF:08X}"


def cable(u: float) -> str:
    """A patch cable amount (-50..50) as stored."""
    v = int(round(u * 21474836))
    return f"0x{v & 0xFFFFFFFF:08X}"


def pan_knob(p: float) -> float:
    """Pan -1..1 onto the 0..50 knob (25 centre)."""
    return 25.0 + 25.0 * max(-1.0, min(1.0, p))


@dataclass
class Row:
    """One kit row: its one-shot and how the Deluge plays it."""
    note: int
    name: str
    sample: np.ndarray = field(repr=False)
    loop_mode: int = ONCE
    pan: float = 0.0           # -1..1
    volume: float = 40.0       # knob 0..50 (40 is the Deluge's default)
    reverb: float = 0.0        # knob 0..50: send to the song's reverb
    delay: float = 0.0         # knob 0..50: this row's own delay amount
    release: float = 0.0       # knob 0..50: envelope release (CUT rows)
    # Cable amount for the hair of pitch per hit. Pitch cables are squared and
    # a full one spans three octaves (PatchCableSet::getModifiedPatchCableAmount),
    # so 5 is about a quarter of a semitone either way - the preview's 2.5%.
    random_pitch: float = 5.0
    sidechain: bool = False    # this row pumps the song's sidechain


@dataclass
class Kit:
    family: str
    name: str
    rows: list[Row]

    @property
    def folder(self) -> str:
        return "SAMPLES/RIDDIM/" + self.name.replace("RIDDIM ", "")

    def sample_path(self, row: Row) -> str:
        return f"{self.folder}/{row.note} {row.name.upper()}.WAV"


# --------------------------------------------------------------------------
# The one-shots
# --------------------------------------------------------------------------

def _trim(x: np.ndarray, floor_db: float = -60.0, fade: float = 0.01) -> np.ndarray:
    """Cut the silence after the hit, with a short fade so it never clicks."""
    from ..render.audio import SR
    peak = float(np.max(np.abs(x))) or 1.0
    above = np.nonzero(np.abs(x) > peak * 10 ** (floor_db / 20))[0]
    end = int(above[-1]) + 1 if len(above) else len(x)
    x = x[:max(end, int(0.02 * SR))].copy()
    n = min(len(x), int(fade * SR))
    x[len(x) - n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32)
    return x


def render_rows(family: str, seed: int = 1) -> list[Row]:
    """Every row's one-shot, made by the preview's own voices, with the
    preview's drum room baked in and - for dub - its desk (the top rolled
    off, a little tape). Levels relative to one another as the preview mixes
    them; the whole kit normalised together."""
    from ..render import audio as A

    rnd = A._Rand(seed)
    garage = family == "garage"
    dub = family == "dub"
    inst = A.instruments(family, seed, 140.0 if family != "dub" else 72.0,
                         family == "dubstep", garage)
    vel = 112
    room_ir = A._norm_ir(A._sample("ir/nice_drum_room"))

    def kit_voice(note: int) -> np.ndarray:
        if dub and note == A.SNARE:
            # The one-drop's backbeat: a cross-stick with a little snare body.
            x = A._play("hands/stick_m", 1.0, 2.2 * A._vel(vel))
            body = A._drum(A.SNARE, 0.3, vel, rnd, False) * 0.3
            out = np.zeros(max(len(x), len(body)), dtype=np.float32)
            out[:len(x)] += x
            out[:len(body)] += body
            return out
        if dub and note == A.RIM:
            return A._play("hands/stick_h", 1.0, 2.2 * A._vel(vel))
        if note == A.HAT:
            # The open hat, whole: CUT mode lets the note length close it.
            if inst.get("hat_open") is not None:
                x = inst["hat_open"].play(rnd.rng, vel).mean(axis=1)
                return (A._hp(x, 300) * inst["hat_open_k"]).astype(np.float32)
            x = A._hp(A._play("pearl/hihat-open", 1.0), 2500)
            return (x * 1.5 * A._vel(vel)).astype(np.float32)
        if note == A.RISER:
            return A._drum(A.RISER, 4.0, vel, rnd, garage)
        if note == A.SIREN:
            return A.kit_hit(inst, A.SIREN, 3.0, vel, rnd, garage)
        return A.kit_hit(inst, note, 0.3, vel, rnd, garage)

    def hand(note: int) -> np.ndarray:
        name, _pan, g = A.HANDS[note]
        return A._play(name, 1.0, g * A._vel(vel) * 1.4)

    def room(x: np.ndarray) -> np.ndarray:
        # The preview's drum stem: the hit plus 0.22 of a real drum room.
        wet = A._convolve(A._hp(x, 120), room_ir).mean(axis=1)
        pad = np.zeros(len(x) + int(0.35 * A.SR), dtype=np.float32)
        pad[:len(x)] += x
        out = pad.copy()
        out[:len(wet)] += 0.22 * wet[:len(pad)]
        return out

    def desk(x: np.ndarray) -> np.ndarray:
        # The 70s desk on dub's kit (audio.DUB_KIT_TONE): rolled-off top, tape.
        d = A._lp(x, 7000)
        k = 1.0 / (float(np.max(np.abs(d))) or 1.0)
        return (A._sat(d * k, 1.4) * (np.tanh(1.4) / (1.4 * k))).astype(np.float32)

    gain_drums = A.GAIN.get("drums", 0.75)
    gain_perc = A.GAIN.get("perc", 0.5)
    rows: list[Row] = []
    for name, note in sorted(KIT_ROWS.items(), key=lambda kv: kv[1]):
        if note in A.HANDS:
            x = hand(note) * (gain_perc / gain_drums)
            pan = A.HANDS[note][1]
        elif note == A.SIREN:
            # The siren as the preview throws it: into the hall and the echo,
            # not the drum room - those stay live on the Deluge (`_sends`).
            x = kit_voice(note) * 1.6
            pan = A.DRUM_PAN[A.SIREN]
        else:
            x = room(kit_voice(note))
            if dub and A.DUB_KIT_TONE:
                x = desk(x)
            if note in (A.PERC, A.SHAKER):
                x = x * (gain_perc / gain_drums)
            pan = A.DRUM_PAN.get(note, 0.0)
        mode = {A.HAT: CUT, A.RISER: STRETCH}.get(note, ONCE)
        rows.append(Row(note=note, name=name, sample=_trim(np.asarray(x, dtype=np.float32)),
                        loop_mode=mode, pan=pan,
                        release=4.0 if mode == CUT else 0.0,
                        random_pitch=0.0 if note in (A.RISER, A.SIREN) else 5.0,
                        sidechain=note == A.KICK))

    peak = max(float(np.max(np.abs(r.sample))) for r in rows) or 1.0
    for r in rows:
        r.sample = (r.sample * (0.89 / peak)).astype(np.float32)
    _sends(family, rows)
    return rows


def _sends(family: str, rows: list[Row]) -> None:
    """What stays live on the Deluge: the sends the preview's desk makes.
    The backbeat into the reverb (the plate at 140, the spring in dub) and
    into the dotted-eighth delay - lightly: a Deluge delay knob is its
    feedback, so the send is a few repeats, not a wash; toms and the crash into the reverb; dub's
    stick and congas into the echo."""
    by = {r.note: r for r in rows}
    snare, rim = by[KIT_ROWS["snare"]], by[KIT_ROWS["rim"]]
    if family == "dub":
        snare.reverb, snare.delay = 14.0, 7.0
        rim.reverb, rim.delay = 10.0, 4.0
        by[KIT_ROWS["conga_hi"]].delay = 3.0
    elif family == "dubstep":
        snare.reverb, snare.delay = 20.0, 4.0
        rim.reverb = 10.0
    else:
        snare.reverb = 10.0
        rim.reverb = 6.0
    for name in ("crash", "riser", "tom_lo", "tom_hi"):
        by[KIT_ROWS[name]].reverb = max(by[KIT_ROWS[name]].reverb, 12.0)
    siren = by[KIT_ROWS["siren"]]
    siren.reverb, siren.delay = 16.0, (8.0 if family == "dub" else 5.0)


def build(family: str, seed: int = 1) -> Kit:
    if family not in KITS:
        raise ValueError(f"unknown family {family!r}; known: {', '.join(KITS)}")
    return Kit(family=family, name=KITS[family], rows=render_rows(family, seed))


# --------------------------------------------------------------------------
# The preset
# --------------------------------------------------------------------------

def _attrs(**kv) -> str:
    return " ".join(f"{k}={quoteattr(str(v))}" for k, v in kv.items())


def _sound_xml(kit: Kit, row: Row) -> str:
    n = len(row.sample)
    sound = {"name": row.name.upper(), "polyphonic": "auto", "voicePriority": 1,
             "mode": "subtractive"}
    if row.sidechain:
        sound["sideChainSend"] = 2147483647
    sound.update({"modFXType": "none", "lpfMode": "24dB", "hpfMode": "HPLadder",
                  "filterRoute": "H2L", "maxVoices": 8})
    params = {
        "oscAVolume": "0x7FFFFFFF", "oscBVolume": "0x80000000", "noiseVolume": "0x80000000",
        "volume": knob(row.volume), "pan": knob(pan_knob(row.pan)),
        "lpfFrequency": "0x7FFFFFFF", "lpfResonance": "0x80000000",
        "hpfFrequency": "0x80000000", "hpfResonance": "0x80000000",
        "pitchAdjust": "0x00000000", "oscAPitchAdjust": "0x00000000", "oscBPitchAdjust": "0x00000000",
        "modFXRate": "0x00000000", "modFXDepth": "0x00000000",
        "delayRate": "0x00000000", "delayFeedback": knob(row.delay),
        "reverbAmount": knob(row.reverb),
        "stutterRate": "0x00000000", "sampleRateReduction": "0x80000000", "bitCrush": "0x80000000",
    }
    env = {"attack": knob(0), "decay": knob(20), "sustain": knob(50), "release": knob(row.release)}
    cables = [("velocity", "volume", cable(50))]
    if row.random_pitch:
        cables.append(("random", "pitch", cable(row.random_pitch)))
    lines = [
        f"\t\t<sound {_attrs(**sound)}>",
        f"\t\t\t<osc1 {_attrs(type='sample', loopMode=row.loop_mode, reversed=0, timeStretchEnable=0, timeStretchAmount=0, fileName=kit.sample_path(row))}>",
        f"\t\t\t\t<zone {_attrs(startSamplePos=0, endSamplePos=n)} />",
        "\t\t\t</osc1>",
        f"\t\t\t<osc2 {_attrs(type='sample', loopMode=0, reversed=0, timeStretchEnable=0, timeStretchAmount=0)}>",
        "\t\t\t</osc2>",
        f"\t\t\t<delay {_attrs(pingPong=0, analog=1, syncLevel=SYNC_8TH, syncType=SYNC_DOTTED)} />",
        f"\t\t\t<defaultParams {_attrs(**params)}>",
        f"\t\t\t\t<envelope1 {_attrs(**env)} />",
        "\t\t\t\t<patchCables>",
    ]
    for src, dst, amt in cables:
        lines.append(f"\t\t\t\t\t<patchCable {_attrs(source=src, destination=dst, amount=amt)} />")
    lines += [
        "\t\t\t\t</patchCables>",
        f"\t\t\t\t<equalizer {_attrs(bass='0x00000000', treble='0x00000000', bassFrequency='0x00000000', trebleFrequency='0x00000000')} />",
        "\t\t\t</defaultParams>",
        "\t\t</sound>",
    ]
    return "\n".join(lines)


def kit_xml(kit: Kit) -> str:
    """The kit preset. Sounds bottom row first: the firmware gives the first
    drum the bottom row, and the AI SysEx puts note 36 there."""
    head = _attrs(firmwareVersion=FIRMWARE_VERSION, earliestCompatibleFirmware=EARLIEST_COMPATIBLE,
                  modFXCurrentParam="feedback", currentFilterType="lpf", modFXType="none",
                  lpfMode="24dB", hpfMode="HPLadder", filterRoute="H2L")
    kit_params = _attrs(reverbAmount="0x80000000", volume="0x00000000", pan="0x00000000",
                        sidechainCompressorShape="0xDC28F5B2", modFXDepth="0x00000000",
                        modFXRate="0xE0000000", stutterRate="0x00000000",
                        sampleRateReduction="0x80000000", bitCrush="0x80000000",
                        modFXOffset="0x00000000", modFXFeedback="0x80000000")
    out = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f"<kit {head}>",
        f"\t<defaultParams {kit_params}>",
        f"\t\t<delay {_attrs(rate='0x00000000', feedback='0x80000000')} />",
        f"\t\t<lpf {_attrs(frequency='0x7FFFFFFF', resonance='0x80000000')} />",
        f"\t\t<hpf {_attrs(frequency='0x80000000', resonance='0x80000000')} />",
        f"\t\t<equalizer {_attrs(bass='0x00000000', treble='0x00000000', bassFrequency='0x00000000', trebleFrequency='0x00000000')} />",
        "\t</defaultParams>",
        f"\t<delay {_attrs(pingPong=1, analog=0, syncLevel=7, syncType=0)} />",
        "\t<soundSources>",
    ]
    out += [_sound_xml(kit, r) for r in sorted(kit.rows, key=lambda r: r.note)]
    out += ["\t</soundSources>", "\t<selectedDrumIndex>0</selectedDrumIndex>", "</kit>", ""]
    return "\n".join(out)


# --------------------------------------------------------------------------
# Onto the card
# --------------------------------------------------------------------------

def _write_wav(path: str, x: np.ndarray, sr: int = 44100) -> None:
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def export(root: str, families=tuple(KITS), seed: int = 1, progress=print) -> list[str]:
    """Write each family's samples and kit preset under `root` (an SD card's
    top level, or a folder to copy onto one). Returns the paths written."""
    written = []
    for fam in families:
        kit = build(fam, seed)
        folder = os.path.join(root, *kit.folder.split("/"))
        os.makedirs(folder, exist_ok=True)
        for row in kit.rows:
            path = os.path.join(root, *kit.sample_path(row).split("/"))
            _write_wav(path, row.sample)
            written.append(path)
        os.makedirs(os.path.join(root, "KITS"), exist_ok=True)
        xml_path = os.path.join(root, "KITS", kit.name + ".XML")
        with open(xml_path, "w", newline="\n") as f:
            f.write(kit_xml(kit))
        written.append(xml_path)
        if progress:
            progress(f"{kit.name}: {len(kit.rows)} rows -> {kit.folder}, KITS/{kit.name}.XML")
    return written
