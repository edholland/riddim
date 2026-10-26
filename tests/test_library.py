"""The sample library: what a file is from its path, multisamples that play
in tune whatever their pack calls the octave, the guitar's skank, the siren
row, and the preview and the kits playing recorded sounds where the packs
have them (a small fake library, made here, stands in for samples/packs)."""

import os
import wave
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from riddim import library
from riddim.compose import KIT_ROWS, compose
from riddim.profile import Profile
from riddim.units import TICKS_PER_BAR

SR = 44100


def _write(path, x, sr=SR):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def _tone(freq, seconds, harmonics=(1.0,)):
    t = np.arange(int(seconds * SR)) / SR
    x = sum(a * np.sin(2 * np.pi * freq * (k + 1) * t) for k, a in enumerate(harmonics))
    return 0.5 * x / max(1.0, sum(harmonics)) * np.minimum(1, t / 0.005)


def _hz(midi):
    return 440.0 * 2 ** ((midi - 69) / 12.0)


NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


@pytest.fixture(scope="module")
def packs(tmp_path_factory):
    """Paths shaped like the real packs', with audio that is what it says."""
    root = str(tmp_path_factory.mktemp("packs"))
    keys = os.path.join(root, "musicradar-soulful-keys-samples", "Soulful Keys", "Multisamples")
    for org in ("Organ02", "Organ03", "Organ04"):
        for name, midi in (("C2", 48), ("F#2", 54), ("C3", 60), ("F#3", 66), ("C4", 72)):
            _write(os.path.join(keys, org, f"SK_{org}_{name}.wav"),
                   _tone(_hz(midi), 1.0, (0.6, 1.0, 0.5, 0.3)))
    # A sub pack that names its notes an octave higher than riddim does:
    # "C2" is 32.7 Hz.
    subs = os.path.join(root, "musicradar-sub-bass-samples", "Sub Bass Weight", "Multis")
    for kind in ("Soft Sub", "Firm Sub"):
        for name, midi in (("C2", 24), ("F#2", 30), ("C3", 36), ("F#3", 42)):
            _write(os.path.join(subs, kind, f"{kind} {name}.wav"), _tone(_hz(midi), 0.6))
    # Guitar chops: a loop of four stabs at 85 bpm, each a minor or major triad.
    chops = os.path.join(root, "musicradar-dub-designer-samples", "85bpm", "Guitar Chops")
    beat = 60 / 85
    for pc in range(0, 12, 2):
        for q, third in (("min", 3), ("maj", 4)):
            x = np.zeros(int(4 * beat * SR))
            stab = sum(_tone(_hz(60 + pc + i), 0.12) for i in (0, third, 7))
            for b in range(4):
                i = int((b + 0.5) * beat * SR)
                x[i:i + len(stab)] += stab
            _write(os.path.join(chops, f"DD_TeleChop_85-{NAMES[pc]}{q}.wav"), x)
    hats = os.path.join(root, "musicradar-hats-cymbals-gongs-sample-pack", "Hats, Cymbals and Gongs",
                        "Hats", "Drum machine")
    rng = np.random.default_rng(0)
    for i in range(1, 4):
        _write(os.path.join(hats, f"Closed hat {i:02d}.wav"), rng.uniform(-0.3, 0.3, 4000)
               * np.exp(-np.arange(4000) / 600))
        _write(os.path.join(hats, f"Open hat {i:02d}.wav"), rng.uniform(-0.3, 0.3, 20000)
               * np.exp(-np.arange(20000) / 6000))
    for bpm in (100, 140):
        sirens = os.path.join(root, "musicradar-alarm-siren-samples", "musicradar-alarm-siren-samples",
                              f"{bpm}bpm")
        for kind in ("EchoProcA", "VerbProcA", "FilterProcB", "ModuProcA"):
            t = np.arange(3 * SR) / SR
            _write(os.path.join(sirens, f"AaS_{kind}_{bpm}-01.wav"),
                   0.4 * np.sin(2 * np.pi * np.cumsum(700 + 300 * np.sin(2 * np.pi * 4 * t)) / SR))
    with open(os.path.join(root, "LICENCES.tsv"), "w") as f:
        f.write("musicradar-alarm-siren-samples.zip\tSampleRadar sirens\troyalty-free\thttp://x\n")
    return root


@pytest.fixture
def with_library(packs, monkeypatch):
    monkeypatch.setenv("RIDDIM_LIBRARY", packs)
    library._LIB.clear()
    yield library.get()
    library._LIB.clear()


# ---------------------------------------------------------------- paths

def test_what_a_file_is_from_its_path():
    assert library.role_of("musicradar-alarm-siren-samples/x/140bpm/AaS_EchoProcA_140-01.wav") == "siren"
    assert library.role_of("musicradar-ska-samples/x/Minor/Organ/Organ130mn01-C1.wav") == "organ"
    assert library.role_of("musicradar-dub-designer-samples/x/85bpm/Guitar Chops/DD_TeleChop_85-Amin.wav") \
        == "skank"
    assert library.role_of("p/Hats, Cymbals and Gongs/Hats/Drum machine/Open hat 01.wav") == "hat_open"
    assert library.role_of("p/Hats, Cymbals and Gongs/Hats/Drum machine/Closed hat 01.wav") == "hat"
    # A pack named for its vocals still has a drum kit in it.
    assert library.role_of("rhythmlab-acapellas4u/acapellas4u_hiphop_pack/drumkit/hihat05.wav") == "hat"
    assert library.role_of("musicradar-808-samples/x/Hits/Bass Drum [BD]/E808_BD[long]-01.wav") == "kick"


def test_notes_chords_and_tempi_from_names():
    assert library.note_of("SK_Organ01_C3.wav") == 60
    assert library.note_of("SK_Organ01_D#4.wav") == 75
    assert library.note_of("Solid Sub F#1.wav") == 42
    assert library.note_of("Organ130mn01-Bb2.wav") == 58
    assert library.note_of("Closed hat 01.wav") is None
    assert library.chord_of("DD_TeleChop_85-Amin.wav") == (9, "min")
    assert library.chord_of("DD_StratChop_105-Cmaj.wav") == (0, "maj")
    assert library.bpm_of("x/140bpm/AaS_EchoProcA_140-01.wav") == 140
    assert library.bpm_of("Hats 01 (120BPM).wav") == 120
    assert library.bpm_of("Closed hat 01.wav") is None


def test_index(packs, tmp_path):
    entries = library.scan(packs, analyse=True)
    roles = {e.role for e in entries}
    assert {"organ", "sub", "skank", "hat", "hat_open", "siren"} <= roles
    siren = next(e for e in entries if e.role == "siren" and "sirens" in "sirens")
    assert siren.seconds == pytest.approx(3.0, abs=0.01) and siren.peak_db < 0
    assert next(e for e in entries if e.pack == "musicradar-alarm-siren-samples").licence == "royalty-free"
    out = library.write_index(entries, str(tmp_path / "index.jsonl"))
    assert sum(1 for _ in open(out)) == len(entries)


# ---------------------------------------------------------------- playing

def test_pitch_is_measured_exactly():
    for midi in (24, 36, 45, 60):
        f = library.f0(np.stack([_tone(_hz(midi), 0.8)] * 2, axis=1).astype(np.float32))
        assert 12 * abs(np.log2(f / _hz(midi))) < 0.1


def test_a_sub_pack_an_octave_off_plays_in_tune(with_library):
    """The fake sub pack calls 32.7 Hz "C2": the zones are moved down an
    octave by their pitch, and C1 (riddim's 36) sounds at 65.4 Hz."""
    sub = with_library.sub("dubstep", 0)
    assert min(sub.zones) == 24
    y = sub.play(36, 1.0, 100, loop=True)
    assert len(y) >= int(1.0 * SR)
    f = library.f0(y)
    assert 12 * abs(np.log2(f / _hz(36))) < 0.3


def test_held_notes_loop_to_length(with_library):
    sub = with_library.sub("dubstep", 0)
    y = sub.play(30, 3.0, 100, release=0.05, loop=True)
    assert len(y) == int(3.05 * SR)
    rms = [float(np.sqrt(np.mean(y[i:i + 4410] ** 2))) for i in range(0, int(2.9 * SR), 4410)]
    assert min(rms) > 0.3 * max(rms)        # still sounding at the end of the hold


def test_skank_is_one_stab_cut_from_the_loop(with_library):
    sk = with_library.skank("dub", 0, 72)
    assert len(sk.stabs) == 12
    y = sk.play(9, "min", 0.2, 100)
    assert 0.08 * SR < len(y) < 0.6 * SR
    # A minor from the nearest minor stab: A# is not a stab, so it is played
    # a semitone from one.
    assert sk.play(10, "min", 0.2, 100).shape[1] == 2


def test_chord_names_read_with_the_bass():
    from riddim.render.audio import chord_name
    assert chord_name([69, 72, 76]) == (9, "min")
    assert chord_name([72, 76, 79]) == (0, "maj")
    # Rootless: C E G over an A in the bass is A minor 7, not C major.
    assert chord_name([72, 76, 79], bass_pc=9) == (9, "min")


def test_off_and_missing_mean_no_library(tmp_path, monkeypatch):
    library._LIB.clear()
    monkeypatch.setenv("RIDDIM_LIBRARY", "off")
    assert library.get() is None
    monkeypatch.setenv("RIDDIM_LIBRARY", str(tmp_path / "nowhere"))
    assert library.get() is None
    library._LIB.clear()


# ---------------------------------------------------------------- the siren

def test_sirens_fire_in_dub_outs_and_later_drops_only():
    from riddim.arrange import dynamics
    found = 0
    for seed in range(1, 9):
        song = compose("steppers_dub", seed=seed)
        drops = [s for i, s in enumerate(song.arrangement.sections)
                 if i and dynamics.drop_entry(song.arrangement.sections[i - 1], s)]
        for n in song.part("drums").notes:
            if n.midi != KIT_ROWS["siren"]:
                continue
            found += 1
            sec = next(s for s in song.arrangement.sections
                       if s.start_tick <= n.start < s.start_tick + s.length_ticks)
            assert sec.dub_throw or (sec in drops and sec is not drops[0])
            assert n.length == 2 * TICKS_PER_BAR
    assert found >= 4


def test_no_sirens_where_the_profile_has_none(monkeypatch):
    from riddim import profile
    quiet = profile.get("steppers_dub")
    monkeypatch.setitem(profile.PROFILES, "steppers_dub",
                        Profile(**{**quiet.__dict__, "sirens": 0.0}))
    for seed in range(1, 5):
        song = compose("steppers_dub", seed=seed)
        assert not any(n.midi == KIT_ROWS["siren"] for n in song.part("drums").notes)


# ---------------------------------------------------------------- render and kits

@pytest.mark.parametrize("prof", ["roots_dub", "dubstep_140", "uk_garage"])
def test_the_preview_plays_the_library(with_library, prof):
    from riddim.render import audio
    from riddim import profile
    song = compose(prof, seed=3)
    fam = profile.get(prof).family
    inst = audio.instruments(fam, song.seed, song.bpm, song.bpm >= 120, "garage" in prof)
    assert "organ" in inst and "siren" in inst
    if fam == "dub":
        assert "skank" in inst
    if fam == "dubstep":
        assert "sub" in inst
    if fam == "garage":
        assert {"hat", "hat_open"} <= set(inst)
    for k, v in inst.items():
        if k.endswith("_k"):
            assert 0.05 <= v <= 20
    drop = next(s for s in song.arrangement.sections if s.energy >= 0.8)
    x = audio.render(song, (drop.start_tick, drop.start_tick + 4 * TICKS_PER_BAR))
    assert np.all(np.isfinite(x)) and np.max(np.abs(x)) > 0.1


def test_without_the_library_nothing_changes(monkeypatch):
    from riddim.render import audio
    library._LIB.clear()
    assert audio.instruments("dub", 1, 72, False, False) == {}


def test_kits_have_a_siren_row(with_library, tmp_path):
    from riddim.deluge import kit
    paths = kit.export(str(tmp_path), ("dub", "garage"), progress=None)
    root = ET.parse(str(tmp_path / "KITS" / "RIDDIM DUB.XML")).getroot()
    sounds = root.find("soundSources").findall("sound")
    assert len(sounds) == len(KIT_ROWS) == 16
    assert sounds[-1].get("name") == "SIREN"
    assert any(p.endswith("51 SIREN.WAV") for p in paths)
    with wave.open(str(tmp_path / "SAMPLES" / "RIDDIM" / "DUB" / "51 SIREN.WAV")) as w:
        assert w.getnframes() > SR


def test_kits_without_the_library_get_the_synth_siren(tmp_path):
    from riddim.deluge import kit
    library._LIB.clear()
    rows = kit.render_rows("dub")
    siren = next(r for r in rows if r.note == KIT_ROWS["siren"])
    assert len(siren.sample) > SR // 2 and float(np.max(np.abs(siren.sample))) > 0.05
