"""Drums on the Deluge: the riddim kits, the kit preset load, and the desk
moves written into kit clips (firmware branch ai-kit-automation)."""

import os
import wave
import xml.etree.ElementTree as ET

import pytest

from riddim import profile as profile_mod
from riddim.compose import KIT_ROWS, compose
from riddim.deluge import render, sysex
from riddim.units import TICKS_PER_BAR

from test_render import FakeDeluge


# ---------------------------------------------------------------- protocol

def test_row_automation_bytes():
    """SET_ROW_PARAM_AUTOMATION as the firmware parses it: row note, alias,
    shape, 14-bit timing (21-bit past 16383), then two 14-bit values."""
    msg = sysex.set_row_param_automation(37, "delay", 96, 768, 0.3, 0.7)
    body = msg[5:]
    assert body[0] == sysex.Command.SET_ROW_PARAM_AUTOMATION == 22
    assert body[1:4] == [37, sysex.PARAMS["delay"], 1]
    assert sysex.dec14(body[4], body[5]) == 96 and sysex.dec14(body[6], body[7]) == 768
    assert sysex.dec14(body[8], body[9]) == sysex.scale_14(0.3)
    assert sysex.dec14(body[10], body[11]) == sysex.scale_14(0.7)
    assert len(body) == 12
    wide = sysex.set_row_param_automation(36, "hpf", 20000, 384, 0.5)[5:]
    assert len(wide) == 14 and wide[3] == 0
    assert sysex.dec21(*wide[4:7]) == 20000
    with pytest.raises(ValueError):
        sysex.set_row_param_automation(36, "wobble", 0, 1, 0.5)


def test_new_aliases_keep_the_old_ones():
    """The first four aliases are every firmware's and must not move."""
    assert [sysex.PARAMS[k] for k in ("lpf", "resonance", "hpf", "reverb")] == [0, 1, 2, 3]
    assert sorted(sysex.PARAMS.values()) == list(range(len(sysex.PARAMS)))


def test_kit_preset_bytes():
    body = sysex.load_kit_preset("RIDDIM DUB")[5:]
    assert body[0] == sysex.Command.LOAD_CURRENT_KIT_PRESET == 23
    assert bytes(body[1:]).decode() == "RIDDIM DUB"


# ---------------------------------------------------------------- compose

def test_kit_parts_carry_the_family_kit_and_its_desk_moves():
    """Every kit part names its family's kit; the drums get the build sweep
    and the backbeat's echo running into each drop, per drum where it is a
    channel's move (the snare row) and whole-kit where it is the desk's."""
    for name in ("roots_dub", "dubstep_140", "uk_garage"):
        song = compose(name, seed=3)
        family = profile_mod.get(name).family
        kit_parts = [p for p in song.parts if p.output == "kit"]
        assert kit_parts
        for p in kit_parts:
            assert p.preset == profile_mod.KIT_PRESETS[family]
        drums = song.part("drums")
        sweeps = [a for a in drums.automation if a.param == "hpf" and a.row is None]
        echo = [a for a in drums.automation if a.row == KIT_ROWS["snare"]]
        assert sweeps, name
        assert echo, name
        for a in sweeps:
            assert a.length == 2 * TICKS_PER_BAR and a.end_value > a.start_value
            # ...and it ends where a section ends: the drop is clean.
            ends = {s.start_tick + s.length_ticks for s in song.arrangement.sections}
            assert a.start + a.length in ends


def test_dub_throws_open_the_backbeat_echo():
    for seed in range(1, 12):
        song = compose("roots_dub", seed=seed)
        throws = [s for s in song.arrangement.sections if s.dub_throw and s.plays("drums")]
        if not throws:
            continue
        drums = song.part("drums")
        for s in throws:
            assert any(a.row == KIT_ROWS["snare"] and a.param == "delay"
                       and a.start == s.start_tick and a.end_value is None
                       for a in drums.automation)
        return
    pytest.skip("no dub throw in these seeds")


# ---------------------------------------------------------------- send

def test_send_with_kits_loads_once_and_clones_from_the_loaded_clip():
    song = compose("roots_dub", seed=3)
    fake = FakeDeluge(clips=6)
    rows = {"sub": 1, "chords": 2, "lead": 3, "pad": 4, "drums": 5, "perc": 6}
    render.send(song, fake, template_rows=rows, progress=False, kits=True)
    loads = fake.commands_of(sysex.Command.LOAD_CURRENT_KIT_PRESET)
    kit_roles = [p.role for p in song.parts if p.output == "kit"]
    assert len(loads) == len(kit_roles)
    assert all(bytes(c[6:]).decode() == "RIDDIM DUB" for c in loads)
    rows_written = fake.commands_of(sysex.Command.SET_ROW_PARAM_AUTOMATION)
    assert rows_written
    # every drums clip after the first clones the first generated drums clip
    template_indices = set(render.rows_to_indices(rows, 6).values())
    clones = [src for dest, src in sorted(fake.cloned_from.items())]
    assert any(src not in template_indices for src in clones)


def test_send_without_kits_is_what_old_firmware_saw():
    song = compose("roots_dub", seed=3)
    fake = FakeDeluge()
    render.send(song, fake, progress=False)
    assert not fake.commands_of(sysex.Command.LOAD_CURRENT_KIT_PRESET)
    assert not fake.commands_of(sysex.Command.SET_ROW_PARAM_AUTOMATION)


# ---------------------------------------------------------------- the kit files

def test_knob_and_cable_encoding_match_the_firmware():
    kit = pytest.importorskip("riddim.deluge.kit")
    # getParamFromUserValue(GLOBAL_VOLUME_POST_FX, 40), and the default
    # velocity->volume cable at 50 (Sound::setupAsDefaultSynth).
    assert kit.knob(40) == "0x4CCCCCA8"
    assert kit.knob(0) == "0x80000000"
    assert kit.knob(50) == "0x7FFFFFD2"
    assert kit.cable(50) == "0x3FFFFFE8"
    assert kit.knob(kit.pan_knob(0.0)) in ("0xFFFFFFE9", "0x00000000")


def test_kit_export(tmp_path):
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    from riddim.deluge import kit
    paths = kit.export(str(tmp_path), ("dub",), progress=None)
    xml_path = tmp_path / "KITS" / "RIDDIM DUB.XML"
    assert str(xml_path) in paths
    root = ET.parse(xml_path).getroot()
    assert root.tag == "kit"
    assert root.find("defaultParams") is not None     # the kit's own params: required
    sounds = root.find("soundSources").findall("sound")
    # bottom row first, in riddim's row order
    order = [n for n, _ in sorted(KIT_ROWS.items(), key=lambda kv: kv[1])]
    assert [s.get("name") for s in sounds] == [n.upper() for n in order]
    for s in sounds:
        osc = s.find("osc1")
        f = tmp_path / osc.get("fileName")
        assert f.exists(), f
        with wave.open(str(f)) as w:
            assert w.getnchannels() == 1 and w.getframerate() == 44100
            assert int(osc.find("zone").get("endSamplePos")) == w.getnframes()
        params = s.find("defaultParams")
        for attr in ("lpfFrequency", "volume", "pan", "reverbAmount", "delayFeedback"):
            assert params.get(attr), attr        # nothing left to an undefined default
        assert params.find("envelope1") is not None
        cables = {(c.get("source"), c.get("destination")) for c in params.iter("patchCable")}
        assert ("velocity", "volume") in cables
    by = {s.get("name"): s for s in sounds}
    assert by["HAT"].find("osc1").get("loopMode") == "0"      # CUT: the note closes it
    assert by["RISER"].find("osc1").get("loopMode") == "3"    # STRETCH: fits any build
    assert by["KICK"].get("sideChainSend")
    # dub's backbeat goes to the echo
    assert by["SNARE"].find("defaultParams").get("delayFeedback") != "0x80000000"
    assert os.path.getsize(xml_path) > 1000
